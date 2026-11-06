#!/usr/bin/env python3
"""
cancer-ch / Stage 0: data audit.

Pulls every public cancer table from the BFS DAM catalog (NKRS incidence +
BFS mortality), plus PxWeb metadata for mortality / population cubes, and
reports which axes actually exist: period granularity, canton, sex, age, ICD-10.

Nothing here is modelled yet. The point is to answer one question first:
what grid (year x region x sex x age x site) do we really have?

Run:  python cancer_audit.py            (fetch + audit)
      python cancer_audit.py --audit    (audit only, reuse downloaded files)
"""
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------- config ----
BASE_DIR = Path(__file__).resolve().parent.parent / "data"   # ../data next to the repo, or set your own
RAW_BFS = BASE_DIR / "raw" / "bfs_dam"
RAW_PX = BASE_DIR / "raw" / "pxweb_meta"
OUT_DIR = BASE_DIR / "audit"

DAM_API = "https://dam-api.bfs.admin.ch/hub/api/dam/assets"
PXWEB_API = "https://www.pxweb.bfs.admin.ch/api/v1/de"

# catalog search terms (extendedSearch hits title, orderNr, summary)
DAM_QUERIES = [("de", "Krebs"), ("en", "cancer"), ("fr", "cancer")]
DAM_TITLE_FILTER = re.compile(r"krebs|cancer|tumeur|tumor", re.I)
DAM_LIMIT = 1000

# seeds verified by hand 2026-11-03 (always fetched, even if search misses)
SEED_ASSETS = {
    36138837: "EN: new cases & deaths by cancer site and age group, 2018-2022",
    36138836: "DE: Alle Krebsarten, 2018-2022",
}

# PxWeb cubes worth inspecting: deaths by cause, population (STATPOP)
PX_TITLE_FILTER = re.compile(r"todesursach|sterbef|todesf|krebs|bevölkerung", re.I)
PX_DBID_FILTER = r"px-x-(14|0102)"
PX_SLEEP = 0.4                                     # BFS PxWeb ~30 req / 10 s

PROXIES = None                                      # e.g. {"https": "http://proxy:3128"}
TIMEOUT = 60
UA = "cancer-ch-audit/0.1 (research, aggregated public data)"

# ------------------------------------------------------------ constants ----
CANTONS = ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS",
           "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE",
           "GE", "JU"]
CANTON_NAMES = ["Zürich", "Bern", "Luzern", "Uri", "Schwyz", "Obwalden", "Nidwalden",
                "Glarus", "Zug", "Fribourg", "Freiburg", "Solothurn", "Basel-Stadt",
                "Basel-Landschaft", "Schaffhausen", "Appenzell", "St. Gallen",
                "Graubünden", "Aargau", "Thurgau", "Ticino", "Tessin", "Vaud", "Waadt",
                "Valais", "Wallis", "Neuchâtel", "Neuenburg", "Genève", "Genf", "Jura"]
REGION_TERMS = re.compile(r"deutschschweiz|westschweiz|suisse romande|suisse alémanique|"
                          r"german[- ]speaking|french[- ]|latin|sprachregion|"
                          r"region linguistique|language region", re.I)
RE_PERIOD = re.compile(r"\b((?:19|20)\d{2})\s*[-–/]\s*((?:19|20)\d{2})\b")
RE_YEAR = re.compile(r"^\s*((?:19|20)\d{2})(?:\.0)?\s*$")
RE_AGE = re.compile(r"^\s*(\d{1,2})\s*[-–]\s*(\d{1,2})\s*$|^\s*(\d{2})\s*\+\s*$|"
                    r"^\s*(\d{2})\s*(?:und mehr|et plus|and over|plus)\s*$", re.I)
RE_ICD = re.compile(r"\bC\d{2}(?:\.\d)?(?:\s*[-–]\s*C?\d{2})?\b")
RE_SEX = re.compile(r"^\s*(männer|frauen|männlich|weiblich|hommes|femmes|men|women|"
                    r"male|female|mann|frau)\s*$", re.I)


FAILED = []          # assets without a downloadable master file


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json"})
    if PROXIES:
        s.proxies.update(PROXIES)
    return s


# ================================================================ FETCH =====
def dam_search(s):
    """Query DAM catalog, return de-duplicated list of cancer-related assets."""
    rows = {}
    for lang, term in DAM_QUERIES:
        data, seen, skip = [], set(), 0
        while skip < 5000:                      # API caps a page (~350); paginate
            try:
                r = s.get(DAM_API, params={"language": lang, "extendedSearch": term,
                                           "limit": DAM_LIMIT, "skip": skip},
                          headers={"Accept-Language": lang}, timeout=TIMEOUT)
                r.raise_for_status()
                page = r.json().get("data", [])
            except Exception as e:
                log(f"DAM search {lang}/{term} skip={skip} failed: {e}")
                break
            new = [a for a in page if (a.get("ids") or {}).get("damId") not in seen]
            if not new:                         # end, or API ignores skip
                break
            seen |= {(a.get("ids") or {}).get("damId") for a in new}
            data += new
            skip += len(page)
        log(f"DAM search {lang}/{term}: {len(data)} hits")
        for a in data:
            dam_id = (a.get("ids") or {}).get("damId")
            title = ((a.get("description") or {}).get("titles") or {}).get("main", "")
            if not dam_id or not DAM_TITLE_FILTER.search(title or ""):
                continue
            rows[dam_id] = {
                "damId": dam_id,
                "orderNr": (a.get("shop") or {}).get("orderNr"),
                "title": title,
                "lang": lang,
                "embargo": (a.get("bfs") or {}).get("embargo"),
                "format": (a.get("description") or {}).get("masterFormat")
                          or (a.get("shop") or {}).get("format"),
            }
    return list(rows.values())


def pick_latest(catalog):
    """One asset per orderNr: latest embargo. orderNr encodes language (su-d / su-e)."""
    df = pd.DataFrame(catalog)
    if df.empty:
        return df
    df["embargo_dt"] = pd.to_datetime(df["embargo"], errors="coerce", utc=True)
    # su-d-14.03... / su-e-14.03... / su-f-14.03... are the same table
    df["key"] = (df["orderNr"].fillna(df["damId"].astype(str))
                 .str.replace(r"^(\w{2})-[defi]-", r"\1-", regex=True))
    # prefer EN, then DE, then FR for the same table across languages
    df["lang_rank"] = df["lang"].map({"en": 0, "de": 1, "fr": 2}).fillna(9)
    df = df.sort_values(["key", "embargo_dt", "lang_rank"], ascending=[True, False, True])
    return df.drop_duplicates("key")


def download_asset(s, dam_id, label=""):
    RAW_BFS.mkdir(parents=True, exist_ok=True)
    existing = list(RAW_BFS.glob(f"{dam_id}.*"))
    if existing:
        return existing[0]
    url = f"{DAM_API}/{dam_id}/master"
    try:
        r = s.get(url, timeout=TIMEOUT, headers={"Accept": "*/*"})
        r.raise_for_status()
    except Exception as e:
        code = getattr(getattr(e, "response", None), "status_code", "err")
        FAILED.append({"damId": dam_id, "status": code, "title": label})
        log(f"  download {dam_id} failed: {code}")
        return None
    ctype = r.headers.get("Content-Type", "")
    ext = ("xlsx" if "spreadsheet" in ctype or r.content[:2] == b"PK"
           else "pdf" if "pdf" in ctype else "csv" if "csv" in ctype
           else "px" if "px" in ctype else "bin")
    path = RAW_BFS / f"{dam_id}.{ext}"
    path.write_bytes(r.content)
    log(f"  {dam_id}.{ext}  {len(r.content)/1024:.0f} kB  {label[:70]}")
    return path


def px_discover(s):
    """Walk PxWeb root, keep cubes whose title matches PX_TITLE_FILTER, save metadata."""
    RAW_PX.mkdir(parents=True, exist_ok=True)
    try:
        dbs = s.get(f"{PXWEB_API}/", timeout=TIMEOUT).json()
    except Exception as e:
        log(f"PxWeb root failed: {e}")
        return []
    log(f"PxWeb: {len(dbs)} databases")
    (OUT_DIR / "pxweb_root.json").write_text(json.dumps(dbs, ensure_ascii=False, indent=1))
    # root "text" may be just the id, so also match by BFS theme in the dbid:
    # px-x-14xx = health (causes of death), px-x-0102 = population (STATPOP)
    matched = [d for d in dbs
               if PX_TITLE_FILTER.search(d.get("text", ""))
               or re.match(PX_DBID_FILTER, d.get("dbid") or d.get("id") or "")]
    log(f"PxWeb: {len(matched)} databases match title filter")
    found, types_seen = [], defaultdict(int)

    def walk(path, title, depth=0):
        """PxWeb v1: a node lists items of type 'l' (folder) or 't' (table)."""
        if depth > 3:
            return
        time.sleep(PX_SLEEP)
        try:
            items = s.get(f"{PXWEB_API}/{path}", timeout=TIMEOUT).json()
        except Exception as e:
            log(f"  {path}: list failed: {e}")
            return
        if isinstance(items, dict) and "variables" in items:   # path already is a table
            save(path, items, title)
            return
        for it in items if isinstance(items, list) else []:
            typ, iid = it.get("type"), it.get("id")
            types_seen[typ] += 1
            if not iid:
                continue
            if typ == "l":
                walk(f"{path}/{iid}", it.get("text", title), depth + 1)
            elif typ == "t":
                out = RAW_PX / f"{iid}.json"
                if not out.exists():
                    time.sleep(PX_SLEEP)
                    try:
                        meta = s.get(f"{PXWEB_API}/{path}/{iid}", timeout=TIMEOUT).json()
                    except Exception as e:
                        log(f"  {iid}: meta failed: {e}")
                        continue
                    save(f"{path}/{iid}", meta, it.get("text", title))
                else:
                    found.append(out)

    def save(path, meta, title):
        name = path.split("/")[-1]
        out = RAW_PX / f"{name}.json"
        meta["_path"], meta["_text"] = path, title
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        found.append(out)
        log(f"  px {name}: {(meta.get('title') or title)[:80]}")

    for db in matched:
        dbid = db.get("dbid") or db.get("id")
        if dbid:
            walk(dbid, db.get("text", ""))
    log(f"PxWeb: {len(found)} cubes saved, item types seen: {dict(types_seen)}")
    return found


def fetch():
    s = session()
    catalog = dam_search(s)
    for dam_id, label in SEED_ASSETS.items():
        if dam_id not in {c["damId"] for c in catalog}:
            catalog.append({"damId": dam_id, "orderNr": None, "title": label,
                            "lang": "seed", "embargo": None, "format": None})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog).to_csv(OUT_DIR / "dam_catalog_all.csv", index=False)

    latest = pick_latest(catalog)
    latest.to_csv(OUT_DIR / "dam_catalog_latest.csv", index=False)
    log(f"DAM: {len(catalog)} cancer assets, {len(latest)} unique tables -> download")
    for _, row in latest.iterrows():
        download_asset(s, int(row["damId"]), row["title"])
    # previous editions too: the period series lives across editions
    log("DAM: older editions (for period history)")
    for c in catalog:
        download_asset(s, int(c["damId"]), c["title"])

    if FAILED:
        f = pd.DataFrame(FAILED).merge(pd.DataFrame(catalog)[["damId", "orderNr"]],
                                       on="damId", how="left")
        f.to_csv(OUT_DIR / "dam_failed.csv", index=False)
        log(f"DAM: {len(f)} assets without master file -> dam_failed.csv")

    px_discover(s)


# ================================================================ AUDIT =====
def scan_values(values):
    """Classify a flat iterable of cell values into dimension evidence."""
    ev = defaultdict(set)
    canton_set, name_set = set(CANTONS), {n.lower() for n in CANTON_NAMES}
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        t = str(v).strip()
        if not t or len(t) > 200:
            continue
        for a, b in RE_PERIOD.findall(t):
            if 1 <= int(b) - int(a) <= 9:
                ev["periods"].add(f"{a}-{b}")
        m = RE_YEAR.match(t)
        if m:
            ev["years"].add(int(m.group(1)))
        if t in canton_set:
            ev["cantons"].add(t)
        elif t.lower() in name_set:
            ev["cantons"].add(t)
        if REGION_TERMS.search(t):
            ev["regions"].add(t[:40])
        if RE_AGE.match(t):
            ev["age_groups"].add(t)
        for c in RE_ICD.findall(t):
            ev["icd"].add(c.replace(" ", ""))
        if RE_SEX.match(t):
            ev["sex"].add(t.lower())
    return ev


def audit_xlsx(path):
    out = []
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    except Exception as e:
        return [{"file": path.name, "sheet": "-", "error": str(e)[:120]}]
    for name, df in sheets.items():
        ev = scan_values(df.values.ravel())
        # a lone 1986 is a case count, not a year: keep years only from a
        # header-like row/column with >=3 distinct years, or from the sheet name
        axis_years = set()
        for line in list(df.values) + list(df.values.T):
            ys = {int(m.group(1)) for v in line if v is not None
                  for m in [RE_YEAR.match(str(v))] if m}
            if len(ys) >= 3 and max(ys) - min(ys) <= 3 * len(ys):
                axis_years |= ys
        if RE_YEAR.match(str(name)):
            axis_years.add(int(str(name).strip()))
        ev["years"] = axis_years
        title = next((str(x) for x in df.iloc[:5, 0].dropna()), "")[:100] if len(df) else ""
        years = sorted(ev["years"])
        out.append({
            "file": path.name, "sheet": name, "title": title,
            "rows": df.shape[0], "cols": df.shape[1],
            "periods": ", ".join(sorted(ev["periods"])),
            "single_years": f"{years[0]}..{years[-1]} ({len(years)})" if years else "",
            "cantons": len(ev["cantons"]),
            "lang_regions": len(ev["regions"]) > 0,
            "sex": ", ".join(sorted(ev["sex"]))[:40],
            "age_groups": len(ev["age_groups"]),
            "icd_codes": len(ev["icd"]),
        })
    return out


def audit_px(path):
    meta = json.loads(path.read_text())
    row = {"cube": path.stem, "title": meta.get("title") or meta.get("_text", "")}
    dims = []
    for v in meta.get("variables", []):
        vals = v.get("values", [])
        texts = v.get("valueTexts", vals)
        d = f"{v.get('text', v.get('code'))}[{len(vals)}]"
        if v.get("time") or re.search(r"jahr|year|année", v.get("text", ""), re.I):
            ys = [int(x) for x in vals if str(x).isdigit()]
            if ys:
                d += f" {min(ys)}..{max(ys)}"
        ev = scan_values(texts)
        if ev["cantons"]:
            d += f" cantons={len(ev['cantons'])}"
        if ev["icd"]:
            d += f" icd={len(ev['icd'])}"
        dims.append(d)
    row["dims"] = " | ".join(dims)
    return row


def verdict(xl, px):
    """Answer the key question per axis."""
    lines = []
    has = lambda col, cond: xl[cond(xl[col])] if not xl.empty and col in xl else xl
    if not xl.empty:
        per = sorted({p for s in xl["periods"].dropna() for p in s.split(", ") if p})
        lines.append(f"- Incidence periods seen across all NKRS/BFS tables: {', '.join(per) or 'none'}")
        yr = has("single_years", lambda c: c.astype(str).str.len() > 0)
        lines.append(f"- Sheets with single-year axis: {len(yr)} of {len(xl)}")
        ct = has("cantons", lambda c: c >= 20)
        lines.append(f"- Sheets with >=20 cantons: {len(ct)}"
                     + (f" ({', '.join(ct['file'].unique()[:5])})" if len(ct) else ""))
        rg = has("lang_regions", lambda c: c == True)
        lines.append(f"- Sheets with language-region split: {len(rg)}")
        ag = has("age_groups", lambda c: c >= 5)
        lines.append(f"- Sheets with age groups (>=5): {len(ag)}")
        ic = has("icd_codes", lambda c: c >= 10)
        lines.append(f"- Sheets with >=10 ICD-10 codes: {len(ic)}")
    if not px.empty:
        mort = px[px["title"].str.contains("Todesursach|Sterbe|Krebs", case=False, na=False)]
        lines.append(f"- PxWeb mortality-related cubes: {len(mort)}")
        for _, r in mort.head(8).iterrows():
            lines.append(f"  - `{r['cube']}`: {r['title'][:80]}  \n    {r['dims'][:300]}")
        pop = px[px["title"].str.contains("Bevölkerung", case=False, na=False)]
        lines.append(f"- PxWeb population cubes: {len(pop)}")
    return "\n".join(lines)


def audit():
    xl_rows = []
    for p in sorted(RAW_BFS.glob("*.xlsx")):
        xl_rows += audit_xlsx(p)
    xl = pd.DataFrame(xl_rows)
    px = pd.DataFrame([audit_px(p) for p in sorted(RAW_PX.glob("*.json"))])

    cat = OUT_DIR / "dam_catalog_all.csv"
    if cat.exists() and not xl.empty:
        c = pd.read_csv(cat)[["damId", "orderNr", "title", "embargo"]]
        c["file_id"] = c["damId"].astype(str)
        xl["file_id"] = xl["file"].str.split(".").str[0]
        xl = xl.merge(c.rename(columns={"title": "catalog_title"}),
                      on="file_id", how="left").drop(columns="file_id")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xl.to_csv(OUT_DIR / "coverage_xlsx.csv", index=False)
    px.to_csv(OUT_DIR / "coverage_pxweb.csv", index=False)

    others = sorted(p.name for p in RAW_BFS.glob("*") if p.suffix != ".xlsx")
    md = [f"# cancer-ch data audit  ({datetime.now():%Y-%m-%d %H:%M})", "",
          "## Verdict", verdict(xl, px), "",
          f"## XLSX sheets ({len(xl)})", "",
          xl.drop(columns=[c for c in ("error",) if c in xl], errors="ignore")
            .to_markdown(index=False) if not xl.empty else "_none_", "",
          f"## PxWeb cubes ({len(px)})", "",
          px.to_markdown(index=False) if not px.empty else "_none_", "",
          f"## Non-XLSX assets ({len(others)}): " + ", ".join(others[:50])]
    (OUT_DIR / "coverage.md").write_text("\n".join(md))
    log(f"audit: {len(xl)} sheets, {len(px)} px cubes -> {OUT_DIR/'coverage.md'}")
    print("\n" + verdict(xl, px))


if __name__ == "__main__":
    if "--audit" not in sys.argv:
        fetch()
    audit()
