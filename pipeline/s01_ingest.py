#!/usr/bin/env python3
"""
cancer-ch / Stage 1: ingest.

PxWeb cubes  -> data/interim/pxweb/<cube>.parquet   (long format, one row per cell)
DAM tables   -> data/raw/bfs_dam/mortality/<orderNr>_<damId>.xlsx (parsed in s02)
Every artefact gets a line in data/interim/manifest.jsonl (url, sha256, rows, time).

Run:  python pipeline/s01_ingest.py --plan    # resolve selections, show cell counts, no pull
      python pipeline/s01_ingest.py           # plan + pull everything
      python pipeline/s01_ingest.py --only px-x-1403030300_161
"""
import hashlib
import itertools
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# ---------------------------------------------------------------- config ----
REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    """paths.data from sources.yaml, relative to the repo root unless absolute;
    the environment variable CANCER_CH_DATA overrides both."""
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()
CONFIG = REPO / "config" / "sources.yaml"
UA = "cancer-ch-ingest/0.1 (research, aggregated public data)"
TIMEOUT = 120

cfg = yaml.safe_load(CONFIG.read_text())
DATA = data_dir(Path(cfg["paths"]["data"]))
PX = cfg["pxweb"]
DAM = cfg["dam"]
OUT_PX = DATA / "interim" / "pxweb"
OUT_DAM = DATA / "raw" / "bfs_dam" / "mortality"
MANIFEST = DATA / "interim" / "manifest.jsonl"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def manifest(entry):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entry["retrieved"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with MANIFEST.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ================================================================ PXWEB =====
def cube_url(cube_id):
    return f"{PX['api']}/{cube_id}/{cube_id}.px"


def resolve(meta, select):
    """Turn {variable text/code: regex} into {code: [value codes]} for every variable."""
    plan = {}
    for v in meta["variables"]:
        codes, texts = v["values"], v.get("valueTexts", v["values"])
        rule = select.get(v["text"], select.get(v["code"]))
        if rule is None:
            plan[v["code"]] = (v["text"], codes)
            continue
        picked = [c for c, t in zip(codes, texts) if re.search(rule, t, re.I)]
        if not picked:
            sample = " | ".join(texts[:25])
            raise ValueError(f"'{v['text']}': regex /{rule}/ matched nothing. "
                             f"Available: {sample}")
        plan[v["code"]] = (v["text"], picked)
    unknown = set(select) - {v["text"] for v in meta["variables"]} \
                         - {v["code"] for v in meta["variables"]}
    if unknown:
        raise ValueError(f"select keys not in cube: {sorted(unknown)}")
    return plan


def n_cells(plan):
    return int(np.prod([len(vals) for _, vals in plan.values()]))


def chunks(plan, max_cells):
    """Split along the largest dimensions until each query is below max_cells."""
    parts = [dict(plan)]
    while any(n_cells(p) > max_cells for p in parts):
        out = []
        for p in parts:
            if n_cells(p) <= max_cells:
                out.append(p)
                continue
            code = max(p, key=lambda k: len(p[k][1]))
            text, vals = p[code]
            if len(vals) == 1:
                raise RuntimeError("cannot split further, raise max_cells")
            half = len(vals) // 2
            out += [{**p, code: (text, vals[:half])}, {**p, code: (text, vals[half:])}]
        parts = out
    return parts


def jsonstat_to_long(js):
    """JSON-stat 2.0 dataset -> long DataFrame with <dim> (label) and <dim>_code columns."""
    ids, sizes = js["id"], js["size"]
    axes = []
    for d in ids:
        cat = js["dimension"][d]["category"]
        idx = cat["index"]
        order = sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx)
        labels = cat.get("label", {})
        axes.append((d, js["dimension"][d].get("label", d), order,
                     [labels.get(c, c) for c in order]))
    total = int(np.prod(sizes))
    val = js["value"]
    if isinstance(val, dict):                     # sparse form
        dense = [None] * total
        for k, x in val.items():
            dense[int(k)] = x
        val = dense
    status = js.get("status") or {}
    if isinstance(status, list):
        status = {i: s for i, s in enumerate(status) if s}
    grid = np.array(list(itertools.product(*[range(len(a[2])) for a in axes])))
    df = pd.DataFrame({"value": pd.to_numeric(pd.Series(val, dtype=object), errors="coerce")})
    for j, (d, label, codes, texts) in enumerate(axes):
        df[label] = np.array(texts, dtype=object)[grid[:, j]]
        df[f"{label}_code"] = np.array(codes, dtype=object)[grid[:, j]]
    if status:
        df["status"] = pd.Series({int(k): v for k, v in status.items()}).reindex(df.index)
    return df


def pull_cube(s, cube, dry=False):
    cid = cube["id"]
    url = cube_url(cid)
    meta = s.get(url, timeout=TIMEOUT).json()
    plan = resolve(meta, cube.get("select") or {})
    parts = chunks(plan, PX["max_cells"])
    dims = ", ".join(f"{t}[{len(v)}]" for t, v in plan.values())
    log(f"{cid} ({cube.get('role')}): {n_cells(plan):,} cells in {len(parts)} request(s)")
    log(f"    {dims}")
    if dry:
        return None
    frames = []
    for i, p in enumerate(parts, 1):
        q = {"query": [{"code": c, "selection": {"filter": "item", "values": v}}
                       for c, (_, v) in p.items()],
             "response": {"format": "json-stat2"}}
        for attempt in range(3):
            time.sleep(PX["sleep"] * (1 + 2 * attempt))
            r = s.post(url, json=q, timeout=TIMEOUT)
            if r.status_code == 429:
                log("    429 rate limit, backing off")
                continue
            r.raise_for_status()
            break
        frames.append(jsonstat_to_long(r.json()))
        if len(parts) > 1:
            log(f"    part {i}/{len(parts)}: {len(frames[-1]):,} rows")
    df = pd.concat(frames, ignore_index=True)
    OUT_PX.mkdir(parents=True, exist_ok=True)
    out = OUT_PX / f"{cid}.parquet"
    df.to_parquet(out, index=False)
    (OUT_PX / f"{cid}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    miss = df["value"].isna().mean() * 100
    log(f"    -> {out.name}: {len(df):,} rows, {miss:.1f}% missing values")
    manifest({"source": "pxweb", "id": cid, "role": cube.get("role"), "url": url,
              "title": meta.get("title"), "rows": len(df), "missing_pct": round(miss, 2),
              "file": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
    return df


# ================================================================== DAM =====
def latest_asset(s, order_nr):
    r = s.get(DAM["api"], params={"orderNr": order_nr, "language": DAM["language"],
                                  "limit": 100}, timeout=TIMEOUT)
    r.raise_for_status()
    hits = [a for a in r.json().get("data", [])
            if (a.get("shop") or {}).get("orderNr") == order_nr]
    if not hits:
        return None
    hits.sort(key=lambda a: (a.get("bfs") or {}).get("embargo") or "", reverse=True)
    return hits[0]


def pull_dam(s, item, dry=False):
    onr = item["id"]
    a = latest_asset(s, onr)
    if a is None:
        log(f"{onr}: not found in DAM")
        return
    dam_id = a["ids"]["damId"]
    title = a["description"]["titles"]["main"]
    emb = ((a.get("bfs") or {}).get("embargo") or "")[:10]
    log(f"{onr} -> {dam_id} ({emb}) {title[:70]}")
    if dry:
        return
    OUT_DAM.mkdir(parents=True, exist_ok=True)
    out = OUT_DAM / f"{onr}_{dam_id}.xlsx"
    if out.exists():
        log("    exists, skip")
        return
    url = f"{DAM['api']}/{dam_id}/master"
    r = s.get(url, timeout=TIMEOUT)
    if r.status_code == 404:
        log("    no master file (404)")
        return
    r.raise_for_status()
    out.write_bytes(r.content)
    log(f"    -> {out.name} {len(r.content)/1024:.0f} kB")
    manifest({"source": "dam", "id": onr, "role": item.get("role"), "damId": dam_id,
              "title": title, "embargo": emb, "url": url, "file": str(out),
              "sha256": hashlib.sha256(r.content).hexdigest()})


# ================================================================= main =====
def main():
    dry = "--plan" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    s = session()
    errors = []
    for cube in PX["cubes"]:
        if only and cube["id"] != only:
            continue
        try:
            pull_cube(s, cube, dry)
        except Exception as e:
            errors.append(cube["id"])
            log(f"{cube['id']}: FAILED {e}")
    for item in DAM["order_nrs"]:
        if only and item["id"] != only:
            continue
        try:
            pull_dam(s, item, dry)
        except Exception as e:
            errors.append(item["id"])
            log(f"{item['id']}: FAILED {e}")
    log(f"done, {len(errors)} error(s){': ' + ', '.join(errors) if errors else ''}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
