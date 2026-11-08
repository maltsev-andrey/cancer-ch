#!/usr/bin/env python
"""
s11_geo: cantons, their geometry and their neighbourhood graph, for the spatial model (s12).

Two independent sources, checked against each other:
  1. A canonical adjacency list of the 26 cantons by BFS number (land borders; hard-coded below,
     this is public knowledge and should not depend on how a shapefile cuts the lakes).
  2. swissBOUNDARIES3D (swisstopo, open data): canton polygons, used for maps and to derive a
     queen-contiguity graph that is compared with the canonical list. Differences are printed
     and must be explainable (lake borders, enclaves); the canonical list is what s12 uses.

Geometry input: put the swissBOUNDARIES3D canton layer into data/raw/geo/ (any of
  swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp  (inside the shp zip, unzipped or as zip)
  swissboundaries3d_*.gpkg                      (layer tlm_kantonsgebiet)
from https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d). The script also
tries to download the current gpkg zip from data.geo.admin.ch; if the host is not reachable
from this machine, download by hand. Without geometry the adjacency outputs are still written
and s12 can run; only the maps are skipped.

Outputs (data/processed/geo/)
  cantons.csv                 bfs_nr, abbr, name_de (as in BFS population tables), name_map
  adjacency.csv               bfs_nr_a, bfs_nr_b (each undirected edge once), source=canonical
  adjacency_check.md          canonical vs geometry-derived edges
  cantons_2056.geojson        dissolved canton polygons, LV95, simplified to 100 m (if geometry)
  nodes_edges.json            for s12: node order, edge index pairs, scaling factor of the ICAR
assets/fig15_canton_map.png   test map: population 2022 per canton (if geometry and population_5y)

Usage
  python pipeline/s11_geo.py [--download] [--geo PATH]
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def data_dir(cfg_path):
    import os
    p = Path(os.environ.get("CANCER_CH_DATA", cfg_path))
    return p if p.is_absolute() else (REPO / p).resolve()


DATA = data_dir(Path(yaml.safe_load((REPO / "config" / "sources.yaml").read_text())["paths"]["data"]))
RAW = DATA / "raw" / "geo"
OUT = DATA / "processed" / "geo"
ASSETS = REPO / "assets"

# BFS canton numbers, abbreviations, names as they appear in BFS population tables (German/
# local form used by px-x-0102020000_104 after canton_key()), and the swissBOUNDARIES3D NAME.
CANTONS = [
    (1, "ZH", "Zürich", "Zürich"), (2, "BE", "Bern", "Bern"), (3, "LU", "Luzern", "Luzern"),
    (4, "UR", "Uri", "Uri"), (5, "SZ", "Schwyz", "Schwyz"), (6, "OW", "Obwalden", "Obwalden"),
    (7, "NW", "Nidwalden", "Nidwalden"), (8, "GL", "Glarus", "Glarus"), (9, "ZG", "Zug", "Zug"),
    (10, "FR", "Fribourg", "Fribourg"), (11, "SO", "Solothurn", "Solothurn"),
    (12, "BS", "Basel-Stadt", "Basel-Stadt"), (13, "BL", "Basel-Landschaft", "Basel-Landschaft"),
    (14, "SH", "Schaffhausen", "Schaffhausen"), (15, "AR", "Appenzell Ausserrhoden", "Appenzell Ausserrhoden"),
    (16, "AI", "Appenzell Innerrhoden", "Appenzell Innerrhoden"), (17, "SG", "St. Gallen", "St. Gallen"),
    (18, "GR", "Graubünden", "Graubünden"), (19, "AG", "Aargau", "Aargau"), (20, "TG", "Thurgau", "Thurgau"),
    (21, "TI", "Ticino", "Ticino"), (22, "VD", "Vaud", "Vaud"), (23, "VS", "Valais", "Valais"),
    (24, "NE", "Neuchâtel", "Neuchâtel"), (25, "GE", "Genève", "Genève"), (26, "JU", "Jura", "Jura"),
]
# Land borders between cantons (BFS numbers). Lake-only contacts are not edges: FR–VD across
# Lake Neuchâtel/Murten count because they also share land; VD–GE, TG–SG etc. are land borders.
CANONICAL_EDGES = {
    (1, 5), (1, 9), (1, 14), (1, 17), (1, 19), (1, 20),
    (2, 3), (2, 6), (2, 7), (2, 4), (2, 10), (2, 11), (2, 19), (2, 22), (2, 23), (2, 24), (2, 26),
    (3, 5), (3, 6), (3, 7), (3, 9), (3, 19),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 18), (4, 21), (4, 23),
    (5, 7), (5, 8), (5, 9), (5, 17),
    (6, 7),
    (8, 17), (8, 18),
    (9, 19),
    (10, 22), (10, 24),
    (11, 13), (11, 19), (11, 26),
    (12, 13),
    (13, 19), (13, 26),
    (14, 20),
    (15, 16), (15, 17),
    (16, 17),
    (17, 18), (17, 20),
    (18, 21),
    (21, 23),
    (22, 23), (22, 24), (22, 25),
    (24, 26),
}
CANONICAL_EDGES = {tuple(sorted(e)) for e in CANONICAL_EDGES}

GPKG_URLS = [
    "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2025-04/swissboundaries3d_2025-04_2056_5728.gpkg.zip",
    "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2025-01/swissboundaries3d_2025-01_2056_5728.gpkg.zip",
    "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2024-01/swissboundaries3d_2024-01_2056_5728.gpkg.zip",
]


CANTON_ALIASES = {"Appenzell A.Rh.": "Appenzell Ausserrhoden", "Appenzell I.Rh.": "Appenzell Innerrhoden",
                  "Appenzell A. Rh.": "Appenzell Ausserrhoden", "Appenzell I. Rh.": "Appenzell Innerrhoden"}


def canton_key(name):
    """'Bern / Berne' -> 'Bern'; BFS abbreviations of the two Appenzell -> full names."""
    k = str(name).split(" / ")[0].strip()
    return CANTON_ALIASES.get(k, k)


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- geometry ----
def find_geometry(explicit=None):
    RAW.mkdir(parents=True, exist_ok=True)
    if explicit:
        return Path(explicit)
    for p in sorted(RAW.rglob("*")):
        n = p.name.lower()
        if n.endswith(".gpkg") or ("kantonsgebiet" in n and n.endswith(".shp")):
            return p
        if n.endswith(".zip"):
            try:
                names = zipfile.ZipFile(p).namelist()
            except zipfile.BadZipFile:
                continue
            if any(x.lower().endswith(".gpkg") for x in names):
                return p
            shp = [x for x in names if "kantonsgebiet" in x.lower() and x.lower().endswith(".shp")]
            if shp:
                return Path(f"zip://{p}!{shp[0]}")
    return None


def download(dest_dir):
    import requests
    for url in GPKG_URLS:
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                dest = dest_dir / url.rsplit("/", 1)[1]
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                log(f"downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
                return dest
            log(f"{r.status_code} {url}")
        except Exception as e:
            log(f"download failed: {type(e).__name__}: {e}")
    return None


def load_cantons(path):
    import geopandas as gpd
    p = str(path)
    if p.endswith(".zip") and not p.startswith("zip://"):
        names = zipfile.ZipFile(p).namelist()
        inner = [x for x in names if x.lower().endswith(".gpkg")]
        p = f"zip://{p}!{inner[0]}"
    if ".gpkg" in p.lower():
        import pyogrio
        layers = [l[0] for l in pyogrio.list_layers(p)]
        layer = next(l for l in layers if "kantonsgebiet" in l.lower())
        g = gpd.read_file(p, layer=layer)
    else:
        g = gpd.read_file(p)
    name_col = next(c for c in g.columns if c.upper() == "NAME")
    num_col = next((c for c in g.columns if c.upper() == "KANTONSNUM"), None)
    g = g.to_crs(2056)
    g["geometry"] = g.geometry.force_2d() if hasattr(g.geometry, "force_2d") else g.geometry
    g = g.dissolve(by=name_col, as_index=False)[[name_col, "geometry"] + ([num_col] if num_col else [])]
    g = g.rename(columns={name_col: "name_map", **({num_col: "bfs_nr_src"} if num_col else {})})
    return g


def geometry_edges(g, buffer_m=50):
    """Neighbours = canton polygons that intersect after a small buffer (so a river centreline
    or a sliver between two polygons does not break contact). Queen contiguity on vertices
    would miss buffered polygons entirely; an intersects-join does not."""
    import geopandas as gpd
    gb = gpd.GeoDataFrame({"bfs_nr": g.bfs_nr.values}, geometry=g.geometry.buffer(buffer_m), crs=g.crs)
    j = gpd.sjoin(gb, gb, predicate="intersects")
    pairs = j[j.bfs_nr_left != j.bfs_nr_right]
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(pairs.bfs_nr_left, pairs.bfs_nr_right)}


# ---------------------------------------------------------------- graph -------
def icar_scaling_factor(n, edges):
    """Scaling factor of the ICAR precision so that the structured component has a
    generalised variance of 1 (Riebler et al. 2016, Freni-Sterrantino et al. 2018).
    Returns the geometric mean of the marginal variances of the (pseudo-)inverse."""
    Q = np.zeros((n, n))
    for a, b in edges:
        Q[a, a] += 1; Q[b, b] += 1; Q[a, b] -= 1; Q[b, a] -= 1
    # (Q + 11'/n)^-1 = Q^+ + 11'/n, so the constrained marginal variances are diag minus 1/n
    V = np.linalg.inv(Q + np.ones((n, n)) / n)
    v = np.diag(V) - 1.0 / n
    return float(np.exp(np.mean(np.log(v))))


def connected(n, edges):
    seen, stack = {0}, [0]
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j not in seen:
                seen.add(j); stack.append(j)
    return len(seen) == n


# ---------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="try to fetch swissBOUNDARIES3D gpkg")
    ap.add_argument("--geo", help="path to the canton layer (shp, gpkg, or zip)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cant = pd.DataFrame(CANTONS, columns=["bfs_nr", "abbr", "name_de", "name_map"])
    cant.to_csv(OUT / "cantons.csv", index=False)

    # canonical graph
    n = len(cant)
    nr_to_idx = {r.bfs_nr: i for i, r in enumerate(cant.itertuples())}
    edges_idx = sorted({(nr_to_idx[a], nr_to_idx[b]) for a, b in CANONICAL_EDGES})
    assert connected(n, edges_idx), "canonical canton graph is not connected"
    deg = np.zeros(n, int)
    for a, b in edges_idx:
        deg[a] += 1; deg[b] += 1
    scale = icar_scaling_factor(n, edges_idx)
    pd.DataFrame(sorted(CANONICAL_EDGES), columns=["bfs_nr_a", "bfs_nr_b"]).assign(source="canonical") \
        .to_csv(OUT / "adjacency.csv", index=False)
    (OUT / "nodes_edges.json").write_text(json.dumps({
        "nodes": cant.bfs_nr.tolist(), "abbr": cant.abbr.tolist(),
        "edges": edges_idx, "n_edges": len(edges_idx), "degree": deg.tolist(),
        "icar_scaling_factor": round(scale, 4)}, indent=1))
    log(f"canonical graph: {n} nodes, {len(edges_idx)} edges, degree {deg.min()}..{deg.max()}, "
        f"ICAR scaling factor {scale:.3f}")

    # population check: canton names in population_5y must match cantons.csv
    pop_path = DATA / "processed" / "population_5y.parquet"
    pop22 = None
    if pop_path.exists():
        pop = pd.read_parquet(pop_path)
        if "canton" in pop:
            names = {canton_key(c) for c in pop.canton.unique()}
            missing = set(cant.name_de) - names
            extra = names - set(cant.name_de)
            log(f"population_5y cantons: {len(names)}; missing from table: {sorted(missing) or 'none'}; "
                f"not a canton: {sorted(extra) or 'none'}")
            if "year" in pop and "midyear" in pop:
                pop22 = pop[pop.year == pop.year.max()].groupby("canton").midyear.sum()

    # geometry
    geo = find_geometry(args.geo)
    if geo is None and args.download:
        geo = download(RAW)
    if geo is None:
        log("no canton geometry found in data/raw/geo; adjacency written, maps skipped. "
            "Download swissBOUNDARIES3D (gpkg or shp zip) into data/raw/geo and rerun.")
        (OUT / "adjacency_check.md").write_text("# s11 adjacency check\n\nno geometry available; canonical list only.\n")
        return
    log(f"geometry: {geo}")
    g = load_cantons(geo)
    g = g.merge(cant, on="name_map", how="outer", indicator=True)
    bad = g[g._merge != "both"]
    if len(bad):
        log(f"name mismatch between geometry and cantons.csv:\n{bad[['name_map', 'name_de', '_merge']]}")
        raise SystemExit(1)
    g = g.drop(columns="_merge").sort_values("bfs_nr").reset_index(drop=True)
    geo_edges = geometry_edges(g)
    only_geo = sorted(geo_edges - CANONICAL_EDGES)
    only_can = sorted(CANONICAL_EDGES - geo_edges)
    ab = dict(zip(cant.bfs_nr, cant.abbr))
    fmt = lambda es: ", ".join(f"{ab[a]}-{ab[b]}" for a, b in es) or "none"
    md = ["# s11 adjacency check", "",
          f"- canonical edges: {len(CANONICAL_EDGES)}; from geometry (queen, 50 m buffer): {len(geo_edges)}",
          f"- in geometry only: {fmt(only_geo)}", f"- in canonical only: {fmt(only_can)}", "",
          "s12 uses the canonical list. Geometry-only edges are usually lake contacts or the buffer;",
          "canonical-only edges would mean a missing land border and must be looked at.", ""]
    (OUT / "adjacency_check.md").write_text("\n".join(md))
    log(f"geometry edges {len(geo_edges)}; only in geometry: {fmt(only_geo)}; only canonical: {fmt(only_can)}")

    g["geometry"] = g.geometry.simplify(100, preserve_topology=True)
    g[["bfs_nr", "abbr", "name_de", "geometry"]].to_file(OUT / "cantons_2056.geojson", driver="GeoJSON")

    # test map
    if pop22 is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g["pop"] = g.name_de.map(pop22)
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
        g.plot(column="pop", cmap="Blues", linewidth=0.6, edgecolor="#898781", ax=ax, legend=True,
               legend_kwds={"label": "resident population (mid-year)", "shrink": 0.6})
        for r in g.itertuples():
            c = r.geometry.representative_point()
            ax.annotate(r.abbr, (c.x, c.y), ha="center", va="center", fontsize=7, color="#0b0b0b")
        ax.set_axis_off()
        ax.set_title(f"Cantons and population, {int(pop.year.max())}: geometry and name join check",
                     loc="left", fontweight="bold")
        fig.text(0.01, 0.02, "Source: swisstopo swissBOUNDARIES3D; BFS px-x-0102020000_104", color="#898781", fontsize=8)
        ASSETS.mkdir(exist_ok=True)
        fig.savefig(ASSETS / "fig15_canton_map.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        log("  -> assets/fig15_canton_map.png")


if __name__ == "__main__":
    main()
