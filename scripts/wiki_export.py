#!/usr/bin/env python3
"""
Convert docs/wiki/*.md (GitHub flavour: relative links to .md files and ../../assets images)
into docs/wiki_bookstack/*.md for the internal BookStack wiki:

  [Data sources](01_data_sources.md)        -> [Data sources](/books/<book>/page/01-data-sources)
  [chapter 05](05_icd_bridge.md#factors)    -> [chapter 05](/books/<book>/page/05-icd-8-icd-10-bridge-s03#factors)
  ![x](../../assets/fig01.png)              -> ![x](fig01.png)   + listed in images.txt to upload

BookStack derives the page slug from the page title (the H1): lowercase, every run of
non-alphanumerics becomes one dash. The H1 is read from each file, so the slug follows
whatever title you paste into BookStack, as long as the title equals the H1.

Run:  python scripts/wiki_export.py            (defaults below)
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "wiki"
DST = REPO / "docs" / "wiki_bookstack"
BOOK = "cancer-ch"                       # BookStack book slug
PAGE_PREFIX = f"/books/{BOOK}/page/"     # absolute path on the wiki host


def slugify(title):
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def h1(text):
    m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    return m.group(1) if m else None


def main():
    DST.mkdir(exist_ok=True)
    files = sorted(SRC.glob("*.md"))
    slugs = {f.name: slugify(h1(f.read_text())) for f in files}
    images = set()
    print("page slugs (must match the page titles in BookStack):")
    for name, slug in slugs.items():
        print(f"  {name:28s} -> {PAGE_PREFIX}{slug}")

    for f in files:
        t = f.read_text()

        def link(m):
            label, target = m.group(1), m.group(2)
            path, _, anchor = target.partition("#")
            if path in slugs:
                return f"[{label}]({PAGE_PREFIX}{slugs[path]}{'#' + anchor if anchor else ''})"
            return m.group(0)

        def image(m):
            alt, target = m.group(1), m.group(2)
            fname = Path(target).name
            images.add(fname)
            return f"![{alt}]({fname})"

        t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, t)
        t = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", link, t)
        (DST / f.name).write_text(t)

    (DST / "images.txt").write_text("\n".join(sorted(images)) + "\n")
    print(f"\n{len(files)} pages written to {DST.relative_to(REPO)}; "
          f"{len(images)} images to upload (see images.txt)")


if __name__ == "__main__":
    main()
