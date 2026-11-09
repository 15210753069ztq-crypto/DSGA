"""Decompress the large streams (stored as .csv.gz to stay below GitHub's file-size limit)."""
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for gz in ROOT.glob("*/*.csv.gz"):
    target = gz.with_suffix("")
    if target.exists():
        print(f"exists   {target.relative_to(ROOT)}")
        continue
    with gzip.open(gz, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"created  {target.relative_to(ROOT)}")
