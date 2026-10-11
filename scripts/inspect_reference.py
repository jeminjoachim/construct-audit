"""
inspect_reference.py — print what score_variants.py needs to know about the corpus
score files and the SQLite corpus. Prints column names and numbers only, no post text.

    python scripts\\inspect_reference.py
"""
import glob
import sqlite3

import pandas as pd

ROOT = "C:/Users/jejo9217/Documents/crypto_data_scraper"

files = sorted(glob.glob(f"{ROOT}/data/processed/corpus_scores/per_month/*.parquet"))
print(f"{len(files)} per-month parquet files; first: {files[0] if files else '-'}")
if files:
    d = pd.read_parquet(files[0])
    print("columns:", d.columns.tolist())
    print("rows:", len(d))
    print(d.select_dtypes(exclude="object").head(3).to_string())

print("\nreport (first 25 lines):")
try:
    with open(f"{ROOT}/data/processed/corpus_scores/scoring_report.txt", encoding="utf-8") as fh:
        for line in fh.readlines()[:25]:
            print("  " + line.rstrip())
except OSError as e:
    print("  could not read:", e)

dbs = glob.glob(f"{ROOT}/**/*.sqlite", recursive=True)
print("\nSQLite files:", dbs)
for db in dbs:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        print(f"  {db.split('/')[-1]} :: {t}: {cols}")
    con.close()
