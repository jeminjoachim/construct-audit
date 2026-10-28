"""
export_forum_public.py: build the public, text-free files for Study 1 (forum posts).

Reads the private files (never published) and writes:
    data/forum_public/edits.csv   one row per edit: post link, the words that changed, labels, verdict
    data/forum_public/scores.csv  copy of results/pilot_score_table.csv (contains no text)

No full post text is written. For AI edits, only the changed words are kept
(e.g. "Where is -> Is there"). Neutral passages are identified by their number in
features.yaml, where their text is published.

Run from the repo root:
    python scripts\\export_forum_public.py
"""
import difflib
import os
import shutil
import sqlite3
import sys

import pandas as pd
import yaml

VARIANTS = "data/private/variants_pilot.csv"
CHECKS = "data/private/variant_checks.csv"
SCORES = "results/pilot_score_table.csv"
CONFIG = "config.yaml"
OUT_DIR = "data/forum_public"
URL = "https://bitcointalk.org/index.php?topic={topic}.msg{msg}#msg{msg}"


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    return df


def topic_ids(post_ids):
    """Look up each post's thread (topic) number in the private SQLite corpus."""
    try:
        with open(CONFIG, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        db, table = cfg["corpus_sqlite"], cfg.get("corpus_table", "posts")
        con = sqlite3.connect(db)
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        id_col = next((c for c in ("post_id", "msg_id", "id") if c in cols), None)
        tp_col = next((c for c in ("topic_id", "topic", "thread_id") if c in cols), None)
        if not id_col or not tp_col:
            print(f"Could not find post/topic columns in {table}: {cols}")
            return {}
        q = f"SELECT {id_col}, {tp_col} FROM {table} WHERE {id_col} IN ({','.join('?' * len(post_ids))})"
        rows = con.execute(q, list(post_ids)).fetchall()
        con.close()
        return {str(a): str(b) for a, b in rows}
    except Exception as e:  # links are optional; the export still works without them
        print(f"Topic lookup failed ({e}); the post_url column will be empty.")
        return {}


def changed_words(original, variant):
    """Word-level differences only, e.g. 'Where is -> Is there; result -> outcome'."""
    a, b = original.split(), variant.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        old, new = " ".join(a[i1:i2]) or "(nothing)", " ".join(b[j1:j2]) or "(nothing)"
        parts.append(f"{old} -> {new}")
    return "; ".join(parts)


def main():
    for path in (VARIANTS, CHECKS, SCORES):
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run this from the repo root.")
    v = read_csv(VARIANTS)
    # The variants file has its own (empty) chk_ columns; drop them so the review
    # decisions come only from the checks file.
    v = v.drop(columns=[col for col in v.columns if col.startswith("chk_")])
    c = read_csv(CHECKS)[["seed_id", "variant_id", "chk_my_label", "chk_verdict"]]
    v = v.merge(c, on=["seed_id", "variant_id"], how="left").fillna("")
    v = v[v["variant_type"] != "original"].copy()

    topics = topic_ids(sorted(v["seed_id"].unique()))
    v["post_url"] = [URL.format(topic=topics[p], msg=p) if p in topics else "" for p in v["seed_id"]]

    def describe(r):
        if r["variant_type"] == "padding":
            return f"neutral passage {r['variant_id'].split('_')[-1]} appended (text in features.yaml)"
        if r["status"] != "OK" or not r["variant_text"]:
            return "(no edit produced)"
        return changed_words(r["original_text"], r["variant_text"])

    v["words_changed"] = v.apply(describe, axis=1)
    v["author_label"] = v["chk_my_label"]
    v["verdict"] = v["chk_verdict"].where(v["variant_type"] != "padding", "not reviewed (script)")
    out = v[["seed_id", "post_url", "variant_id", "variant_type", "expected_label",
             "author_label", "verdict", "auto_flag", "model", "words_changed", "timestamp"]]
    out = out.rename(columns={"seed_id": "post_id", "variant_id": "edit"})

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(f"{OUT_DIR}/edits.csv", index=False, encoding="utf-8")
    shutil.copyfile(SCORES, f"{OUT_DIR}/scores.csv")

    # Safety check: no row should contain a long stretch of an original post.
    longest = out["words_changed"].str.len().max()
    print(f"wrote {len(out)} edits to {OUT_DIR}/edits.csv and copied scores to {OUT_DIR}/scores.csv")
    print(f"posts with a link: {sum(out.drop_duplicates('post_id')['post_url'] != '')} "
          f"of {out['post_id'].nunique()}")
    print(f"longest 'words_changed' entry: {longest} characters. Check any entry that looks "
          "like a whole post before committing.")


if __name__ == "__main__":
    main()
