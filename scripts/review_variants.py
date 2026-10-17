"""
review_variants.py — record your manual checks of the LLM variants, one at a time.

Shows each LLM variant next to its original, asks six questions, and saves after
every answer to data/private/variant_checks.csv (keyed by seed_id + variant_id).
The generated file variants_pilot.csv is never modified.

Blinding: the expected label is NOT shown. Give your own codebook label for the
variant first; the script compares it with the expectation only after you answer.

Run from the repo root:
    python scripts\\review_variants.py            # resumes where you stopped
    python scripts\\review_variants.py --redo 30891460:question_form   # redo one row
Type q at any prompt to save and quit.
"""
import argparse
import csv
import os
import sys
import textwrap

import pandas as pd

VARIANTS = "data/private/variants_pilot.csv"
CHECKS = "data/private/variant_checks.csv"
COLS = ["seed_id", "variant_id", "chk_target_changed", "chk_other_changed",
        "chk_natural", "chk_my_label", "chk_verdict", "chk_note", "matches_expected"]

QUESTIONS = [
    ("chk_target_changed", "Did the NAMED edit happen as intended? (y/n)", {"y", "n"}),
    ("chk_other_changed", "Did anything ELSE change (meaning, tone, referent, claim)? (y/n)",
     {"y", "n"}),
    ("chk_natural", "Could this pass as a real board-159 post? (y/n)", {"y", "n"}),
    ("chk_my_label", "YOUR codebook label for the VARIANT (1 = DD, 0 = not DD)", {"1", "0"}),
    ("chk_verdict", "Verdict: a = accept, r = reject, h = accept but researcher-authored",
     {"a", "r", "h"}),
]
VERDICT = {"a": "accept", "r": "reject", "h": "authored"}


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    return df


def save(checks):
    pd.DataFrame(checks, columns=COLS).to_csv(CHECKS, index=False, encoding="utf-8",
                                              quoting=csv.QUOTE_ALL)


def ask(prompt, allowed):
    while True:
        ans = input(f"  {prompt}: ").strip().lower()
        if ans == "q":
            return None
        if allowed is None or ans in allowed:
            return ans
        print(f"    please answer one of {sorted(allowed)} (or q to quit)")


def wrap(label, text):
    return textwrap.fill(text, width=100, initial_indent=f"  {label:<9}",
                         subsequent_indent=" " * 11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", default=None, help="seed_id:variant_id to review again")
    ap.add_argument("--variants", default=VARIANTS)
    ap.add_argument("--checks", default=CHECKS)
    args = ap.parse_args()
    globals()["VARIANTS"], globals()["CHECKS"] = args.variants, args.checks

    v = read_csv(VARIANTS)
    llm = v[v["variant_type"] == "llm"].reset_index(drop=True)

    checks = read_csv(CHECKS).to_dict("records") if os.path.exists(CHECKS) else []
    if args.redo:
        sid, vid = args.redo.split(":")
        checks = [c for c in checks if not (c["seed_id"] == sid and c["variant_id"] == vid)]
    done = {(c["seed_id"], c["variant_id"]) for c in checks}

    todo = llm[[(s, i) not in done for s, i in zip(llm["seed_id"], llm["variant_id"])]]
    print(f"{len(llm)} LLM variants, {len(done)} already reviewed, {len(todo)} to go.\n")

    for n, (_, r) in enumerate(todo.iterrows(), start=1):
        print("=" * 104)
        print(f"[{n}/{len(todo)}]  seed {r['seed_id']}  |  edit: {r['variant_id']}  |  "
              f"generator status: {r['status']}  |  auto_flag: {r['auto_flag'] or '-'}")
        print(wrap("ORIGINAL", r["original_text"]))
        print(wrap("VARIANT", r["variant_text"] or "(empty)"))
        if r["reason"]:
            print(wrap("MODEL", f"reason: {r['reason']}"))
        print(wrap("CHANGES", r["words_changed_reported"] or "-"))
        print()

        row = {"seed_id": r["seed_id"], "variant_id": r["variant_id"]}
        if r["status"] != "OK":
            print("  Generator did not produce a variant -> recorded as reject.")
            row.update(chk_target_changed="n", chk_other_changed="n", chk_natural="",
                       chk_my_label="", chk_verdict="reject")
        else:
            for col, prompt, allowed in QUESTIONS:
                ans = ask(prompt, allowed)
                if ans is None:
                    save(checks)
                    print(f"\nSaved {len(checks)} reviews to {CHECKS}. Run again to resume.")
                    return
                row[col] = VERDICT[ans] if col == "chk_verdict" else ans

        note = ask("Note (optional, Enter to skip)", None)
        if note is None:
            save(checks)
            print(f"\nSaved {len(checks)} reviews to {CHECKS}. Run again to resume.")
            return
        row["chk_note"] = note

        if row.get("chk_my_label") in {"0", "1"}:
            match = row["chk_my_label"] == r["expected_label"]
            row["matches_expected"] = "y" if match else "n"
            print(f"  -> expected label was {r['expected_label']}: "
                  f"{'matches your label' if match else 'DIFFERS from your label'}")
        else:
            row["matches_expected"] = ""

        checks.append(row)
        save(checks)
        print()

    print(f"All done: {len(checks)} reviews in {CHECKS}\n")
    c = pd.DataFrame(checks, columns=COLS).merge(
        llm[["seed_id", "variant_id"]], on=["seed_id", "variant_id"])
    c["feature"] = c["variant_id"]
    print("Verdicts by edit:")
    print(pd.crosstab(c["feature"], c["chk_verdict"]).to_string())
    print("\nYour label vs expected (accepted variants only):")
    acc = c[c["chk_verdict"].isin(["accept", "authored"])]
    print(pd.crosstab(acc["feature"], acc["matches_expected"]).to_string())


if __name__ == "__main__":
    sys.exit(main())
