"""
generate_variants.py — construct-audit pilot: build minimal-pair variants for each seed.

For every seed in design/seeds_pilot.csv it writes:
  - 1 'original' row (the unedited post, for reference)
  - 1 LLM variant for the seed's assigned feature (question_form or object_swap)
  - 1 LLM 'placebo' variant
  - 5 'padding' variants (deterministic: original + one padding_bank block each)

Output: data/private/variants_pilot.csv (resumable: rows already present are skipped)
        data/private/generation_log.jsonl (raw API request/response per call)

Run from the repo root, with OPENAI_API_KEY set:
    $env:OPENAI_API_KEY = "sk-..."         # PowerShell, current session only
    python scripts\\generate_variants.py
    python scripts\\generate_variants.py --dry-run     # prints prompts, no API calls
"""
import argparse
import csv
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd
import yaml

# ---- settings -------------------------------------------------------------
MODEL = "gpt-6-luna"            # generator model; the LLM *coder* must be a different family
TEMPERATURE = 0.0               # dropped automatically (and logged) if the model rejects it
MAX_TOKENS = 4000               # generous: reasoning models spend output tokens before answering
MAX_RETRIES = 2

FEATURES_PATH = "features.yaml"
SEEDS_PATH = "design/seeds_pilot.csv"
SYSTEM_PROMPT_PATH = "prompts/generation_system_v1.txt"
MASTER_PATH = "data/private/master_coded_v2.csv"
OUT_PATH = "data/private/variants_pilot.csv"
LOG_PATH = "data/private/generation_log.jsonl"

PADDING_SEP = " / "   # corpus text_own uses " / " for line breaks; padding goes on a new line

LLM_FEATURES = {"question_form", "object_swap", "placebo"}
CHECK_COLS = ["chk_target_changed", "chk_other_changed", "chk_natural",
              "chk_my_label", "chk_verdict", "chk_note"]


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()
    return df


# ---- edit instructions per seed --------------------------------------------
def question_form_instruction(seed_label):
    if seed_label == "1":
        return ("Change the question that presupposes something is missing or wrong "
                "('where is X', 'why is X') into a neutral question asking whether X exists "
                "('is there X', 'any X?'). Change only that one question.")
    return ("Change the neutral question asking whether something exists ('is there X', "
            "'any X?', 'are there any X') into one that presupposes it is missing "
            "('where is X', 'where are the X'). Keep the same X. If there are several "
            "questions, change only the one about the project's own material or updates.")


def object_swap_instruction(note):
    m = re.search(r"(\w+)\s*->\s*(\w+)", str(note))
    if not m:
        sys.exit(f"object_swap seed needs 'source -> target' in its note, got: {note!r}")
    src, tgt = m.group(1), m.group(2)
    labels = {
        "team": "the project team (identities, members, experience, profiles)",
        "tokenomics": "token economics (supply, pre-mine, allocation, ICO funds)",
        "disclosure": "disclosed information (numbers, documents, proof of claims)",
        "code_progress": "code and development (source code, wallet, repository, releases)",
    }
    return (f"Keep the act (the demand, challenge, or red flag) and its wording exactly. "
            f"Replace what is being scrutinised, which is about {labels.get(src, src)}, "
            f"with something about {labels.get(tgt, tgt)}. Change only the words needed "
            f"to name the new object; keep any numbers or comparisons structurally parallel.")


def object_swap_instruction_v2(note):
    """v2 (after pilot v1: 1/5 accepted). Swapping the noun alone broke the referent,
    because the predicate belongs to the object (a percentage belongs to tokenomics).
    v2 swaps the object together with its own predicate, and keeps the act fixed."""
    m = re.search(r"(\w+)\s*->\s*(\w+)", str(note))
    if not m:
        sys.exit(f"object_swap seed needs 'source -> target' in its note, got: {note!r}")
    src, tgt = m.group(1), m.group(2)
    examples = {
        "team": "a named team member's background, the team's identities, LinkedIn profiles, "
                "or the team's stated experience",
        "code_progress": "the source code, the GitHub repository, the wallet release, "
                         "or the testnet",
    }
    return (
        "This post scrutinises the project. Rewrite it so that it scrutinises a different "
        f"object: {examples.get(tgt, tgt)}, instead of the current one ({src}).\n"
        "Keep fixed: the act (asking, demanding proof, challenging, accusing), the sentence "
        "structure, the question form, the tone, the spelling, and every sentence that is not "
        "about the object.\n"
        "Change: the object AND the specific claim or detail attached to it, so the new "
        "sentence is a concrete, checkable statement that makes sense for the new object "
        "(e.g. 'the ICO amount does not match the address balance' -> 'the team listed on the "
        "site does not match their LinkedIn profiles').\n"
        "The result must read as a natural forum post. Up to 8 words may differ. Return "
        "NOT_APPLICABLE only if the act itself would have to change."
    )


def build_user_prompt(post_text, instruction, system_path=""):
    unit = "SENTENCE" if "disclosure" in os.path.basename(system_path) else "POST"
    return f"{unit}:\n<<<\n{post_text}\n>>>\n\nEDIT:\n{instruction}"


# ---- automatic checks -------------------------------------------------------
def auto_checks(original, variant):
    ow, vw = original.split(), variant.split()
    sm = difflib.SequenceMatcher(a=ow, b=vw)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes()
                  if tag != "equal")
    return {
        "wc_original": len(ow),
        "wc_variant": len(vw),
        "wc_delta": len(vw) - len(ow),
        "words_edited": changed,
        "token_similarity": round(sm.ratio(), 3),
        "identical": original.strip() == variant.strip(),
    }


def auto_flag(row):
    if row["variant_type"] != "llm":
        return ""
    flags = []
    if row["status"] != "OK":
        flags.append("not_ok")
    if row.get("identical"):
        flags.append("identical")
    if abs(row.get("wc_delta", 0)) > 3:
        flags.append("length")
    if row.get("token_similarity", 1) < 0.6:
        flags.append("large_edit")
    return ";".join(flags)


# ---- API call ---------------------------------------------------------------
def call_llm(client, system_prompt, user_prompt):
    """OpenAI Responses API. Returns (parsed, raw, model_used, attempts, temperature_used)."""
    global TEMPERATURE
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        kwargs = dict(model=MODEL, instructions=system_prompt, input=user_prompt,
                      max_output_tokens=MAX_TOKENS)
        if TEMPERATURE is not None:
            kwargs["temperature"] = TEMPERATURE
        try:
            resp = client.responses.create(**kwargs)
            raw = resp.output_text or ""
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else None
            if not parsed or "status" not in parsed:
                raise ValueError(f"no valid JSON object in response (status={resp.status})")
            return parsed, raw, resp.model, attempt, TEMPERATURE
        except Exception as e:          # noqa: BLE001 — log and retry
            last_err = e
            if TEMPERATURE is not None and "temperature" in str(e).lower():
                print("  model rejected 'temperature'; continuing without it (logged per row)")
                TEMPERATURE = None
                continue
            time.sleep(2 * (attempt + 1))
    return ({"variant": "", "words_changed": [], "status": "ERROR",
             "reason": str(last_err)[:300]}, "", MODEL, MAX_RETRIES, TEMPERATURE)


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--features", default=FEATURES_PATH)
    ap.add_argument("--seeds", default=SEEDS_PATH)
    ap.add_argument("--texts", default=MASTER_PATH,
                    help="CSV with the seed texts (post_id + text_own, or id + text)")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--system", default=SYSTEM_PROMPT_PATH, help="system prompt file")
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--only", default=None,
                    help="generate only this variant_id, e.g. object_swap_v2 (revision run)")
    args = ap.parse_args()
    FEATURES, SEEDS, TEXTS = args.features, args.seeds, args.texts
    globals()["OUT_PATH"], globals()["LOG_PATH"] = args.out, args.log

    with open(FEATURES, encoding="utf-8") as fh:
        feats = yaml.safe_load(fh)
    padding_mode = feats.get("meta", {}).get("padding_mode", "append")
    feat_by_name = {f["name"]: f for f in feats["features"]}
    padding_bank = feats.get("padding_bank") or []
    if len(padding_bank) != 5:
        sys.exit(f"Expected 5 padding_bank blocks, found {len(padding_bank)}.")

    with open(args.system, encoding="utf-8") as fh:
        system_prompt = fh.read().strip()

    seeds = read_csv(SEEDS)
    for need in ["post_id", "feature", "seed_label", "expected_variant_label", "note"]:
        if need not in seeds.columns:
            sys.exit(f"seeds file missing column: {need}")
    dup = seeds["post_id"][seeds["post_id"].duplicated()].tolist()
    if dup:
        sys.exit(f"Each seed may carry one feature only; duplicated post_ids: {dup}")
    missing = set(seeds["feature"]) - set(feat_by_name)
    if missing:
        sys.exit(f"Features in seeds but not in features.yaml: {missing}")
    for f in ["placebo", "padding"]:
        if f not in feat_by_name:
            sys.exit(f"features.yaml has no '{f}' entry.")

    master = read_csv(TEXTS)
    id_c = "post_id" if "post_id" in master.columns else "id"
    tx_c = "text_own" if "text_own" in master.columns else "text"
    master[id_c] = master[id_c].str.strip()
    text_by_id = dict(zip(master[id_c], master[tx_c]))
    absent = [p for p in seeds["post_id"] if p not in text_by_id]
    if absent:
        sys.exit(f"Seed post_ids not found in master: {absent}")

    provenance = {
        "features_sha256": sha256_file(FEATURES),
        "seeds_sha256": sha256_file(SEEDS),
        "system_prompt_sha256": sha256_text(system_prompt),
        "master_sha256": sha256_file(TEXTS),
        "features_version": feats.get("meta", {}).get("version", ""),
    }

    done = set()
    if os.path.exists(OUT_PATH):
        prev = read_csv(OUT_PATH)
        done = set(zip(prev["seed_id"], prev["variant_id"]))
        print(f"resuming: {len(done)} rows already in {OUT_PATH}")

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("Set OPENAI_API_KEY first.")
        from openai import OpenAI
        client = OpenAI()

    new_rows = []
    for _, s in seeds.iterrows():
        sid, feature, seed_label = s["post_id"], s["feature"], s["seed_label"]
        original = str(text_by_id[sid])
        base = {"seed_id": sid, "seed_label": seed_label, "assigned_feature": feature,
                "original_text": original, **provenance}

        jobs = [("original", "original", None, seed_label)]
        if feature == "question_form":
            jobs.append((feature, "llm", question_form_instruction(seed_label),
                         s["expected_variant_label"]))
        elif feature == "object_swap":
            jobs.append((feature, "llm", object_swap_instruction(s["note"]),
                         s["expected_variant_label"]))
            jobs.append(("object_swap_v2", "llm", object_swap_instruction_v2(s["note"]),
                         s["expected_variant_label"]))
        elif feat_by_name[feature].get("edit_instruction_by_label"):
            by_label = feat_by_name[feature]["edit_instruction_by_label"]
            instr = by_label.get(str(seed_label)) or by_label.get(int(seed_label))
            if not instr:
                sys.exit(f"{feature}: no edit instruction for seed label {seed_label}")
            target = str(s.get("target", "") or "").strip()
            if target:
                instr = (instr.strip() + "\nThe words this edit applies to (chosen from the codebook "
                         f"rule before generation): {target}")
            jobs.append((feature, "llm", instr.strip(), s["expected_variant_label"]))
        else:
            sys.exit(f"No instruction builder for feature {feature!r}")
        jobs.append(("placebo", "llm", feat_by_name["placebo"]["edit_instruction"].strip(),
                     seed_label))
        for k in range(len(padding_bank)):
            jobs.append((f"padding_{k + 1}", "padding", None, seed_label))

        for variant_id, vtype, instruction, expected in jobs:
            if args.only and variant_id != args.only:
                continue
            if not args.only and variant_id.endswith("_v2"):
                continue          # revisions run only when asked for explicitly
            if (sid, variant_id) in done:
                continue
            row = {**base, "variant_id": variant_id, "variant_type": vtype,
                   "edited_feature": variant_id.split("_")[0] if vtype == "padding"
                   else variant_id,
                   "expected_label": expected, "instruction": instruction or "",
                   "model": "", "attempts": "", "temperature": "", "status": "", "reason": "",
                   "words_changed_reported": "",
                   "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

            if vtype == "original":
                row.update(variant_text=original, status="OK")
            elif vtype == "padding":
                block = padding_bank[int(variant_id.split("_")[1]) - 1]
                if padding_mode == "clause":
                    body = original.rstrip()
                    end = body[-1] if body and body[-1] in ".!?" else ""
                    padded = body[:len(body) - len(end)] + ", " + block.strip().strip(",. ") + end
                else:
                    padded = original.rstrip() + PADDING_SEP + block.strip()
                row.update(variant_text=padded,
                           status="OK", instruction=f"append padding_bank[{variant_id[-1]}]")
            else:
                user_prompt = build_user_prompt(original, instruction, args.system)
                if args.dry_run:
                    print(f"\n=== {sid} / {variant_id} ===\n{user_prompt}")
                    continue
                parsed, raw, model_used, attempts, temp_used = call_llm(
                    client, system_prompt, user_prompt)
                with open(LOG_PATH, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({"seed_id": sid, "variant_id": variant_id,
                                         "timestamp": row["timestamp"], "model": model_used,
                                         "temperature": temp_used,
                                         "user_prompt": user_prompt, "raw": raw},
                                        ensure_ascii=False) + "\n")
                row.update(variant_text=parsed.get("variant", ""),
                           status=parsed.get("status", ""),
                           reason=parsed.get("reason", ""),
                           words_changed_reported=json.dumps(
                               parsed.get("words_changed", []), ensure_ascii=False),
                           model=model_used, attempts=attempts, temperature=temp_used)
                print(f"{sid} / {variant_id}: {row['status']}")

            row.update(auto_checks(original, row["variant_text"]))
            row["auto_flag"] = auto_flag(row)
            for c in CHECK_COLS:
                row[c] = ""
            new_rows.append(row)

    if args.dry_run:
        print("\nDry run finished; nothing written.")
        return
    if not new_rows:
        print("Nothing new to generate.")
        return

    out = pd.DataFrame(new_rows)
    if os.path.exists(OUT_PATH):
        out = pd.concat([read_csv(OUT_PATH), out.astype(str)], ignore_index=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    llm = out[out["variant_type"] == "llm"]
    print(f"\nwrote {len(out)} rows to {OUT_PATH}")
    print("LLM variants by status:")
    print(llm["status"].value_counts().to_string())
    print("LLM variants with an auto_flag:", int((llm["auto_flag"].fillna("") != "").sum()))


if __name__ == "__main__":
    main()
