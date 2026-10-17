"""
score_envclaims.py — score with the published climatebert/environmental-claims model.

  --reproduce   Score the test split and compare with the paper's Table 2 (ClimateBERT,
                test: P 76.5, R 92.5, F1 83.8, acc 90.9). Run this first. If test accuracy
                is near 100%, the released model saw the test split and seeds are in-sample.
  --variants    Score a generate_variants.py output file (originals, padding, accepted LLM
                variants) and write a text-free shift table, same layout as the forum pilot.
  --doclevel    Document-level test: pseudo-reports built from test sentences, with neutral
                non-claim sentences added. Compares presence and share measures.

    pip install transformers torch datasets
    python scripts\\score_envclaims.py --reproduce
    python scripts\\score_envclaims.py --variants
    python scripts\\score_envclaims.py --doclevel
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd

MODEL_ID = "climatebert/environmental-claims"
DATA = "data/envclaims"
VARIANTS = f"{DATA}/variants_envclaims.csv"
CHECKS = f"{DATA}/variant_checks_envclaims.csv"
RESULTS = "results/envclaims"
PAPER_TEST = {"precision": 76.5, "recall": 92.5, "f1": 83.8, "accuracy": 90.9}
REPRO_TOL = 2.5        # percentage points
ENVWORD = re.compile(r"\b(?:environment\w*|climate|carbon|emission\w*|energy|green|renewable|"
                     r"ecolog\w*|biodivers\w*|waste|water|pollut\w*|sustainab\w*|responsib\w*)\b", re.I)


class Model:
    def __init__(self, model_id, revision=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.m = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval()
        id2label = {int(k): str(v).lower() for k, v in self.m.config.id2label.items()}
        pos = [k for k, v in id2label.items() if v in ("yes", "1", "label_1", "claim", "environmental claim")]
        self.pos = pos[0] if pos else max(id2label)
        self.revision = getattr(self.m.config, "_commit_hash", None) or revision or ""
        print(f"model {model_id} (revision {self.revision}); labels {id2label}; positive = {self.pos}")

    def proba(self, texts, batch=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(list(texts[i:i + batch]), padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
                p = self.torch.softmax(self.m(**enc).logits, dim=-1)[:, self.pos]
                out.extend(p.cpu().numpy().tolist())
        return np.array(out)


def metrics(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": 100 * p, "recall": 100 * r, "f1": 100 * f1,
            "accuracy": 100 * float((y == yhat).mean()), "fpr": 100 * fp / max(1, int((y == 0).sum()))}


def reproduce(model):
    rows = []
    for split in ("train", "validation", "test"):
        path = f"{DATA}/{split}.csv"
        if not os.path.exists(path):
            sys.exit(f"{path} missing: run envclaims_prepare.py first")
        d = pd.read_csv(path)
        p = model.proba(d["text"].tolist())
        d["score"] = p
        d.to_csv(f"{DATA}/{split}_scored.csv", index=False)
        rows.append({"split": split, "n": len(d), **{k: round(v, 1) for k, v in
                                                     metrics(d["label"], (p >= 0.5).astype(int)).items()}})
    r = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    r.to_csv(f"{RESULTS}/reproduction.csv", index=False)
    print(r.to_string(index=False))
    test = r[r["split"] == "test"].iloc[0]
    gaps = {k: test[k] - v for k, v in PAPER_TEST.items()}
    print("\npaper test (ClimateBERT):", PAPER_TEST)
    print("difference:", {k: round(v, 1) for k, v in gaps.items()})
    if test["accuracy"] > 98:
        print("WARNING: test accuracy near 100% -> the released model was probably trained on the "
              "test split. Treat all seeds as in-sample and compare against placebo, not original.")
    elif max(abs(v) for v in gaps.values()) <= REPRO_TOL:
        print(f"OK: reproduces the paper's test results within {REPRO_TOL} points.")
    else:
        print("MISMATCH: the released model does not reproduce Table 2. Record the revision "
              "hash and report the reproduced numbers, not the paper's.")


def variants(model):
    v = pd.read_csv(VARIANTS, dtype=str, keep_default_na=False)
    if os.path.exists(CHECKS):
        c = pd.read_csv(CHECKS, dtype=str, keep_default_na=False)
        v = v.drop(columns=[x for x in v.columns if x.startswith("chk_")])
        v = v.merge(c[["seed_id", "variant_id", "chk_verdict", "chk_my_label"]],
                    on=["seed_id", "variant_id"], how="left").fillna("")
    else:
        print(f"WARNING: {CHECKS} not found; scoring LLM variants without manual checks "
              "(do the checks before interpreting anything).")
        v["chk_verdict"], v["chk_my_label"] = "", ""
    have_checks = os.path.exists(CHECKS)
    llm_ok = (v["variant_type"] == "llm") & (v["status"] == "OK")
    keep = v["variant_type"].isin(["original", "padding"]) | \
        (v["chk_verdict"].isin(["accept", "authored"]) if have_checks else llm_ok)
    v = v[keep].copy()
    v["score"] = model.proba(v["variant_text"].tolist())
    v["label_truth"] = np.where(v["variant_type"] == "llm", v["chk_my_label"], v["seed_label"])
    v["include_primary"] = (v["variant_type"] != "llm") | ((v["chk_verdict"] == "accept")
                                                            & (v["auto_flag"] == ""))
    base = v[v["variant_id"] == "original"].set_index("seed_id")["score"]
    plac = v[v["variant_id"] == "placebo"].set_index("seed_id")["score"]
    v["score_original"] = v["seed_id"].map(base)
    v["score_placebo"] = v["seed_id"].map(plac)
    v["delta_vs_original"] = v["score"] - v["score_original"]
    v["delta_vs_placebo"] = v["score"] - v["score_placebo"]
    v["model_revision"] = model.revision
    os.makedirs(RESULTS, exist_ok=True)
    v.to_csv(f"{DATA}/scores_envclaims.csv", index=False, quoting=csv.QUOTE_ALL)
    cols = ["seed_id", "seed_label", "variant_id", "edited_feature", "expected_label",
            "label_truth", "include_primary", "score_original", "score_placebo", "score",
            "delta_vs_original", "delta_vs_placebo", "model_revision"]
    table = v[v["variant_id"] != "original"][cols]
    table.round(4).to_csv(f"{RESULTS}/envclaims_score_table.csv", index=False)
    prim = table[table["include_primary"]]
    print(prim.drop(columns="model_revision").round(3).to_string(index=False))
    print("\nMean shift vs original, by edit (primary rows):")
    print(prim.groupby("edited_feature")["delta_vs_original"].agg(["count", "mean", "min", "max"])
          .round(3).to_string())


def doclevel(model, n_docs=200, doc_len=10, seed=1):
    path = f"{DATA}/test_scored.csv"
    if not os.path.exists(path):
        sys.exit("run --reproduce first (it writes test_scored.csv)")
    d = pd.read_csv(path)
    rng = np.random.default_rng(seed)
    pos = d[d["label"] == 1]
    neg = d[d["label"] == 0]
    neutral = neg[~neg["text"].str.contains(ENVWORD)]          # no environmental vocabulary
    print(f"test: {len(pos)} claims, {len(neg)} non-claims, {len(neutral)} neutral non-claims "
          "(no environmental vocabulary) used as padding")
    rows = []
    for k_claims in (1, 3):
        for pad_pool, pool_name in ((neutral, "neutral"), (neg, "any_nonclaim")):
            for n_pad in (0, 5, 10, 20):
                for doc in range(n_docs):
                    cl = pos.sample(k_claims, random_state=int(rng.integers(1e9)))
                    nc = neg.sample(doc_len - k_claims, random_state=int(rng.integers(1e9)))
                    pad = pad_pool.sample(n_pad, random_state=int(rng.integers(1e9))) if n_pad else pad_pool.iloc[:0]
                    s = np.concatenate([cl["score"], nc["score"], pad["score"]])
                    rows.append({"k_claims": k_claims, "padding_pool": pool_name, "n_padding": n_pad,
                                 "true_share": k_claims / (doc_len + n_pad),
                                 "cc_share": float((s >= 0.5).mean()),
                                 "presence_max": float(s.max() >= 0.5),
                                 "presence_mean": float(s.mean() >= 0.5),
                                 "mean_score": float(s.mean()), "max_score": float(s.max())})
    r = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    summ = (r.groupby(["k_claims", "padding_pool", "n_padding"])
             [["true_share", "cc_share", "mean_score", "max_score", "presence_max", "presence_mean"]]
             .mean().round(3).reset_index())
    summ["cc_bias"] = (summ["cc_share"] - summ["true_share"]).round(3)
    summ.to_csv(f"{RESULTS}/doclevel_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nRead: presence_max should stay ~1.0 as padding grows (the report still contains a claim);"
          "\nmean_score / presence_mean show what an averaging measure does; cc_bias shows how a"
          "\nclassify-and-count share drifts as the true share falls.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--doclevel", action="store_true")
    ap.add_argument("--revision", default=None, help="pin a model commit hash")
    a = ap.parse_args()
    if not (a.reproduce or a.variants or a.doclevel):
        ap.print_help(); return
    model = Model(MODEL_ID, a.revision)
    if a.reproduce:
        reproduce(model)
    if a.variants:
        variants(model)
    if a.doclevel:
        doclevel(model)


if __name__ == "__main__":
    main()
