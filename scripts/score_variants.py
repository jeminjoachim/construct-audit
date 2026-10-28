"""
score_variants.py — score pilot variants with the frozen Paper 1 classifier (v1).

What it does
  1. Verifies the model file by SHA-256 and checks its stored settings against textnorm.py.
  2. Calibration: re-scores reference posts whose scores this model produced before,
     under each pipeline option (embedding normalisation x aggregation), and keeps
     the one that reproduces them. Stops if none does. Use --no-calibration only
     if you have set both options yourself from train_chunk_mean.py.
  3. Scores originals, padding variants, and LLM variants you accepted
     (variant_checks.csv), plus each padding block on its own.
  4. Writes per-row scores (private) and a text-free summary table (shareable).

"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

VARIANTS = "data/private/variants_pilot.csv"
CHECKS = "data/private/variant_checks.csv"
FEATURES = "features.yaml"
OUT_ROWS = "data/private/scores_pilot.csv"
OUT_TABLE = "results/pilot_score_table.csv"
OUT_PADDING = "results/padding_blocks_alone.csv"
CALIB_TOL = 1e-4        # max abs difference accepted when reproducing reference scores
CALIB_N = 40            # reference posts used for calibration


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    return df


def load_textnorm(paper1_root):
    path = os.path.join(paper1_root, "textnorm.py")
    if not os.path.exists(path):
        sys.exit(f"textnorm.py not found at {path} (set paper1_root in config.yaml)")
    spec = importlib.util.spec_from_file_location("textnorm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha256(path)


class Scorer:
    def __init__(self, bundle, tn, encoder):
        self.clf = bundle["model"]
        self.tn = tn
        self.enc = encoder
        self._cache = {}

    def _chunk_embeddings(self, text, normalize):
        key = (text, normalize)
        if key not in self._cache:
            norm = self.tn.normalise(pd.Series([text])).iloc[0]
            chunks = self.tn.chunk_post(norm) or [norm]
            emb = self.enc.encode(chunks, normalize_embeddings=normalize,
                                  convert_to_numpy=True, show_progress_bar=False)
            self._cache[key] = np.asarray(emb, dtype=np.float32)
        return self._cache[key]

    def score(self, text, normalize, mode):
        emb = self._chunk_embeddings(text, normalize)
        if mode == "embedding":
            return float(self.clf.predict_proba(emb.mean(axis=0, keepdims=True))[0, 1]), len(emb)
        return float(self.clf.predict_proba(emb)[:, 1].mean()), len(emb)


def calibrate(scorer, cfg, master_text):
    opts_norm = [True, False] if cfg.get("normalize_embeddings", "auto") == "auto" \
        else [bool(cfg["normalize_embeddings"])]
    opts_mode = ["embedding", "probability"] if cfg.get("aggregation_mode", "auto") == "auto" \
        else [cfg["aggregation_mode"]]

    ref_path = cfg.get("reference_scores_csv")
    if not ref_path or not os.path.exists(ref_path):
        sys.exit("reference_scores_csv missing: needed to confirm the scoring pipeline "
                 "reproduces Paper 1. Use --no-calibration only with both options set.")
    if ref_path.lower().endswith(".parquet"):
        ref = pd.read_parquet(ref_path).astype(str)
    else:
        ref = read_csv(ref_path)
    id_col, score_col = cfg.get("reference_id_col", "post_id"), cfg.get("reference_score_col", "score")
    for col in (id_col, score_col):
        if col not in ref.columns:
            sys.exit(f"Column {col!r} not in reference file. Columns: {ref.columns.tolist()}")
    ref = ref[ref[score_col].isin(["", "nan", "None"]) == False]
    text_col = cfg.get("reference_text_col", "")
    if text_col and text_col in ref.columns:
        ref["text"] = ref[text_col]
    elif cfg.get("corpus_sqlite"):
        import sqlite3
        ref = ref.sample(min(CALIB_N, len(ref)), random_state=1).copy()
        ids = ref[id_col].tolist()
        con = sqlite3.connect(f"file:{cfg['corpus_sqlite']}?mode=ro", uri=True)
        q = (f"SELECT CAST({cfg['corpus_id_col']} AS TEXT), {cfg['corpus_text_col']} "
             f"FROM {cfg['corpus_table']} WHERE CAST({cfg['corpus_id_col']} AS TEXT) "
             f"IN ({','.join('?' * len(ids))})")
        found = dict(con.execute(q, ids).fetchall())
        con.close()
        ref["text"] = ref[id_col].map(found)
    else:
        ref["text"] = ref[id_col].map(master_text)
    ref = ref[ref["text"].fillna("") != ""]
    if len(ref) < 10:
        sys.exit(f"Only {len(ref)} reference posts have text available; need at least 10. "
                 "Set corpus_sqlite (or reference_text_col) in config.yaml.")
    ref = ref.sample(min(CALIB_N, len(ref)), random_state=1)
    target = ref[score_col].astype(float).to_numpy()

    nchunks = np.array([scorer.score(t, True, "embedding")[1] for t in ref["text"]])
    print(f"  reference: {len(ref)} posts, {int((nchunks > 1).sum())} with more than one chunk")
    results = []
    for normalize, mode in itertools.product(opts_norm, opts_mode):
        got = np.array([scorer.score(t, normalize, mode)[0] for t in ref["text"]])
        diff = got - target
        results.append((normalize, mode, float(np.max(np.abs(diff)))))
        print(f"  calibration: normalize={normalize!s:<5} aggregation={mode:<11} "
              f"max |diff| = {results[-1][2]:.2e}  median |diff| = {np.median(np.abs(diff)):.2e}  "
              f"corr = {np.corrcoef(got, target)[0, 1]:.3f}  mean diff = {diff.mean():+.3f}")
    best = min(results, key=lambda r: r[2])
    if best[2] > CALIB_TOL:
        sys.exit(f"No option reproduces the reference scores (best max |diff| {best[2]:.2e} > "
                 f"{CALIB_TOL}). Check paper1_root, the model file, the reference file, and "
                 "the sentence-transformers version before scoring anything.")
    print(f"  -> using normalize={best[0]}, aggregation={best[1]} on {len(ref)} reference posts\n")
    return best[0], best[1], len(ref), best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-calibration", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # 1. model integrity
    model_path = cfg["model_path"]
    got_sha = sha256(model_path)
    if got_sha != cfg["model_sha256"]:
        sys.exit(f"Model hash mismatch:\n  expected {cfg['model_sha256']}\n  got      {got_sha}")
    bundle = joblib.load(model_path)
    tn, tn_sha = load_textnorm(cfg["paper1_root"])
    for key, tn_val in [("chunk_words", tn.CHUNK_WORDS), ("chunk_overlap", tn.CHUNK_OVERLAP),
                        ("min_chunk_words", tn.MIN_CHUNK_WORDS)]:
        if bundle.get(key) != tn_val:
            sys.exit(f"{key}: model says {bundle.get(key)}, textnorm.py says {tn_val}")
    if bundle.get("aggregation") != "mean":
        sys.exit(f"Model aggregation is {bundle.get('aggregation')!r}, script supports 'mean' only")
    import sklearn
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    print(f"model OK ({got_sha[:16]}), textnorm {tn.NORM_VERSION} ({tn_sha[:12]}), "
          f"scikit-learn {sklearn.__version__}, sentence-transformers {sentence_transformers.__version__}")
    if sklearn.__version__ != "1.9.0":
        print("  WARNING: model was saved with scikit-learn 1.9.0")

    encoder = SentenceTransformer(bundle["embed_model"], device=cfg.get("device", "cpu"))
    scorer = Scorer(bundle, tn, encoder)

    master = read_csv(cfg.get("master_path", "data/private/master_coded_v2.csv"))
    master_text = dict(zip(master["post_id"].str.strip(), master["text_own"]))

    # 2. calibration
    if args.no_calibration:
        if cfg.get("normalize_embeddings", "auto") == "auto" or \
                cfg.get("aggregation_mode", "auto") == "auto":
            sys.exit("--no-calibration needs normalize_embeddings and aggregation_mode set explicitly")
        normalize, mode, n_cal, cal_err = bool(cfg["normalize_embeddings"]), \
            cfg["aggregation_mode"], 0, float("nan")
        print("  calibration skipped (options set by hand)\n")
    else:
        normalize, mode, n_cal, cal_err = calibrate(scorer, cfg, master_text)

    # 3. select rows
    v = read_csv(VARIANTS)
    c = read_csv(CHECKS)
    v = v.drop(columns=[x for x in v.columns if x.startswith("chk_")])   # empty in the generated file
    v = v.merge(c[["seed_id", "variant_id", "chk_verdict", "chk_my_label"]],
                on=["seed_id", "variant_id"], how="left")
    v["chk_verdict"] = v["chk_verdict"].fillna("")
    keep = (v["variant_type"].isin(["original", "padding"])) | \
           (v["chk_verdict"].isin(["accept", "authored"]))
    v = v[keep].copy()
    v["label_truth"] = np.where(v["variant_type"] == "llm", v["chk_my_label"], v["seed_label"])
    v["include_primary"] = (v["variant_type"] != "llm") | \
        ((v["chk_verdict"] == "accept") & (v["auto_flag"].fillna("") == ""))
    v["exploratory_feature"] = v["edited_feature"].str.startswith("object_swap")

    scores, nchunks = [], []
    for t in v["variant_text"]:
        s, n = scorer.score(t, normalize, mode)
        scores.append(s)
        nchunks.append(n)
    v["score"], v["n_chunks"] = scores, nchunks
    v["wc"] = tn.wordcount(v["variant_text"]).to_numpy()

    prov = {"model_sha256": got_sha, "textnorm_sha256": tn_sha,
            "embed_model": bundle["embed_model"], "normalize_embeddings": normalize,
            "aggregation_mode": mode, "calibration_n": n_cal,
            "calibration_max_abs_diff": cal_err, "sklearn": sklearn.__version__,
            "sentence_transformers": sentence_transformers.__version__}
    for k, val in prov.items():
        v[k] = val

    cols = ["seed_id", "seed_label", "assigned_feature", "variant_id", "variant_type",
            "edited_feature", "expected_label", "label_truth", "chk_verdict", "auto_flag",
            "include_primary", "exploratory_feature", "wc", "n_chunks", "score",
            "variant_text"] + list(prov)
    v[cols].to_csv(OUT_ROWS, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    # 4. summary table (no post text)
    os.makedirs("results", exist_ok=True)
    base = v[v["variant_id"] == "original"].set_index("seed_id")["score"]
    plac = v[v["variant_id"] == "placebo"].set_index("seed_id")["score"]
    v["score_original"] = v["seed_id"].map(base)
    v["score_placebo"] = v["seed_id"].map(plac)
    v["delta_vs_original"] = v["score"] - v["score_original"]
    v["delta_vs_placebo"] = v["score"] - v["score_placebo"]
    table = v[v["variant_id"] != "original"][[
        "seed_id", "seed_label", "variant_id", "edited_feature", "expected_label",
        "label_truth", "include_primary", "exploratory_feature", "wc", "n_chunks",
        "score_original", "score_placebo", "score", "delta_vs_original", "delta_vs_placebo"]]
    table.round(4).to_csv(OUT_TABLE, index=False)

    # padding blocks scored alone
    with open(FEATURES, encoding="utf-8") as fh:
        bank = yaml.safe_load(fh).get("padding_bank", [])
    pad = pd.DataFrame({"block": range(1, len(bank) + 1),
                        "score_alone": [scorer.score(b, normalize, mode)[0] for b in bank],
                        "wc": tn.wordcount(pd.Series(bank)).to_numpy()})
    pad.round(4).to_csv(OUT_PADDING, index=False)

    pd.set_option("display.width", 160)
    print(f"wrote {len(v)} scored rows to {OUT_ROWS}")
    print(f"wrote summary to {OUT_TABLE} and {OUT_PADDING}\n")
    show = table.copy()
    show["edit"] = show["edited_feature"]
    print("Per-variant shifts (primary rows):")
    print(show[show["include_primary"]][["seed_id", "variant_id", "expected_label",
                                         "label_truth", "score_original", "score",
                                         "delta_vs_original", "delta_vs_placebo"]]
          .round(3).to_string(index=False))
    print("\nMean shift vs original, by edit (primary rows):")
    print(show[show["include_primary"]].groupby("edit")["delta_vs_original"]
          .agg(["count", "mean", "min", "max"]).round(3).to_string())
    print("\nPadding blocks scored alone:")
    print(pad.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
