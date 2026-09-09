"""Run the significance and variance analysis over the per-item results.

Answers two questions the accuracy tables cannot:

  1. Are the gaps between models real, or sampling noise? The models answered
     identical items, so the comparison is paired and McNemar's test applies.
  2. Which of the things that vary across items actually moves the outcome?
     Chi-square per factor with an effect size, plus a variance decomposition
     that separates real subject difficulty from binomial noise.

Writes results/v2/significance_<benchmark>.json. Run after run_benchmark.py:

    python analyse_significance.py
"""

import glob
import json
import os
import sys

import config
from metrics import significance as S
from metrics import tokenization as T

RESULTS = "results"
OUT_DIR = os.path.join("results", "v2")

SHORT = {
    "google/gemma-3-4b-it": "Gemma-3-4B",
    "meta-llama/llama-3.2-3b-instruct": "Llama-3.2-3B",
    "mistralai/ministral-8b-2512": "Ministral-8B",
}


def short(model):
    return SHORT.get(model, model.split("/")[-1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_items(benchmark):
    """Per-item records per model, verified item-aligned.

    Every paired test here assumes results[i] refers to the same item for
    every model. That is checked rather than trusted: a silent misalignment
    would produce confident, wrong p-values.
    """
    suffix = "_mmlu.json" if benchmark == "mmlu" else "_lambada_test.json"
    models = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*" + suffix))):
        if "summary" in os.path.basename(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "results" in data:
            models[data["model"]] = data["results"]

    if len(models) < 2:
        return None, None

    key = ((lambda r: (r.get("subject"), r.get("index")))
           if benchmark == "mmlu" else (lambda r: r.get("index")))
    keys = [[key(r) for r in rows] for rows in models.values()]
    if any(k != keys[0] for k in keys):
        raise SystemExit(f"{benchmark}: per-item records are not aligned across "
                         f"models; paired tests would be invalid")

    correctness = {short(m): [bool(r.get("correct")) for r in rows]
                   for m, rows in models.items()}
    return correctness, models


def lambada_passages(n_items, seed=42):
    """Rebuild the exact split the run used, for true passage lengths.

    The per-item records store only a truncated preview, so length has to come
    back from the dataset. load_dataset is deterministic given (n, seed), and
    the reconstruction is verified against the stored targets below.
    """
    import evaluate_lambada as E
    return E.load_dataset(config.DATASET_FILES["test"], n_items, seed)


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def mmlu_factors(rows_by_model):
    """Group correctness by the things that vary across MMLU items."""
    factors = {}
    any_rows = next(iter(rows_by_model.values()))

    # Pool across models: the question is whether a factor moves accuracy at
    # all, not whether it moves it for one model.
    for level_key, name in (("category", "subject category"),
                            ("subject", "subject")):
        groups = {}
        for rows in rows_by_model.values():
            for r in rows:
                groups.setdefault(r.get(level_key) or "unknown", []).append(
                    bool(r.get("correct")))
        factors[name] = groups

    # Position of the correct answer. A model with no option-position bias
    # should be flat here; a slope is evidence of letter preference.
    groups = {}
    for rows in rows_by_model.values():
        for r in rows:
            letter = r.get("correct_letter")
            if letter:
                groups.setdefault(f"answer = {letter}", []).append(
                    bool(r.get("correct")))
    if groups:
        factors["correct-option position"] = groups

    del any_rows
    return factors


def lambada_factors(rows_by_model, passages):
    """Group correctness by true passage length and by target fragmentation."""
    factors = {}

    # True passage length, in quartiles. NOT the stored preview: that is cut to
    # a fixed character budget, so its word count correlates -0.20 with real
    # length and bucketing by it is worse than not bucketing at all.
    lengths = sorted({passages[r["index"]]["context_words"]
                      for rows in rows_by_model.values() for r in rows})
    cuts = [lengths[int(len(lengths) * q)] for q in (0.25, 0.5, 0.75)]

    def bucket(w):
        if w <= cuts[0]:
            return f"Q1 <={cuts[0]}w"
        if w <= cuts[1]:
            return f"Q2 {cuts[0] + 1}-{cuts[1]}w"
        if w <= cuts[2]:
            return f"Q3 {cuts[1] + 1}-{cuts[2]}w"
        return f"Q4 >{cuts[2]}w"

    groups = {}
    for rows in rows_by_model.values():
        for r in rows:
            groups.setdefault(bucket(passages[r["index"]]["context_words"]),
                              []).append(bool(r.get("correct")))
    factors["passage length (true)"] = dict(sorted(groups.items()))

    # Target fragmentation, per model, since each has its own tokenizer.
    per_model = {}
    for model_short, rows in rows_by_model.items():
        tok = TOKENIZERS.get(model_short)
        if tok is None:
            continue
        groups = {}
        for r in rows:
            n = T.tokens_for_word(r["target"], tok)
            key = "1 token" if n <= 1 else "2 tokens" if n == 2 else "3+ tokens"
            groups.setdefault(key, []).append(bool(r.get("correct")))
        if groups:
            per_model[model_short] = dict(sorted(groups.items()))
    return factors, per_model


TOKENIZERS = {}


# ---------------------------------------------------------------------------

def analyse(benchmark):
    correctness, rows_by_model = load_items(benchmark)
    if not correctness:
        print(f"  {benchmark}: fewer than two models with per-item results")
        return None

    n = len(next(iter(correctness.values())))
    print(f"  {benchmark}: {len(correctness)} models x {n:,} items")

    out = {
        "benchmark": benchmark,
        "n_items": n,
        "models": list(correctness),
        "accuracy": {m: round(sum(v) / len(v), 4) for m, v in correctness.items()},
        "omnibus": S.cochrans_q(correctness),
        "pairwise_mcnemar": S.pairwise_mcnemar(correctness),
        "error_overlap": S.error_overlap(correctness),
        "oracle": S.oracle_ceiling(correctness),
        "factors": {},
    }

    short_rows = {short(m): rows for m, rows in rows_by_model.items()}

    if benchmark == "mmlu":
        factors = mmlu_factors(short_rows)
        for name, groups in factors.items():
            if name == "subject":
                # 57 levels is too many for a readable chi-square table; the
                # useful question there is how much of the spread is real.
                out["subject_variance"] = S.variance_decomposition(groups)
                # Per-model as well, since a model can be uniformly weak or
                # spiky across subjects and the composite hides which.
                out["subject_variance_by_model"] = {
                    m: S.variance_decomposition(
                        _group(rows, "subject")) for m, rows in short_rows.items()}
            else:
                out["factors"][name] = S.chi2_independence(groups)
    else:
        passages = lambada_passages(n)
        for m, rows in short_rows.items():
            mismatch = sum(1 for r in rows
                           if passages[r["index"]]["target"] != r["target"])
            if mismatch:
                raise SystemExit(
                    f"LAMBADA reconstruction mismatch for {m}: {mismatch} of "
                    f"{len(rows)} targets differ. Passage lengths would be "
                    f"wrong; refusing to report length analysis.")
        for p in passages:
            p["context_words"] = len(p["context"].split())

        for m in short_rows:
            TOKENIZERS[m] = T.get_tokenizer(
                next(k for k in SHORT if SHORT[k] == m))

        factors, per_model = lambada_factors(short_rows, passages)
        for name, groups in factors.items():
            out["factors"][name] = S.chi2_independence(groups)
        out["fragmentation_by_model"] = {
            m: S.chi2_independence(g) for m, g in per_model.items()}
        out["length_note"] = (
            "passage length is measured from the reconstructed dataset split "
            "(deterministic given n and seed, and verified target-by-target "
            "against the stored results), not from the truncated preview held "
            "in the per-item records")

    return out


def _group(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(r.get(key) or "unknown", []).append(bool(r.get("correct")))
    return groups


def repair_context_blocks():
    """Recompute LAMBADA position sensitivity from true passage lengths.

    The stored blocks were built from `context_preview`, which is truncated to
    a fixed character budget - so the quartiles ranked items by mean word
    length, not passage length. Runs from here on store `context_words`; these
    files predate that, so the lengths are recovered from the reconstructed
    split and the block is rewritten in place.
    """
    from metrics import context as C

    paths = sorted(glob.glob(os.path.join(OUT_DIR, "*_lambada_v2.json")))
    if not paths:
        return
    print("\nRepairing context blocks (true passage length)")

    _, rows_by_model = load_items("lambada")
    if not rows_by_model:
        return
    n = len(next(iter(rows_by_model.values())))
    passages = lambada_passages(n)

    for path in paths:
        with open(path, encoding="utf-8") as f:
            blocks = json.load(f)
        rows = rows_by_model.get(blocks.get("model"))
        if not rows:
            continue
        if any(passages[r["index"]]["target"] != r["target"] for r in rows):
            print(f"    {os.path.basename(path)}: reconstruction mismatch, skipped")
            continue

        enriched = [dict(r, context_words=len(passages[r["index"]]["context"].split()))
                    for r in rows]
        fixed = C.position_sensitivity(enriched)
        old = ((blocks.get("context") or {}).get("position_sensitivity") or {})
        blocks.setdefault("context", {})["position_sensitivity"] = fixed
        blocks["context"]["position_sensitivity"]["superseded_note"] = (
            "recomputed from true passage length; the earlier value bucketed by "
            "the truncated context_preview and is not comparable")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)
        print(f"    {short(blocks['model']):<14} span {old.get('span')} "
              f"-> {fixed.get('span')} (preview-based -> true length)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Significance analysis")
    written = 0
    for benchmark in ("mmlu", "lambada"):
        result = analyse(benchmark)
        if not result:
            continue
        path = os.path.join(OUT_DIR, f"significance_{benchmark}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"    wrote {path}")
        written += 1

        q = result["omnibus"]
        if q.get("available"):
            print(f"    Cochran's Q = {q['q_statistic']} (df {q['df']}), "
                  f"p = {S.format_p(q['p_value'])}")
        for r in result["pairwise_mcnemar"]:
            print(f"    {r['model_a']} vs {r['model_b']}: "
                  f"delta {r['accuracy_delta']:+.4f}, "
                  f"p_adj = {S.format_p(r['p_adjusted'])} "
                  f"{'SIGNIFICANT' if r['significant_adjusted'] else 'n.s.'}")

    repair_context_blocks()
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
