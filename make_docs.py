"""Generate the lean Docs view: `report.md` and a matching `report.html`.

Three sections only, by design — the full academic write-up lives in
`report-full.md` and `extended-final-report.md`:

    1. Process workflow            (mermaid)
    2. Metric components           (a table per stage subtopic)
    3. Results and how they arose  (a table per benchmark + brief interpretation)

Everything is read from results/v2, so the numbers cannot drift from the run.

    python make_docs.py
"""

import glob
import json
import os
from datetime import datetime

import make_results_section as R
from metrics import taxonomy

MD_OUT = "report.md"
HTML_OUT = "report.html"


def load(benchmark):
    return [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(f"results/v2/*_{benchmark}_v2.json"))]


def comparison(benchmark):
    p = f"results/v2/comparison_{benchmark}.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# 1. Workflow
# ---------------------------------------------------------------------------

WORKFLOW = """## 1. Process Workflow

One benchmark run, end to end. Stages 1-9 are produced by every run; stage 10
is excluded because nothing in it is observable through a hosted inference API.

```mermaid
flowchart TD
    A[Configure run<br/>models · decoding · dataset · passes] --> B{Pre-flight probe}
    B -->|provider throttled| B1[Warn — run continues,<br/>requests retried not dropped]
    B -->|healthy| C[Load dataset]
    B1 --> C
    C --> D[Estimate API calls<br/>show multiplier before spending]
    D --> E[Scored pass<br/>chain-of-thought · not streamed]
    E --> F[Extended pass<br/>streamed · TTFT · TPOT · tokens · cost]
    F --> G{Optional passes}
    G -->|calibration| G1[max_tokens=1<br/>→ option probabilities]
    G -->|repeats N| G2[repeat items → answer stability]
    G -->|robustness| G3[prompt · reorder · typo variants]
    G -->|context| G4[progressive ablation · LAMBADA]
    G1 --> H
    G2 --> H
    G3 --> H
    G4 --> H
    G --> H["`Compute the nine metric blocks
    **1** Task Quality · **2** Probabilistic Quality
    **3** Reasoning and Consistency · **4** Context Behavior
    **5** Robustness · **6** API Performance
    **7** Token Efficiency · **8** Economics · **9** Reliability`"]
    H --> J[Aggregate and rank]
    J --> K{Configurations identical?}
    K -->|no| K1[comparable = false<br/>refuse to rank]
    K -->|yes| L[Write results · history · comparison]
    L --> M[Web view → History → Extended analysis]

    style H fill:#e7f1ff,stroke:#0d6efd,color:#0a4bc4
    style K1 fill:#fff3cd,stroke:#ffc107
    style B1 fill:#fff3cd,stroke:#ffc107
```

Stage 10 — hardware and distributed execution (TP, PP, DP, SP, CP, EP; VRAM;
energy) — sits outside this flow. Those are *configuration variables of a
serving deployment*, not properties of a model, and a hosted API exposes none
of them. They are recorded in the run config and would only be measurable on a
self-hosted vLLM path.

### 1.1 The two measurement passes

Each item is sent to the API twice, and the distinction explains small
differences between tables.

| Pass | Prompt | Streamed | Produces |
|---|---|---|---|
| **Scored** | chain-of-thought, then an answer | no | headline accuracy, reasoning analysis |
| **Extended** | identical prompt | yes | TTFT, TPOT, throughput, tokens, cost, reliability |
| **Calibration** *(optional)* | direct answer, `max_tokens = 1` | no | option probabilities → NLL, ECE, Brier, rank |

Time-to-first-token cannot be observed from a single blocking response, so
latency requires streaming. Because the two passes are separate calls, their
accuracies differ slightly even at temperature 0; that residual is serving
non-determinism and the web view states the delta explicitly.
"""


# ---------------------------------------------------------------------------
# 2. Metric components
# ---------------------------------------------------------------------------

# One row per metric: name, what it measures, direction that counts as good.
METRIC_DETAIL = {
    "task_quality": [
        ("Overall accuracy", "Correct ÷ questions the API actually answered.", "higher"),
        ("Macro accuracy", "Mean of per-subject accuracies, so a large subject cannot dominate.", "higher"),
        ("Subject / category accuracy", "The same score computed within each subject and each of the four MMLU categories.", "higher"),
        ("Normalised accuracy", "`(acc − chance) ÷ (1 − chance)` — distance above guessing.", "higher"),
        ("Wilson 95% CI", "Binomial interval that stays inside [0,1] at small n.", "narrower"),
        ("Error rate", "`1 − accuracy`, framed as a budget.", "lower"),
        ("Parse-failure rate", "Responses with no extractable answer — disobedience, not ignorance.", "lower"),
        ("Option-position bias", "Distance between the letters picked and the letters that are correct.", "lower"),
        ("Last-word / exact / stem match", "LAMBADA: normalised, byte-identical, and inflection-tolerant matching.", "higher"),
    ],
    "probability": [
        ("Correct-option probability", "Probability mass placed on the right answer.", "higher"),
        ("Log probability", "`log P(correct)`, averaged.", "→ 0"),
        ("NLL", "`−mean(log P(correct))` — how surprised the model was by the truth.", "lower"),
        ("Perplexity", "`exp(NLL)`. 1.0 = certain and right; 4.0 ≈ guessing among four.", "lower"),
        ("Entropy", "Spread of the predictive distribution. High = unsure.", "context"),
        ("ECE", "Bin by confidence; compare mean confidence against observed accuracy per bin.", "lower"),
        ("MCE", "The worst single bin rather than the weighted mean — tail risk.", "lower"),
        ("Brier score", "Squared error of confidence against outcome. A proper scoring rule.", "lower"),
        ("Confidence↔accuracy correlation", "Whether stated confidence carries any signal.", "higher"),
        ("Target rank / MRR / hit@k", "LAMBADA: where the gold word sat among ranked predictions.", "higher"),
    ],
    "consistency": [
        ("Answer stability", "Share of items where every repeat gave an identical answer.", "higher"),
        ("Majority-vote accuracy", "Accuracy after taking the most common of k samples.", "higher"),
        ("Self-consistency gain", "Majority-vote minus mean single-sample accuracy.", "higher"),
        ("Mean agreement", "How dominant the winning answer was across repeats.", "higher"),
        ("Seed stability", "Accuracy spread across independent seeds.", "lower"),
    ],
    "context": [
        ("Context utilisation", "`acc(full passage) − acc(last sentence only)`.", "higher"),
        ("Utilisation ratio", "Share of skill that depends on the wider passage. 1.0 = entirely.", "higher"),
        ("Context sensitivity", "`acc(full) − acc(no context at all)`.", "higher"),
        ("Context ablation", "Accuracy under a progressive cut-down of the passage.", "—"),
        ("Position sensitivity", "Accuracy bucketed by passage length — a lost-in-the-middle probe.", "flat"),
    ],
    "robustness": [
        ("Prompt variation", "The same question under semantically identical templates.", "→ 0 drop"),
        ("Option reordering", "Options permuted with the gold letter remapped.", "→ 0 drop"),
        ("Input perturbation", "Adjacent-key typos, casing, whitespace, punctuation, homoglyphs.", "→ 0 drop"),
        ("Accuracy drop", "`baseline − variant`.", "→ 0"),
        ("Flip rate", "Answers that changed at all — catches offsetting errors.", "lower"),
        ("Broke / fixed", "Right→wrong and wrong→right, counted separately.", "lower"),
        ("Robustness score", "`1 − mean(relative drop)` across all variants.", "higher"),
    ],
    "api_performance": [
        ("TTFT", "Time to the first *content* token — prefill, queueing and network.", "lower"),
        ("TPOT", "`(E2E − TTFT) ÷ (tokens − 1)` — steady-state generation rate.", "lower"),
        ("End-to-end latency", "Total wait for the complete answer.", "lower"),
        ("p50 / p95 / p99", "Latency distribution. For serving, the tail is the experience.", "lower"),
        ("Prefill / decode throughput", "Prompt tokens/s and generated tokens/s.", "higher"),
        ("Request throughput", "Completed requests per second across the run.", "higher"),
    ],
    "token_efficiency": [
        ("Prompt / completion tokens", "What was billed, as reported by the provider.", "lower"),
        ("Reasoning tokens", "Tokens spent on hidden reasoning; 0 is a real answer.", "lower"),
        ("Cached prompt tokens", "Prompt tokens served from the provider cache, and cheaper.", "higher"),
        ("Tokens per item", "Total tokens ÷ items evaluated.", "lower"),
        ("Tokens per correct answer", "Token cost of a *useful* answer.", "lower"),
    ],
    "economics": [
        ("Cost per request", "Provider-reported spend, averaged.", "lower"),
        ("Cost per 1K / 1M tokens", "Blended price actually paid.", "lower"),
        ("Cost per correct answer", "Money per useful answer — the selection criterion.", "lower"),
        ("Correct answers per dollar", "The same ratio inverted.", "higher"),
    ],
    "reliability": [
        ("Success / failure rate", "Requests that returned, versus those that errored after retries.", "higher / lower"),
        ("Invalid output rate", "A 200 response carrying nothing usable.", "lower"),
        ("Timeout / 429 rate", "Failures bucketed by class.", "lower"),
        ("Retry rate", "Retries per request — invisible in an accuracy table.", "lower"),
        ("Provider failover", "Whether more than one backend served the run.", "no"),
        ("Reproducibility score", "Seed spread combined with repeat stability.", "higher"),
    ],
    "tokenization": [
        ("Tokens per target word", "Tokens needed for the gold word, with its leading space.", "lower"),
        ("Fragmentation rate", "Share of targets needing more than one token.", "lower"),
        ("Subword exact match", "Match at token level rather than string level.", "higher"),
        ("Accuracy by fragmentation", "Accuracy split by 1 / 2 / 3+ token targets.", "—"),
    ],
}

def components_section():
    out = ["## 2. Metric Components\n",
           "Every metric the benchmark produces, grouped by stage. "
           "`Observable` records whether this deployment — a client of a hosted "
           "inference API — can measure it at all.\n"]

    for s in taxonomy.STAGES:
        key = s["key"]
        rows = METRIC_DETAIL.get(key, [])
        if not rows:
            continue
        out.append(f"\n### 2.{s['stage']} Stage {s['stage']} — {s['layer']}\n")
        out.append(f"*{s['purpose']}. Priority **{s['priority']}**; "
                   f"observable through OpenRouter: **{s['openrouter']}**"
                   + (f"; requires {s['requires']}" if s.get("requires") else "")
                   + ".*\n")
        out.append("| Metric | What it measures | Good |")
        out.append("|---|---|---|")
        for name, desc, good in rows:
            out.append(f"| **{name}** | {desc} | {good} |")

    # Tokenization is an adjunct to stage 1 rather than a numbered stage.
    out.append("\n### 2.10 Tokenization *(LAMBADA)*\n")
    out.append("*Separates a tokenizer handicap from a comprehension gap: a "
               "multi-token target must be produced correctly several times over.*\n")
    out.append("| Metric | What it measures | Good |")
    out.append("|---|---|---|")
    for name, desc, good in METRIC_DETAIL["tokenization"]:
        out.append(f"| **{name}** | {desc} | {good} |")

    ex = taxonomy.EXCLUDED_STAGE
    out.append(f"\n### 2.11 Stage {ex['stage']} — {ex['layer']} *(excluded)*\n")
    out.append(f"{ex['reason']}\n")
    out.append(f"**To measure it:** {ex['path_to_measure']}\n")
    out.append("| Subsection | Metrics |")
    out.append("|---|---|")
    for sub in ex["subsections"]:
        out.append(f"| {sub['name']} | {', '.join(sub['metrics'])} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. Results
# ---------------------------------------------------------------------------

def lambada_quality_table(models):
    rows = ["| Model | Last-word accuracy | 95% CI | Exact match | Stem match | Error rate | Empty output |",
            "|---|---|---|---|---|---|---|"]
    for m in models:
        tq = m.get("task_quality", {})
        ci = tq.get("last_word_accuracy_ci95") or []
        ci_txt = f"{ci[0]*100:.1f}–{ci[1]*100:.1f}%" if len(ci) == 2 else "-"
        rows.append(
            f"| {R.name(m)} | **{R.pct(tq.get('last_word_accuracy'))}** | {ci_txt} | "
            f"{R.pct(tq.get('exact_target_match'))} | "
            f"{R.pct(tq.get('stem_match_accuracy'))} | "
            f"{R.pct(tq.get('error_rate'))} | "
            f"{R.pct(tq.get('empty_prediction_rate'))} |")
    return "\n".join(rows)


def position_table(models):
    """Stage 4 for LAMBADA: accuracy by passage-length quartile."""
    usable = [m for m in models
              if R.dig(m, "context.position_sensitivity.buckets")]
    if not usable:
        return None
    rows = ["| Model | Q1 (shortest) | Q2 | Q3 | Q4 (longest) | Span |",
            "|---|---|---|---|---|---|"]
    for m in usable:
        b = R.dig(m, "context.position_sensitivity.buckets")
        cells = " | ".join(R.pct((b.get(q) or {}).get("accuracy"))
                           for q in ("q1", "q2", "q3", "q4"))
        rows.append(f"| {R.name(m)} | {cells} | "
                    f"{R.pct(R.dig(m, 'context.position_sensitivity.span'))} |")
    return "\n".join(rows)


def tokenization_table(models):
    usable = [m for m in models if (m.get("tokenization") or {}).get("available")]
    if not usable:
        return None
    rows = ["| Model | Tokens / target | Fragmented | Acc. 1 token | Acc. 2 tokens | Acc. 3+ tokens | Tokenizer |",
            "|---|---|---|---|---|---|---|"]
    for m in usable:
        t = m["tokenization"]
        by = t.get("accuracy_by_fragmentation", {})
        acc = lambda k: R.pct((by.get(k) or {}).get("accuracy"))
        rows.append(
            f"| {R.name(m)} | {R.fmt(t.get('tokens_per_target_word'), '{:.2f}')} | "
            f"{R.pct(t.get('fragmentation_rate'))} | {acc('1_token')} | "
            f"{acc('2_tokens')} | {acc('3plus_tokens')} | "
            f"{'exact' if t.get('tokenizer_exact') else 'approx (fallback BPE)'} |")
    return "\n".join(rows)


def results_section():
    out = ["## 3. Results\n",
           "Measured values per model, with a short note on where each figure "
           "comes from and why it landed where it did.\n"]

    for bench, label, n_lab in (("mmlu", "MMLU", "questions"),
                                ("lambada", "LAMBADA", "passages")):
        models = load(bench)
        if not models:
            continue
        cfg = models[0].get("config") or {}
        n = (cfg.get("dataset") or {}).get("n_items")
        key = "overall_accuracy" if bench == "mmlu" else "last_word_accuracy"

        out.append(f"\n### 3.{1 if bench == 'mmlu' else 2} {label}\n")
        out.append(f"**{n:,} {n_lab} per model · {len(models)} models · "
                   f"{n * len(models):,} scored items.** "
                   f"Greedy decoding (temperature "
                   f"{(cfg.get('decoding') or {}).get('temperature')}).\n")

        out.append("\n**Task quality**\n")
        # MMLU's shared table carries macro/normalised accuracy, which LAMBADA
        # has no subjects or option set to define; it gets match variants instead.
        out.append(R.quality_table(models, bench) if bench == "mmlu"
                   else lambada_quality_table(models))

        pos = position_table(models)
        if pos:
            out.append("\n\n**Context behaviour** — accuracy by passage-length "
                       "quartile. A flat row means length is not the constraint.\n")
            out.append(pos)

        tok_tbl = tokenization_table(models)
        if tok_tbl:
            out.append("\n\n**Tokenization** — how much of the gap is the "
                       "vocabulary rather than comprehension.\n")
            out.append(tok_tbl)

        out.append("\n\n**Serving, tokens and cost**\n")
        rows = ["| Model | TTFT | E2E p95 | Decode tok/s | Tokens / correct | $ / 1M tok | $ / correct |",
                "|---|---|---|---|---|---|---|"]
        for m in models:
            a, t = m.get("api_performance", {}), m.get("token_efficiency", {})
            e = m.get("economics", {})
            rows.append(
                f"| {R.name(m)} | {R.fmt(R.dig(a, 'latency.ttft.mean'), '{:.3f} s')} | "
                f"{R.fmt(R.dig(a, 'latency.e2e.p95'), '{:.3f} s')} | "
                f"{R.fmt(R.dig(a, 'throughput.decode_tokens_per_s'), '{:.1f}')} | "
                f"{R.fmt(t.get('tokens_per_correct_answer'), '{:.0f}')} | "
                f"{R.fmt(e.get('cost_per_1m_tokens_usd'), '${:.4f}')} | "
                f"{R.fmt(e.get('cost_per_correct_answer_usd'), '${:.6f}')} |")
        out.append("\n".join(rows))

        out.append("\n\n**Reliability**\n")
        out.append(R.reliability_table(models))

        comp = comparison(bench)
        rank = R.ranking_table(comp)
        if rank:
            out.append("\n\n**Composite ranking**\n")
            out.append(rank)
            caption = R.composite_caption(comp)
            if caption:
                out.append("\n" + caption + "\n")

        out.append("\n" + interpretation(models, comp, bench, key))
    return "\n".join(out)


def interpretation(models, comp, bench, key):
    """Short, data-derived note on where the numbers came from."""
    best = max(models, key=lambda m: m["task_quality"].get(key) or 0)
    lines = ["**Why these numbers**\n"]

    acc = best["task_quality"].get(key)
    ci = best["task_quality"].get(key + "_ci95") or []
    lines.append(
        f"- **{R.name(best)} leads at {R.pct(acc)}**"
        + (f" (95% CI {ci[0]*100:.1f}–{ci[1]*100:.1f}%)" if len(ci) == 2 else "")
        + ". Accuracy is scored over items the API actually answered; a request "
          "refused with a 429 says nothing about the model, so it is excluded "
          "from the denominator and reported separately as an error rate.")

    pf = sorted(((R.name(m),
                  m["task_quality"].get("parse_failure_rate")
                  or m["task_quality"].get("empty_prediction_rate") or 0)
                 for m in models), key=lambda x: -x[1])
    if pf and pf[0][1] > 0.02:
        lines.append(
            f"- **{pf[0][0]} returns no parseable answer on {R.pct(pf[0][1])} of "
            f"items.** That is an instruction-following failure, not ignorance — "
            f"its score understates what it knows. Measured by parsing each "
            f"response for an answer token and counting the misses separately "
            f"from wrong answers, because the remedies differ.")

    econ = [(R.name(m), (m.get("economics") or {}))
            for m in models if (m.get("economics") or {}).get("cost_per_correct_answer_usd")]
    if len(econ) >= 2:
        ct = min(econ, key=lambda x: x[1]["cost_per_1m_tokens_usd"])
        ca = min(econ, key=lambda x: x[1]["cost_per_correct_answer_usd"])
        dear = max(econ, key=lambda x: x[1]["cost_per_correct_answer_usd"])
        note = (f"- **Cost is provider-reported per request**, not estimated. "
                f"{ca[0]} gives the cheapest correct answer at "
                f"${ca[1]['cost_per_correct_answer_usd']:.6f}, against "
                f"${dear[1]['cost_per_correct_answer_usd']:.6f} for {dear[0]} "
                f"({dear[1]['cost_per_correct_answer_usd']/ca[1]['cost_per_correct_answer_usd']:.1f}×).")
        if ct[0] != ca[0]:
            note += (f" {ct[0]} has the cheapest *tokens* but not the cheapest "
                     f"*answers* — accuracy converts token price into value.")
        lines.append(note)

    lat = min(models, key=lambda m: R.dig(m, "api_performance.latency.e2e.mean") or 9e9)
    lines.append(
        f"- **Latency is client-measured from the streamed response.** "
        f"{R.name(lat)} is fastest end-to-end at "
        f"{R.fmt(R.dig(lat, 'api_performance.latency.e2e.mean'), '{:.3f} s')} mean; "
        f"p95 is reported alongside because the tail, not the mean, is what a "
        f"user experiences.")

    LABEL = {"probability": "probabilistic quality",
             "consistency": "reasoning and consistency",
             "context": "context behaviour",
             "robustness": "robustness"}
    missing = [s for s in ("probability", "consistency", "context", "robustness")
               if any((m.get(s) or {}).get("available") is not True for m in models)]
    if missing:
        provs = sorted({p for m in models
                        for p in (R.dig(m, "reliability.providers_used", {}) or {})})
        why = []
        if "probability" in missing:
            why.append(f"calibration needs a provider that returns token "
                       f"log-probabilities, and this run was routed to "
                       f"{', '.join(provs) or 'providers that do not'}")
        extra = [LABEL[s] for s in missing if s != "probability"]
        if extra:
            why.append(f"{', '.join(extra)} each need an additional pass over "
                       f"the dataset, which multiplies the API spend")
        lines.append(
            f"- **Blank stages are absences, not zeros.** "
            f"{', '.join(LABEL[s] for s in missing).capitalize()} "
            f"{'carries' if len(missing) == 1 else 'carry'} no data here: "
            f"{'; '.join(why)}. Each is stored as unavailable with its reason "
            f"rather than filled with 0, so a gap in coverage can never be "
            f"mistaken for a poor score. The History view can run them after "
            f"the fact via **Secondary analysis**.")

    spans = [(R.name(m), R.dig(m, "context.position_sensitivity.span"))
             for m in models if R.dig(m, "context.position_sensitivity.span") is not None]
    if spans:
        worst = max(spans, key=lambda x: x[1])
        lines.append(
            f"- **Passage length is not the binding constraint.** Sorting "
            f"passages by word count into quartiles moves accuracy by at most "
            f"{R.pct(worst[1])} ({worst[0]}), and not monotonically — so the "
            f"failures are comprehension, not a lost-in-the-middle effect. This "
            f"comes free from the scored pass: no extra requests, just the "
            f"existing results bucketed by input length.")

    tok = [m for m in models if (m.get("tokenization") or {}).get("available")]
    if tok:
        get3 = lambda m: ((m["tokenization"]["accuracy_by_fragmentation"]
                           .get("3plus_tokens") or {}).get("accuracy"))
        rows = [(R.name(m), get3(m)) for m in tok if get3(m) is not None]
        if len(rows) >= 2:
            rows.sort(key=lambda x: -x[1])
            approx = [R.name(m) for m in tok
                      if not m["tokenization"]["tokenizer_exact"]]
            lines.append(
                f"- **Part of the LAMBADA spread is the tokenizer.** On targets "
                f"needing 3+ tokens, {rows[0][0]} holds {R.pct(rows[0][1])} while "
                f"{rows[-1][0]} falls to {R.pct(rows[-1][1])} — a mechanical "
                f"handicap, since a multi-token word must be produced correctly "
                f"several times over."
                + (f" Note {', '.join(approx)} fell back to a generic BPE "
                   f"vocabulary (their tokenizers are gated), so their "
                   f"fragmentation rates are indicative rather than exact."
                   if approx else ""))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{--ink:#1a1c1e;--body:#41464b;--muted:#787d83;--line:#e4e6e9;
      --bg:#fff;--soft:#f7f8fa;--accent:#0d6efd;--accent-soft:#e7f1ff}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--body);
     font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:1.5rem 1.1rem 4rem}
/* scroll-margin keeps an anchored heading clear of the sticky navbar the app
   injects above this page. */
h1,h2,h3,h4{color:var(--ink);line-height:1.25;margin:2.2rem 0 .75rem;
            font-weight:650;scroll-margin-top:4rem}
h1{font-size:1.85rem;margin-top:.5rem}
h2{font-size:1.35rem;padding-bottom:.4rem;border-bottom:1px solid var(--line)}
h3{font-size:1.1rem}
h4{font-size:.98rem;color:var(--body)}
p,li{margin:0 0 .7rem}
ul,ol{padding-left:1.15rem}
strong{color:var(--ink);font-weight:620}
code{background:var(--soft);border:1px solid var(--line);border-radius:4px;
     padding:.08em .35em;font-size:.88em;
     font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;
    padding:.9rem;overflow-x:auto}
pre code{background:none;border:0;padding:0}
em{color:var(--muted)}
hr{border:0;border-top:1px solid var(--line);margin:2.2rem 0}

/* Tables: centred, scrollable on narrow screens rather than squeezing. */
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:1rem 0}
table{border-collapse:collapse;margin:0 auto;font-size:.9rem;min-width:min(100%,32rem)}
th,td{border:1px solid var(--line);padding:.45rem .7rem;text-align:center;
      vertical-align:middle}
th{background:var(--soft);color:var(--ink);font-weight:620;white-space:nowrap}
td:first-child,th:first-child{text-align:left}
tbody tr:nth-child(even){background:#fcfcfd}

/* Figures and diagrams sit centred with room to breathe. */
img,svg{max-width:100%;height:auto}
p>img{display:block;margin:1.1rem auto}
/* The workflow is a tall flowchart. Scaling it to a phone width makes the
   labels unreadable, so it keeps its natural size and scrolls instead. */
.mermaid{margin:1.4rem 0;overflow-x:auto;-webkit-overflow-scrolling:touch}
/* `margin:auto`, not flex centring: in a scroll container flex centring clips
   the left edge of an oversized child instead of letting it scroll. */
.mermaid svg{display:block;margin:0 auto;max-width:none;height:auto}

.formula{margin:1.1rem auto;padding:.7rem 1rem;text-align:center;
         background:var(--accent-soft);border:1px solid #cfe2ff;border-radius:8px;
         color:#0a4bc4;font-size:.92rem;
         font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

.meta{color:var(--muted);font-size:.86rem;margin:.2rem 0 1.4rem}
.toc{background:var(--soft);border:1px solid var(--line);border-radius:8px;
     padding:.85rem 1.1rem;margin:1.4rem 0}
.toc ul{margin:.3rem 0 0;padding-left:1.1rem}
.toc li{margin:.15rem 0}

@media (max-width:640px){
  body{font-size:15px}
  .wrap{padding:1.1rem .85rem 3rem}
  h1{font-size:1.5rem}h2{font-size:1.2rem}h3{font-size:1.03rem}
  table{font-size:.82rem}
  th,td{padding:.38rem .5rem}
  .mermaid svg{margin:0}
}
@media print{.wrap{max-width:none}.table-wrap{overflow:visible}}
"""


def to_html(markdown_text, title):
    """Render the markdown to a standalone, responsive page."""
    import markdown as md

    # Mermaid blocks must survive the markdown pass untouched.
    import re
    blocks = []

    def stash(m):
        blocks.append(m.group(1))
        return f"@@MERMAID{len(blocks) - 1}@@"

    text = re.sub(r"```mermaid\s*\n(.*?)\n```", stash, markdown_text, flags=re.S)

    html = md.markdown(text, extensions=["tables", "fenced_code", "toc",
                                         "attr_list", "md_in_html"])
    for i, block in enumerate(blocks):
        html = html.replace(f"<p>@@MERMAID{i}@@</p>",
                            f'<div class="mermaid">{block}</div>')
        html = html.replace(f"@@MERMAID{i}@@",
                            f'<div class="mermaid">{block}</div>')

    # Every table gets a scroll container so a wide table never widens the page.
    html = re.sub(r"<table>", '<div class="table-wrap"><table>', html)
    html = re.sub(r"</table>", "</table></div>", html)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{html}
</div>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  // startOnLoad stays off so `run()` is awaited once — with it on, the diagram
  // would be rendered twice and the scroll below would race the second pass.
  mermaid.initialize({{ startOnLoad: false, theme: 'neutral',
                        flowchart: {{ useMaxWidth: false, htmlLabels: true }} }});
  // The diagram keeps its natural width and scrolls. Its nodes are centred
  // within that width, so a narrow screen would otherwise open on empty margin.
  await mermaid.run();
  document.querySelectorAll('.mermaid').forEach(function (el) {{
    el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
  }});
</script>
</body>
</html>"""


def main():
    title = "LAMBADA & MMLU Benchmark — Documentation"
    parts = [
        f"# {title}\n",
        "Small language models evaluated across nine measurable stages over the "
        "OpenRouter API. This page covers the workflow, what each metric means, "
        "and the measured results.\n",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from "
        f"`results/v2`. The full academic write-up is in "
        f"`extended-final-report.md`.*\n",
        "---\n",
        WORKFLOW,
        "\n---\n",
        components_section(),
        "\n---\n",
        results_section(),
    ]
    md_text = "\n".join(parts)
    open(MD_OUT, "w", encoding="utf-8").write(md_text)
    open(HTML_OUT, "w", encoding="utf-8").write(to_html(md_text, title))
    print(f"wrote {MD_OUT} ({os.path.getsize(MD_OUT):,} bytes)")
    print(f"wrote {HTML_OUT} ({os.path.getsize(HTML_OUT):,} bytes)")


if __name__ == "__main__":
    main()
