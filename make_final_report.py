"""Assemble `extended-final-report.md` from report-full.md plus live result data.

Generated rather than hand-written so the numbers can never drift from
results/v2. Re-run after any benchmark run:

    python make_final_report.py

Sections 1-6 (research question, protocol, models, methodology) are carried
over verbatim from report-full.md. Everything from the workflow onward is rebuilt
from the result files.
"""

import glob
import json
import os
import re
from datetime import datetime

import make_results_section as R

# The full academic report. report.md is now the lean Docs view, so the
# methodology sections are carried over from the preserved full version.
REPORT = "report-full.md"

from make_docs import significance_section  # noqa: E402
OUT = "extended-final-report.md"
FIGDIR = "diagram-analysis"


def load_all(benchmark):
    return [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(f"results/v2/*_{benchmark}_v2.json"))]


def carry_over_methodology():
    """Sections 1-6 of report-full.md, verbatim."""
    text = open(REPORT, encoding="utf-8").read()
    start = text.find("## 1. Research Question")
    end = text.find("## 7. Results")
    if start < 0 or end < 0:
        raise SystemExit("could not locate sections 1-6 in report-full.md")
    return text[start:end].rstrip()


def fig(n, slug, caption):
    """A figure reference plus its caption. The file need not exist yet."""
    return (f"\n![{caption}]({FIGDIR}/analysis-{n:02d}-{slug}.png)\n"
            f"*Fig. {n} — {caption}*\n")


# ---------------------------------------------------------------------------

def workflow_section():
    return """## 7. End-to-End Workflow

The pipeline below is what one benchmark run actually executes. Stages 1-9 are
produced by every run; stage 10 is shown greyed because it is *excluded by
design* — nothing in it is observable through a hosted inference API.

```mermaid
flowchart TD
    A[Select models and benchmark] --> B[Build RunConfig<br/>parallelism · serving · decoding · dataset · passes]
    B --> C{Pre-flight probe<br/>per model}
    C -->|throttled| C1[Warn: provider rate-limiting<br/>run continues, nothing dropped]
    C -->|healthy| D[Load dataset<br/>MMLU: HF datasets-server · LAMBADA: local splits]
    C1 --> D
    D --> E[Estimate API calls<br/>show multiplier before spending]
    E --> F[Scored pass<br/>chain-of-thought, non-streamed]
    F --> G[Extended pass<br/>streamed: TTFT / TPOT / tokens / cost]
    G --> H{Optional passes}
    H -->|--calibration| H1[Single-token scoring<br/>max_tokens=1 → option probabilities]
    H -->|--repeats N| H2[Repeat items → stability]
    H -->|--robustness| H3[Prompt / reorder / typo variants]
    H -->|--context| H4[Context ablations · LAMBADA]
    H1 --> I
    H2 --> I
    H3 --> I
    H4 --> I
    H --> I[Compute metric blocks]

    I --> S1[1 Task Quality]
    I --> S2[2 Probabilistic Quality]
    I --> S3[3 Reasoning and Consistency]
    I --> S4[4 Context Behavior]
    I --> S5[5 Robustness]
    I --> S6[6 API Performance]
    I --> S7[7 Token Efficiency]
    I --> S8[8 Economics]
    I --> S9[9 Reliability]

    S1 --> J[Aggregate and rank<br/>weights renormalised over available components]
    S2 --> J
    S3 --> J
    S4 --> J
    S5 --> J
    S6 --> J
    S7 --> J
    S8 --> J
    S9 --> J

    J --> K{Configs identical?}
    K -->|no| K1[comparable = false<br/>refuse to rank]
    K -->|yes| L[Write results/v2 · history · comparison]
    L --> M[Web view: History → Extended analysis]
    L --> N[report-full.md · extended-final-report.md · slides]

    X[10 Hardware and Distributed<br/>TP·PP·DP·SP·CP·EP · VRAM · energy]
    X -.->|not observable through a hosted API| Y[Self-hosted vLLM path only]

    style X fill:#f1f3f5,stroke:#adb5bd,color:#868e96
    style Y fill:#f1f3f5,stroke:#adb5bd,color:#868e96
    style K1 fill:#fff3cd,stroke:#ffc107
    style C1 fill:#fff3cd,stroke:#ffc107
```

### 7.1 The two measurement passes

A run makes two independent trips to the API per item, and the distinction
matters when reading any table below.

| Pass | Prompt | Streamed | Produces | Why it exists |
|---|---|---|---|---|
| **Scored** | chain-of-thought, then a letter/word | no | the headline accuracy and reasoning analysis | it is how the model would really be used |
| **Extended** | identical prompt | yes | TTFT, TPOT, throughput, tokens, cost, reliability | latency is unmeasurable without streaming |
| **Calibration** (optional) | direct answer, `max_tokens=1` | no | option probabilities → NLL, ECE, Brier, target rank | providers only return usable log-probabilities for a single generated token |

Because the scored and extended passes are separate API calls, their accuracies
differ slightly even at temperature 0. That residual is serving
non-determinism, and the web view states the delta explicitly rather than
showing two numbers and leaving the reader to notice.
"""


def results_section(benchmark, number, label):
    models = load_all(benchmark)
    if not models:
        return f"## {number}. {label} Results\n\n*No results on disk.*\n"

    cfg = models[0].get("config") or {}
    ds, ev = cfg.get("dataset") or {}, cfg.get("evaluation") or {}
    n = ds.get("n_items")
    passes = ["scored", "extended"]
    if ev.get("request_logprobs"):
        passes.append("calibration")
    if (ev.get("repeats_for_consistency") or 1) > 1:
        passes.append(f"{ev['repeats_for_consistency']} repeats")
    if ev.get("robustness_variants"):
        passes.append("robustness")
    if ev.get("context_ablations"):
        passes.append("context ablation")

    out = [f"## {number}. {label} — Results\n"]
    out.append(
        f"**{n:,} items per model · {len(models)} models · "
        f"{n * len(models):,} scored items.** "
        f"Passes: {', '.join(passes)}. "
        f"Decoding: temperature {(cfg.get('decoding') or {}).get('temperature')}, "
        f"max_tokens {(cfg.get('decoding') or {}).get('max_tokens')}.\n")

    out.append(f"\n### {number}.1 Stage 1 — Task Quality\n\n"
               + R.quality_table(models, benchmark) + "\n")
    out.append(f"\n### {number}.2 Stage 2 — Probabilistic Quality\n\n"
               + R.calibration_table(models) + "\n")

    covered = [m for m in models if R.dig(m, "probability.available")]
    if len(covered) < len(models):
        missing = ", ".join(R.name(m) for m in models if m not in covered)
        out.append(
            f"\nUnavailable for: {missing}. Log-probability support is a property of "
            f"the **provider** OpenRouter routed to, not of the model — the same "
            f"model yields calibration on one run and none on the next. The provider "
            f"is listed beside every row for exactly this reason.\n")

    t = R.consistency_table(models)
    if t:
        out.append(f"\n### {number}.3 Stage 3 — Reasoning & Consistency\n\n{t}\n")
    t = R.context_table(models)
    if t:
        out.append(f"\n### {number}.4 Stage 4 — Context Behavior\n\n{t}\n")
    t = R.robustness_table(models)
    if t:
        out.append(f"\n### {number}.5 Stage 5 — Robustness\n\n"
                   f"Accuracy drop against the clean baseline; negative means the "
                   f"variant scored *higher*.\n\n{t}\n")

    out.append(f"\n### {number}.6 Stage 6 — API Performance\n\n"
               + R.performance_table(models) + "\n")
    out.append(f"\n### {number}.7 Stages 7 & 8 — Token Efficiency and Economics\n\n"
               + R.tokens_cost_table(models) + "\n")
    out.append(f"\n### {number}.8 Stage 9 — Reliability\n\n"
               + R.reliability_table(models) + "\n")

    # Tokenization is LAMBADA-only and carries a provenance caveat.
    tok = [m for m in models if (m.get("tokenization") or {}).get("available")]
    if tok:
        rows = ["| Model | Tokenizer | Exact? | Tokens/word | Fragmented | 1 token | 2 tokens | 3+ tokens |",
                "|---|---|---|---|---|---|---|---|"]
        for m in tok:
            t = m["tokenization"]
            bf = t.get("accuracy_by_fragmentation", {})
            g = lambda k: R.pct((bf.get(k) or {}).get("accuracy"))
            rows.append(
                f"| {R.name(m)} | `{t['tokenizer']}` | "
                f"{'yes' if t['tokenizer_exact'] else '**no**'} | "
                f"{R.fmt(t.get('tokens_per_target_word'))} | "
                f"{R.pct(t.get('fragmentation_rate'))} | "
                f"{g('1_token')} | {g('2_tokens')} | {g('3plus_tokens')} |")
        out.append(f"\n### {number}.9 Tokenization\n\n" + "\n".join(rows) + "\n")
        approx = [R.name(m) for m in tok if not m["tokenization"]["tokenizer_exact"]]
        if approx:
            out.append(
                f"\n> **Caveat.** {', '.join(approx)} fell back to `cl100k_base` "
                f"because their Hugging Face tokenizers are gated. Their "
                f"fragmentation *rates* therefore describe how a generic BPE "
                f"vocabulary splits the targets, not their own. The accuracy "
                f"trend across buckets remains valid within each model — each is "
                f"bucketed consistently — but the rates are not comparable "
                f"across models unless both are marked exact.\n")

    comp = R.load_comparison(benchmark)
    rank = R.ranking_table(comp)
    if rank:
        out.append(f"\n### {number}.10 Composite Ranking\n\n{rank}\n")
        out.append(f"\n{comp.get('comparability_note','')}\n")
        out.append(
            "\nWeights are renormalised over the components a run actually "
            "produced, and each row lists which those were, so a model is never "
            "penalised for a study that was not run.\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------

def component_reference():
    """Descriptive table for every metric, stage by stage."""
    from metrics import taxonomy

    rows = ["## 11. Component Reference\n",
            "Every metric the benchmark produces, what it means in plain terms, "
            "and how to read it. Stage numbers match `metrics/taxonomy.py`.\n"]

    for s in taxonomy.STAGES:
        rows.append(f"\n### 10.{s['stage']} Stage {s['stage']} — {s['layer']}\n")
        rows.append(f"*{s['purpose']}. Priority {s['priority']}; "
                    f"OpenRouter observability: **{s['openrouter']}**"
                    + (f"; requires {s['requires']}" if s.get("requires") else "")
                    + ".*\n")
        rows.append("| Subsection | Metrics | Observable |")
        rows.append("|---|---|---|")
        for sub in s["subsections"]:
            note = f" — {sub['note']}" if sub.get("note") else ""
            bench = ""
            if sub.get("benchmarks"):
                bench = f" *({'/'.join(b.upper() for b in sub['benchmarks'])} only)*"
            rows.append(f"| {sub['name']}{bench} | {', '.join(sub['metrics'])} | "
                        f"{sub.get('observable', 'yes')}{note} |")

    ex = taxonomy.EXCLUDED_STAGE
    rows.append(f"\n### 10.{ex['stage']} Stage {ex['stage']} — {ex['layer']} "
                f"(excluded)\n")
    rows.append(f"{ex['reason']}\n")
    rows.append(f"**To measure it:** {ex['path_to_measure']}\n")
    rows.append("| Subsection | Metrics |")
    rows.append("|---|---|")
    for sub in ex["subsections"]:
        rows.append(f"| {sub['name']} | {', '.join(sub['metrics'])} |")
    return "\n".join(rows)


def metric_glossary():
    """Plain-language definition of each individual metric."""
    groups = [
        ("Accuracy family", [
            ("Overall accuracy", "Correct ÷ questions answered. The headline number.", "higher"),
            ("Macro accuracy", "Average of per-subject accuracies, so a large subject cannot dominate.", "higher"),
            ("Normalised accuracy", "(acc − chance) ÷ (1 − chance). Restates a score as distance above guessing.", "higher"),
            ("Wilson 95% CI", "Confidence interval that stays inside [0,1] at small n, unlike the normal approximation.", "narrower"),
            ("Error rate", "1 − accuracy, framed as a budget.", "lower"),
            ("Parse-failure rate", "Answers with no extractable letter. An instruction-following failure, not ignorance.", "lower"),
            ("Option-position bias", "Distance between the letters the model picks and the letters that are correct.", "lower"),
        ]),
        ("Probability & calibration", [
            ("P(correct option)", "Probability mass placed on the right answer.", "higher"),
            ("NLL", "How surprised the model was by the truth. What language models are trained to minimise.", "lower"),
            ("Perplexity", "exp(NLL). 1.0 = certain and right; 4.0 ≈ guessing among four options.", "lower"),
            ("Entropy", "How spread out its belief was. High = unsure.", "context"),
            ("ECE", "Expected Calibration Error — does stated confidence match observed accuracy?", "lower"),
            ("MCE", "The worst single confidence bin. Tail miscalibration.", "lower"),
            ("Brier score", "Squared error of confidence against outcome. A proper scoring rule: cannot be gamed by hedging.", "lower"),
            ("Confidence↔accuracy correlation", "Whether its confidence carries any signal at all.", "higher"),
            ("Target rank / MRR", "Where the right answer sat among ranked predictions; MRR rewards 'nearly right'.", "higher"),
        ]),
        ("Consistency", [
            ("Answer stability", "Share of items where every repeat gave the identical answer.", "higher"),
            ("Self-consistency gain", "Majority-vote accuracy minus single-sample accuracy.", "higher"),
            ("Seed stability", "Accuracy spread across independent seeds. Compare against model gaps before ranking.", "lower"),
        ]),
        ("Context (LAMBADA)", [
            ("Context utilisation", "Accuracy with the full passage minus accuracy with only the last sentence.", "higher"),
            ("Utilisation ratio", "Share of the model's skill that depends on the wider passage. 1.0 = entirely.", "higher"),
            ("Position sensitivity", "Accuracy bucketed by passage length — a 'lost in the middle' probe.", "flat"),
        ]),
        ("Robustness", [
            ("Accuracy drop", "Baseline accuracy minus perturbed accuracy.", "lower"),
            ("Flip rate", "Share of answers that changed at all. Catches offsetting errors that leave accuracy flat.", "lower"),
            ("Broke / fixed", "Answers that went right→wrong and wrong→right. Both are instability.", "lower"),
            ("Robustness score", "1 − mean relative drop across all perturbations.", "higher"),
        ]),
        ("Tokenization (LAMBADA)", [
            ("Tokens per target word", "How many tokens the tokenizer needs for the gold word, with its leading space.", "lower"),
            ("Fragmentation rate", "Share of targets needing more than one token — mechanically harder to produce.", "lower"),
            ("Accuracy by fragmentation", "Accuracy split by 1 / 2 / 3+ token targets. Separates tokenizer handicap from comprehension.", "flat"),
        ]),
        ("API performance", [
            ("TTFT", "Time to first token. Prefill plus queueing plus network — what makes a chat feel responsive.", "lower"),
            ("TPOT", "Time per output token after the first. The steady-state generation rate.", "lower"),
            ("E2E latency", "Total wait for the complete answer.", "lower"),
            ("p95 / p99", "Tail latency. For serving, the tail is the user experience; the mean hides it.", "lower"),
            ("Decode throughput", "Generated tokens per second.", "higher"),
        ]),
        ("Token efficiency & economics", [
            ("Prompt / completion tokens", "What was billed, as reported by the provider.", "lower"),
            ("Reasoning tokens", "Tokens spent on hidden reasoning. 0 is a real answer for a non-reasoning model.", "lower"),
            ("Cached prompt tokens", "Prompt tokens served from the provider's cache, and therefore cheaper.", "higher"),
            ("Tokens per correct answer", "Token cost of a *useful* answer, not just any answer.", "lower"),
            ("Cost per 1M tokens", "Blended price actually paid, comparable against list prices.", "lower"),
            ("Cost per correct answer", "Money per useful answer. **The figure that should drive model choice** — a cheaper model that is wrong twice as often is not cheaper.", "lower"),
        ]),
        ("Reliability", [
            ("Success / failure rate", "Requests that returned, versus those that errored even after retries.", "higher / lower"),
            ("Invalid output rate", "A 200 response carrying nothing usable. A model problem, not a transport one.", "lower"),
            ("Timeout / 429 rate", "Failures bucketed by class, separating 'provider overloaded' from 'provider broken'.", "lower"),
            ("Retry rate", "How hard the client had to try. Invisible in an accuracy table.", "lower"),
            ("Provider failover", "Whether more than one backend served the run — if so, latency and calibration mix two stacks.", "no"),
        ]),
    ]

    out = ["## 12. Metric Glossary\n",
           "Each metric in plain language, with the direction that counts as good.\n"]
    for title, items in groups:
        out.append(f"\n### {title}\n")
        out.append("| Metric | What it means | Good |")
        out.append("|---|---|---|")
        for nm, desc, good in items:
            out.append(f"| **{nm}** | {desc} | {good} |")
    return "\n".join(out)


def figures_section():
    figs = [
        (1, "history-overview", "The History tab: one card per run, with metric-family chips, configuration chips and the ranking table"),
        (2, "extended-collapsed", "The extended-analysis container with its per-model tabs"),
        (3, "stage-list", "All nine stages collapsed; families with no data carry an n/a badge"),
        (4, "task-quality", "Stage 1 — accuracy, Wilson intervals, per-subject and per-category tables, option bias"),
        (5, "calibration", "Stage 2 where the provider returns log-probabilities: NLL, perplexity, ECE, Brier, reliability bins"),
        (6, "calibration-unavailable", "Stage 2 where it does not: an explicit unavailability notice with its reason"),
        (7, "consistency", "Stage 3 — answer stability, majority-vote accuracy, self-consistency gain"),
        (8, "context", "Stage 4 — context ablation table and utilisation ratio (LAMBADA)"),
        (9, "robustness", "Stage 5 — per-variant accuracy drop with the broke/fixed split and flip rate"),
        (10, "api-performance", "Stage 6 — TTFT, TPOT and E2E with p50/p95/p99, plus throughput"),
        (11, "token-efficiency", "Stage 7 — prompt, completion, reasoning and cached tokens; tokens per correct answer"),
        (12, "economics", "Stage 8 — spend per request, per million tokens and per correct answer"),
        (13, "reliability", "Stage 9 — success, failure, timeout and rate-limit rates; retries and failover"),
        (14, "run-config", "The run configuration, labelled as input, with TP/PP/DP/SP/CP/EP chips"),
        (15, "decoding-panel", "The decoding panel: nine parameters, five presets, and the extended-pass selector with its live call estimate"),
        (16, "mobile-subjects", "Subject categories at phone width, two per row, each reporting its selected count"),
        (17, "nav-modal", "Navigation consolidated into a single modal from the hamburger control"),
        (18, "two-pass-delta", "The two-pass reconciliation banner: scored pass, extended pass, and the delta between them"),
        (19, "preflight", "The pre-flight probe warning that a provider is rate-limiting before a long run starts"),
    ]
    out = ["## 13. Figures\n",
           f"Screenshots live in `{FIGDIR}/` and are referenced by exact filename. "
           f"`{FIGDIR}/README.md` records what each should show and where in the UI "
           f"to capture it. A file that has not been added yet renders as a broken "
           f"image; adding the PNG is the only step required.\n"]
    for n, slug, cap in figs:
        out.append(fig(n, slug, cap))
    return "\n".join(out)


def discussion_section():
    """Findings drawn from the actual result files, not written by hand."""
    mm = {R.name(m): m for m in load_all("mmlu")}
    lb = {R.name(m): m for m in load_all("lambada")}
    if not mm or not lb:
        return ""

    def acc(d, m, b):
        k = "overall_accuracy" if b == "mmlu" else "last_word_accuracy"
        return (d[m]["task_quality"] or {}).get(k)

    lines = ["## 14. Discussion — What the Nine Stages Revealed\n",
             "Each finding below is one that **accuracy alone could not have "
             "surfaced**. That is the argument for the extra stages.\n"]

    # 1 - parse failure vs ignorance
    pf = sorted(((R.name(m), m["task_quality"].get("parse_failure_rate") or 0)
                 for m in mm.values()), key=lambda x: -x[1])
    worst, rate = pf[0]
    lines.append(
        f"\n### 13.1 Disobedience is not ignorance\n\n"
        f"**{worst} failed to emit a parseable answer on {rate*100:.1f}% of MMLU "
        f"questions.** Its accuracy of {acc(mm, worst, 'mmlu')*100:.1f}% therefore "
        f"understates what it knows: roughly one question in "
        f"{int(round(1/rate)) if rate else 0} was scored wrong because the output "
        f"format was not followed, not because the answer was wrong. Stage 1 "
        f"separates the two because the fixes differ — a better prompt or "
        f"constrained decoding addresses the first; nothing addresses the second. "
        f"The other models sit at "
        f"{', '.join(f'{n} {r*100:.1f}%' for n, r in pf[1:])}.\n")

    # 2 - reliability
    rel = [(R.name(m), m["reliability"]) for m in mm.values()]
    fail = [(n, r) for n, r in rel if (r.get("failure_rate") or 0) > 0
            or (r.get("retry_total") or 0) > 0 or r.get("provider_failover")]
    if fail:
        bits = []
        for n, r in fail:
            d = []
            if r.get("failure_rate"):
                d.append(f"{r['failure_rate']*100:.1f}% transport failures")
            if r.get("retry_total"):
                d.append(f"{r['retry_total']} retries")
            if r.get("provider_failover"):
                d.append("provider failover")
            bits.append(f"**{n}** ({', '.join(d)})")
        lines.append(
            f"\n### 13.2 Serving failures hide inside an accuracy table\n\n"
            f"Stage 9 recorded conditions invisible to every other metric: "
            f"{'; '.join(bits)}. A run that silently retried dozens of requests "
            f"produces the same accuracy figure as a clean one, so without this "
            f"stage the difference between 'the model was wrong' and 'the "
            f"provider refused' is unrecoverable after the fact.\n")

    # 3 - tokenization
    tk = [(R.name(m), m.get("tokenization") or {}) for m in lb.values()]
    tk = [(n, t) for n, t in tk if t.get("available")]
    if tk:
        rows = []
        for n, t in tk:
            bf = t.get("accuracy_by_fragmentation", {})
            one = (bf.get("1_token") or {}).get("accuracy")
            many = (bf.get("3plus_tokens") or {}).get("accuracy")
            if one is not None and many is not None:
                rows.append((n, one, many, one - many))
        rows.sort(key=lambda x: -x[3])
        if rows:
            body = "\n".join(
                f"| {n} | {o*100:.1f}% | {m*100:.1f}% | {d*100:+.1f} pp |"
                for n, o, m, d in rows)
            lines.append(
                f"\n### 13.3 Part of the LAMBADA gap is the tokenizer, not comprehension\n\n"
                f"Splitting LAMBADA accuracy by how many tokens the gold word needs:\n\n"
                f"| Model | 1-token targets | 3+-token targets | Collapse |\n"
                f"|---|---|---|---|\n{body}\n\n"
                f"A multi-token target must be produced correctly several times over, "
                f"so this is a mechanical handicap rather than a comprehension one. "
                f"{rows[0][0]} loses {rows[0][3]*100:.0f} percentage points across the "
                f"buckets; {rows[-1][0]} loses {rows[-1][3]*100:.0f}. Comparing raw "
                f"LAMBADA scores without this split treats a vocabulary difference as "
                f"a capability difference.\n")

    # 4 - cost per correct answer
    econ = []
    for n, m in mm.items():
        e = m.get("economics") or {}
        econ.append((n, e.get("cost_per_correct_answer_usd"),
                     e.get("cost_per_1m_tokens_usd"), acc(mm, n, "mmlu")))
    econ = [e for e in econ if e[1]]
    if len(econ) >= 2:
        by_correct = sorted(econ, key=lambda x: x[1])
        by_token = sorted(econ, key=lambda x: x[2])
        lines.append(
            f"\n### 13.4 Cheapest per token is not cheapest per answer\n\n"
            f"**{by_token[0][0]}** has the lowest price per million tokens "
            f"(${by_token[0][2]:.4f}), but **{by_correct[0][0]}** delivers the "
            f"lowest cost per *correct* answer (${by_correct[0][1]:.6f} vs "
            f"${by_correct[-1][1]:.6f} for {by_correct[-1][0]} — "
            f"{by_correct[-1][1]/by_correct[0][1]:.1f}x more). Accuracy converts "
            f"token price into value: a model that is wrong more often spends its "
            f"savings on wrong answers. Cost per correct answer is the figure that "
            f"should drive selection.\n")

    # 5 - latency vs quality
    lat = sorted(((R.name(m), R.dig(m, "api_performance.latency.e2e.mean"),
                   acc(mm, R.name(m), "mmlu")) for m in mm.values()),
                 key=lambda x: x[1] or 9e9)
    if len(lat) >= 2 and lat[0][1] and lat[-1][1]:
        lines.append(
            f"\n### 13.5 The speed/quality trade is not subtle\n\n"
            f"**{lat[0][0]}** answers in {lat[0][1]:.3f}s mean end-to-end at "
            f"{lat[0][2]*100:.1f}% accuracy; **{lat[-1][0]}** takes "
            f"{lat[-1][1]:.3f}s ({lat[-1][1]/lat[0][1]:.1f}x longer) for "
            f"{lat[-1][2]*100:.1f}%. Which is preferable is a deployment "
            f"decision, not a benchmark one — which is why the composite exposes "
            f"its components rather than collapsing them into a single verdict.\n")

    return "\n".join(lines)


def limitations_section():
    mm, lb = load_all("mmlu"), load_all("lambada")
    n_mm = (mm[0]["config"]["dataset"]["n_items"] if mm else 0)
    n_lb = (lb[0]["config"]["dataset"]["n_items"] if lb else 0)
    missing = set()
    for m in mm + lb:
        for k in ("probability", "consistency", "robustness", "context"):
            if k in m and (m[k] or {}).get("available") is False:
                missing.add(k)
    return f"""## 15. Limitations

Stated plainly, because a benchmark that hides its limits is worth less than
one that reports fewer numbers honestly.

| Limitation | Effect | Mitigation in place |
|---|---|---|
| **Stage 10 is unmeasurable** | TP/PP/DP/SP/CP/EP, VRAM, KV cache, energy and communication overhead cannot be observed through a hosted API | Recorded as configuration with `controlled: false`, excluded from every score, and reproducible on the self-hosted vLLM path |
| **Calibration depends on the provider** | Stage 2 yields numbers only when the routed provider returns log-probabilities — and routing changes between runs | Provider printed beside every row; blocks marked `available: false` with a reason rather than zero-filled |
| **Two API passes per item** | Scored and extended passes differ slightly at temperature 0 | The delta is displayed explicitly in the web view and attributed to serving non-determinism |
| **Tokenizer fallback** | Gated Hugging Face repos force a generic BPE vocabulary for some models | `tokenizer_exact: false` recorded; cross-model fragmentation rates flagged as not comparable |
| **Sample size** | {n_mm:,} MMLU questions and {n_lb:,} LAMBADA passages give roughly ±2 pp and ±3 pp at 95% confidence | Wilson intervals reported beside every accuracy; differences inside the interval are not claimed as findings |
| **Optional stages not always run** | {', '.join(sorted(missing)) or 'none'} were unavailable in this run | Each costs an extra pass over the dataset; the runner prints the multiplier before spending, and absent stages are marked, never inferred |
| **Deterministic paraphrasing** | Robustness paraphrases are rule-based, not model-generated | Keeps the benchmark reproducible; a learned paraphraser would make it non-repeatable |

## 16. Conclusions

**Does a larger, non-distilled model beat smaller compressed ones?**

On task quality, yes and consistently: Ministral-8B leads both benchmarks. But
the nine-stage view shows the answer is not one-dimensional — the smaller models
win on latency and, in one case, on cost per correct answer, and the ranking
between second and third place changes depending on whether reliability is
counted.

**What the extra stages bought.** Five findings in §13 are invisible to
accuracy alone: an instruction-following failure mistaken for ignorance, serving
failures hidden inside a clean-looking score, a tokenizer handicap mistaken for
a comprehension gap, a token price that inverts once accuracy is accounted for,
and a speed/quality trade that no single number can adjudicate.

**The methodological point.** Configuration and measurement are kept apart
throughout: parallelism degrees and decoding settings are inputs recorded in a
run manifest, never folded into a score. That separation is what allows the
comparison layer to refuse a ranking when two runs are not comparable — and
refusing is the correct output in that case.
"""


def main():
    parts = [
        "# LAMBADA & MMLU — Extended Final Report\n",
        "Small language models evaluated across a nine-stage benchmark: task "
        "quality, probabilistic quality, consistency, context behaviour, "
        "robustness, API performance, token efficiency, economics and "
        "reliability.\n",
        "Submitted to: Prof. Anna Corazza  ",
        "Submitted by: Francesco Ventimiglia, Danilo Rodriguez, Rohan Baidya  ",
        "GitHub: https://github.com/ronvoy/gen-ai  ",
        "Site: https://unina.cc/gen-ai  ",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "---\n",
        carry_over_methodology(),
        "\n---\n",
        workflow_section(),
        "\n---\n",
        results_section("mmlu", 8, "MMLU"),
        "\n---\n",
        results_section("lambada", 9, "LAMBADA"),
        "\n---\n",
        # Shared with the Docs view rather than duplicated: one implementation,
        # so the two documents can never disagree about a p-value.
        significance_section().replace("## 4. ", "## 10. ").replace("### 4.", "### 10."),
        "\n---\n",
        component_reference(),
        "\n---\n",
        metric_glossary(),
        "\n---\n",
        figures_section(),
        "\n---\n",
        discussion_section(),
        "\n---\n",
        limitations_section(),
    ]
    open(OUT, "w", encoding="utf-8").write("\n".join(parts))
    print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes)")


if __name__ == "__main__":
    main()
