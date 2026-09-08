# Self-Study Guide — SLM Benchmarking (MMLU + LAMBADA)

A working companion to this repository. Every metric is explained twice: once
in plain language ("ELI5"), once precisely. Read the ELI5 column to build
intuition, the description column when you need to defend a number.

**The one idea to take away:** there are two kinds of variable in a benchmark.
Things *you set* (configuration) and things *you measure* (metrics). Mixing
them is the most common way benchmark reports go wrong. Section 8 is entirely
about that distinction.

---

## Table of Contents

1. [The mental model](#1-the-mental-model)
2. [MMLU — Task Quality](#2-mmlu--task-quality)
3. [MMLU — Probability & Calibration](#3-mmlu--probability--calibration)
4. [Consistency](#4-consistency)
5. [Robustness](#5-robustness)
6. [LAMBADA — Task Quality & Target Probability](#6-lambada--task-quality--target-probability)
7. [LAMBADA — Context & Tokenization](#7-lambada--context--tokenization)
8. [Systems metrics vs configuration variables](#8-systems-metrics-vs-configuration-variables)
9. [The parallelism degrees](#9-the-parallelism-degrees-tp-pp-dp-sp-cp-ep)
10. [Benchmark process flow](#10-benchmark-process-flow)
11. [What this setup can and cannot measure](#11-what-this-setup-can-and-cannot-measure)
12. [How to read a result file](#12-how-to-read-a-result-file)
13. [Statistical honesty](#13-statistical-honesty)
14. [Glossary](#14-glossary)

---

## 0. The ten stages at a glance

The benchmark is organised into ten stages. **Nine are measurable through
OpenRouter; the tenth is not, and is excluded rather than estimated.** The
authoritative definition is `metrics/taxonomy.py`.

| Stage | Layer | Band | OpenRouter | This guide |
|---|---|---|---|---|
| 1 | Task Quality | P0 | yes | §2, §6 |
| 2 | Probabilistic Quality | P1 | **conditional** — needs a provider that returns logprobs | §3 |
| 3 | Reasoning & Consistency | P1 | yes — needs repeats | §4 |
| 4 | Context Behavior | P1 | yes — needs ablation pass | §7 |
| 5 | Robustness | P1 | yes — needs perturbation pass | §5 |
| 6 | API Performance | P0 | yes | §8 |
| 7 | Token Efficiency | P0 | yes | §8 |
| 8 | Economics | P0 | yes | §8 |
| 9 | Reliability | P0 | yes | §8 |
| 10 | ~~Hardware & Distributed~~ | — | **no** | §9, §11 |

**P0** stages come out of every run for free. **P1** stages each cost an extra
pass over the dataset, which is why they are opt-in and why the runner prints
the API-call multiplier before spending anything.

---

## 1. The mental model

A benchmark is an experiment. It has:

| Role | Name | Examples | Where it lives |
|---|---|---|---|
| What you choose | **Independent variables** | model, temperature, TP/PP/DP, batch size, prompt template | `benchmark_config.py` |
| What you observe | **Dependent variables** | accuracy, ECE, TTFT, cost | `metrics/` |
| What you must hold still | **Controls** | dataset, sample seed, decoding params | `RunConfig.config_key()` |

If two runs differ in more than one independent variable, you cannot attribute
a change in the dependent variables to any single cause. `RunConfig.diff()`
exists to check exactly this before you compare anything.

**Two benchmarks, two different questions:**

| | MMLU | LAMBADA |
|---|---|---|
| Format | 4-option multiple choice | predict the final word |
| Tests | knowledge + reasoning across 57 subjects | long-range reading comprehension |
| Chance score | 25% | ~0% (open vocabulary) |
| Scored by | letter match | word match |
| Fails when | model lacks knowledge | model ignores distant context |

---

## 2. MMLU — Task Quality

| Metric | ELI5 | Description | Range / good |
|---|---|---|---|
| **Overall Accuracy** | Out of every question asked, how many did it get right? | Micro-average: `correct / total`, pooled over all questions regardless of subject. | 0–1, higher better. 0.25 = guessing |
| **Macro Accuracy** | Average the score *per subject*, then average those — so a big subject can't dominate. | Mean of per-subject accuracies. The number MMLU papers headline. | 0–1, higher better |
| **Subject Accuracy** | The report card, one line per school subject. | Accuracy within each of the 57 subjects, with a Wilson 95% CI. | per subject |
| **Category Accuracy** | Same, grouped into 4 big buckets: STEM, Humanities, Social Sciences, Other. | Accuracy aggregated by category. Shows *where* a model is strong. | per category |
| **Error Rate** | The share it got wrong. | `1 − accuracy`. Same information, framed as a budget. | 0–1, lower better |
| **Normalised Accuracy** | "How much better than a coin flip, really?" | `(acc − 0.25) / 0.75`. Turns 30% (which looks OK) into 0.067 (which looks honest). | 0–1, higher better |
| **Parse Failure Rate** | How often the answer was unreadable rather than wrong. | Share of responses with no extractable letter. A *different* failure from being wrong — it's an instruction-following problem. | 0–1, lower better |
| **Option Bias (TVD)** | Does it just love picking "C"? | Total variation distance between the letters it picks and the letters that are actually correct. | 0 = unbiased |

> **Why macro ≠ micro matters.** MMLU subjects range from 100 to 1 500
> questions. Pooled accuracy silently weights `professional_law` 15× more than
> `abstract_algebra`. Report both, or say which one you mean.

> **Why parse failure is separated.** A small model that outputs a paragraph
> instead of a letter isn't ignorant — it's disobedient. Counting that as
> "wrong" hides the actual fix (better prompt / constrained decoding).

---

## 3. MMLU — Probability & Calibration

Calibration asks: **when the model says it's 90% sure, is it right 90% of the
time?** A model that's right 60% of the time but *knows which 60%* is far more
useful than one that's right 65% with no self-awareness — you can route the
uncertain cases to a bigger model.

| Metric | ELI5 | Description | Range / good |
|---|---|---|---|
| **Correct Option Probability** | How much belief did it put on the right answer? | Mean `P(correct option)` from the answer token's distribution. | 0–1, higher better |
| **Log Probability** | Same thing, on a log scale so tiny numbers stay readable. | Mean `log P(correct)`. | ≤0, closer to 0 better |
| **NLL** | "How surprised was the model by the truth?" | Negative log-likelihood, `−mean(log P(correct))`. The quantity LMs are trained to minimise. | ≥0, lower better |
| **Cross Entropy** | Same number as NLL here. | Cross entropy against one-hot truth. Listed separately because readers look for it. | ≥0, lower better |
| **Perplexity** | "How many options was it effectively torn between?" | `exp(NLL)`. PPL 1 = certain and right; PPL 4 = as good as guessing on 4 options. | ≥1, lower better |
| **Entropy** | How spread out was its opinion? | Shannon entropy of the option distribution. High = unsure, 0 = all-in on one. | ≥0 |
| **Normalised Entropy** | Entropy on a 0–1 scale. | `H / log(n_options)`. Lets you compare 4-way MMLU with open-vocab LAMBADA. | 0–1 |
| **ECE** | "Does its confidence match reality?" | Expected Calibration Error: bin by confidence, compare avg confidence vs actual accuracy per bin, weight by bin size. | 0–1, **lower better** |
| **MCE** | The worst single bin. | Maximum Calibration Error — tail risk, not average risk. | 0–1, lower better |
| **Brier Score** | One number for "accurate *and* appropriately confident". | Mean squared error between confidence and outcome. A *proper scoring rule*: you can't game it by always saying 50%. | 0–1, lower better |
| **Confidence↔Accuracy correlation** | Does it know when it knows? | Point-biserial correlation between confidence and correctness. Near 0 = its confidence is noise. | −1..1, higher better |

### How the numbers are obtained here — and the catch

Chat APIs don't hand out probabilities for free. This project measured the
following on OpenRouter (September 2026):

| Model | Provider | Returns logprobs? |
|---|---|---|
| Llama-3.2-3B | Parasail | ✅ yes |
| Gemma-3-4B | DeepInfra | ❌ no |
| Ministral-8B | Mistral | ❌ no |

And a second, subtler catch: **with `max_tokens > 1` the provider returns
logprobs only for the *final* token** — which is the end-of-sequence marker,
whose distribution tells you nothing about the answer.

The fix is **single-token constrained scoring**: ask the question with
`max_tokens=1`, so the one generated token *is* the answer letter, and its
`top_logprobs` gives P(A), P(B), P(C), P(D) directly.

```
max_tokens=1  → tok[0]='A'          top: A −0.00, B −10.13, C −10.50, D −11.13  ✅ usable
max_tokens=2  → tok[0]='<|eot_id|>' top: eot −0.20, '.' −1.70, ...              ❌ useless
```

So the harness runs **two passes**:

| Pass | Prompt style | Gives you | Cost |
|---|---|---|---|
| Reasoning | chain-of-thought, then a letter | accuracy as the model would really be used | ~400 tokens |
| Scoring | direct, `max_tokens=1` | the probability distribution → all calibration | 1 token |

They answer different questions, so neither replaces the other. When logprobs
are unavailable the calibration block is written as
`{"available": false, "reason": ...}` — **never filled with zeros**, because a
fabricated ECE is worse than a missing one.

---

## 4. Consistency

Ask the same question twice. Do you get the same answer?

| Metric | ELI5 | Description | Range / good |
|---|---|---|---|
| **Answer Stability** | Ask 3 times — did it say the same thing all 3 times? | Fraction of items where every repeat gave an identical answer. | 0–1, higher better |
| **Self-Consistency** | Ask 5 times, take the most common answer. Does that beat asking once? | Majority-vote accuracy vs mean single-sample accuracy. A positive gain means the right answer *is* in there, it just doesn't surface reliably. | gain ≥0 is good |
| **Mean Agreement** | How dominant was the winning answer? | Mean of `max(vote count) / k`. 1.0 = unanimous every time. | 1/k..1 |
| **Seed Stability** | Run the whole benchmark 3× with different seeds — how much does the score wobble? | Std-dev and range of accuracy across seeds. | lower better |

> **The uncomfortable finding this usually produces:** at temperature 0, output
> should be deterministic — yet it often isn't. That instability comes from the
> *serving stack* (batching non-determinism, provider failover, kernel
> scheduling), not from sampling. If `accuracy_std` across seeds is comparable
> to the gap between two models, **that gap is noise** and ranking them is not
> supportable.

---

## 5. Robustness

Change the input in a way that shouldn't matter. Does the score survive?

| Metric | ELI5 | Description |
|---|---|---|
| **Prompt Variation** | Ask the same thing in different words. | Same question under 5 semantically identical templates (`terse`, `verbose`, `role_free`, …). |
| **Paraphrase** | Reword the question. | Deterministic surface rewrites ("Which of the following" → "Which one"). |
| **Option Reordering** | Shuffle A/B/C/D. The right answer is still right. | Permute options, remap the gold letter. Directly exposes positional bias. |
| **Perturbation** | Add realistic mess: typos, stray spaces, ALL CAPS. | Seeded character-level noise (adjacent-key typos, transpositions, casing, punctuation, unicode look-alikes). |

Scored by:

| Metric | ELI5 | Description | Good |
|---|---|---|---|
| **Accuracy Drop** | How much did the score fall? | `baseline_acc − variant_acc` | ~0 |
| **Relative Drop** | Fall as a share of the original. | `drop / baseline` | ~0 |
| **Flip Rate** | How many answers *changed at all*? | `(broke + fixed) / n` | low |
| **Robustness Score** | One 0–1 summary. | `1 − mean(relative drop)`, clipped to [0,1]. | →1 |

> **Why flip rate exists.** Accuracy can be unchanged because equal numbers of
> answers broke and got accidentally fixed. That's instability wearing a
> robustness costume. `broke` and `fixed` are reported separately for exactly
> this reason.

> **All perturbations are meaning-preserving.** A change that alters the correct
> answer would measure dataset noise, not model robustness. Option reordering is
> the one that moves the gold letter — so it carries the remapping with it.

---

## 6. LAMBADA — Task Quality & Target Probability

| Metric | ELI5 | Description | Good |
|---|---|---|---|
| **Last-Word Accuracy** | Did it guess the final word? | Normalised (lowercased, de-punctuated) exact match. The standard LAMBADA metric. | higher |
| **Exact Target Match** | Did it match *character for character*? | Byte-identical, case and punctuation included. The gap vs the above shows how much your normalisation is flattering the model. | higher |
| **Stem Match** | Right word, wrong ending — "run" vs "running". | Match after stripping common inflections. Separates "wrong word" from "wrong form". | higher |
| **Target PPL** | How surprised was it by the true word? | `exp(−log P(target))` for the target token. | lower |
| **Full-Sequence PPL** | Same, over everything it generated. | Perplexity of the whole generated sequence. | lower |
| **NLL / Cross Entropy** | Surprise, in log units. | As in §3. | lower |
| **Target LogProb** | Raw score it gave the right word. | `log P(target)`. | →0 |
| **Target Rank** | Where did the right word sit in its ranked guesses? | 1-based position in the top-k. Rank 1 = it was the top pick. | 1 |
| **MRR** | Score that rewards "nearly right". | Mean of `1/rank`; 0 when the target never appears in top-k. Rank 2 scores 0.5 where accuracy scores 0. | higher |
| **Hit Rate @k** | Was it in the top 1/5/10? | Share of targets within top-k. | higher |
| **Prediction Entropy** | How torn was it? | Entropy of the next-token distribution. | context-dep. |

> **Why MRR earns its place.** Small models often score near-zero raw accuracy
> on LAMBADA. MRR distinguishes "had no idea" from "had it ranked 2nd" — a real
> difference that accuracy alone flattens to the same 0.

---

## 7. LAMBADA — Context & Tokenization

### Context — *is it actually reading the passage?*

LAMBADA is built so the last word is guessable from the **whole passage** but
not from the **final sentence alone**. That design only pays off if you test it.

| Metric | ELI5 | Description |
|---|---|---|
| **Context Utilization** | How much did the earlier text actually help? | `accuracy(full) − accuracy(last sentence only)` |
| **Utilization Ratio** | What share of its skill depends on the wider passage? | `gain / full_accuracy`. 1.0 = entirely context-driven, 0 = context added nothing. |
| **Context Sensitivity** | How much better than answering with no passage at all? | `accuracy(full) − accuracy(no context)` |
| **Context Ablation** | Feed it less and less, watch the score fall. | Sweep: full → last 20 words → last 10 → last sentence → nothing. |
| **Context-Length Sensitivity** | Does more context reliably help? | The accuracy curve over those ablations. Should rise monotonically. |
| **Position Sensitivity** | Does it lose the plot on long passages? | Accuracy bucketed by passage length (quartiles). A "lost in the middle" probe. |

> **The result that changes your interpretation.** A model scoring 40% on full
> LAMBADA that still scores 38% on the last sentence alone is *not* doing
> long-range comprehension — it's exploiting local n-gram statistics. Its
> headline number means something entirely different from a model that drops to
> 10%. Without this ablation you cannot tell those two apart.

> **Shuffled sentences** separates "uses the words" from "uses the order". A
> model unaffected by shuffling is doing bag-of-words matching.

### Tokenization — *is the tokenizer handicapping it?*

| Metric | ELI5 | Description |
|---|---|---|
| **Tokens Per Target Word** | How many pieces does the target word get chopped into? | Mean token count for the gold word (with its leading space). |
| **Fragmentation Rate** | What share of targets aren't a single clean token? | Share needing >1 token — these must be produced correctly several times over. |
| **Subword Exact Match** | Match at token level, not string level. | Catches leading-space / unicode mismatches. Divergence from string accuracy = a text-handling bug, not a model one. |
| **Accuracy by Fragmentation** | Score split by 1-token / 2-token / 3+-token targets. | **The payoff metric.** If accuracy collapses across buckets, the LAMBADA score is substantially a tokenizer artefact. |

> **Why the leading space matters.** BPE encodes `" dog"` and `"dog"` as
> *different tokens*. LAMBADA targets always follow a space; omitting it inflates
> every count and makes every model look worse than it is.

> **Why this is a fairness issue.** Gemma has a 262 k vocabulary, Llama 128 k.
> The same word may be 1 token for one and 3 for the other. Comparing their
> LAMBADA scores without measuring fragmentation is not a fair fight.

---

## 8. Systems metrics vs configuration variables

**This is the section the whole project design turns on.**

### The distinction

| | Configuration variable | Metric |
|---|---|---|
| Nature | An input you **choose** | An output you **observe** |
| Examples | TP, PP, DP, SP, CP, EP, dtype, batch size, temperature | TTFT, throughput, VRAM, accuracy, ECE |
| Lives in | `benchmark_config.py` → `RunConfig` | `metrics/` → result blocks |
| Changing it | is the experiment | is the finding |

**Why they must not be mixed.** If you drop `tensor_parallel: 4` into the same
dictionary as `accuracy: 0.47`, you invite a category error: ranking models on
a composite that silently blends a hardware layout with a quality score, or
"comparing" TP against a Brier score. A parallelism degree is not a virtue.

So:

```
RunConfig        →  independent variables (what we chose)
metrics/*        →  dependent variables   (what we observed)
analyse_sweep()  →  the relationship between them
```

### Systems metrics (dependent)

| Metric | ELI5 | Description | Good |
|---|---|---|---|
| **TTFT** | How long until the first word appears? | Time To First Token — dominated by prefill + queueing + network. What makes a chat feel responsive. | lower |
| **TPOT** | How fast do words stream after that? | Time Per Output Token, `(E2E − TTFT) / (tokens − 1)`. Excludes the first token deliberately. | lower |
| **E2E** | Total wait for the whole answer. | End-to-end latency. | lower |
| **Prefill Throughput** | How fast can it read the prompt? | `prompt_tokens / TTFT`. | higher |
| **Decode Throughput** | How fast can it write? | `completion_tokens / (E2E − TTFT)`. | higher |
| **Request Throughput** | How many requests per second overall? | `n_requests / wall_time`. System-level, not per-request. | higher |
| **p95 / p99 latency** | The bad days, not the average day. | Tail percentiles. For serving, the tail *is* the user experience. | lower |
| **VRAM / KV Cache** | How much GPU memory does it need? | Weights + KV cache. KV grows linearly with sequence × batch. | lower |
| **Energy** | Joules burned per answer. | Requires on-host power telemetry (NVML/RAPL). | lower |
| **Cost** | Actual money. | Provider-reported USD per request. | lower |
| **Cost per Correct Answer** | Money per *useful* answer. | `total_cost / n_correct`. **The number that should drive model choice** — a cheaper model that's wrong twice as often is not cheaper. | lower |
| **Cost per 1M tokens** | Blended token price actually paid. | `total_cost / total_tokens × 1e6`. Comparable against published list prices. | lower |
| **Reasoning tokens** | Tokens burned thinking, not answering. | From `completion_tokens_details`. 0 for a non-reasoning model — a real answer, not a gap. | lower |
| **Cached prompt tokens** | Prompt tokens the provider served from cache. | From `prompt_tokens_details`. Cheaper than fresh tokens. | higher |
| **Scaling Efficiency** | Did 4 GPUs give you 4× the speed? | `speedup / device_ratio`. 1.0 = perfect; the shortfall is communication overhead. **Self-hosted only.** | →1 |

### Stage 9 — Reliability: did the calls actually succeed?

Easy to forget, and invisible in an accuracy table: a run that quietly retried a
third of its requests and dropped four to timeouts produces the same accuracy
figure as a clean one.

| Metric | ELI5 | Description | Good |
|---|---|---|---|
| **Success rate** | How many calls came back at all? | Requests that returned without error. | →1 |
| **Failure rate** | How many died even after retries? | Transport failures, post-retry. | →0 |
| **Invalid output rate** | Came back, but said nothing usable. | HTTP 200 whose body yielded no answer. Counted apart from failures: it's a model/prompt problem, not a transport one. | →0 |
| **Timeout / 429 rate** | Which *kind* of failure? | Errors bucketed by class, so you can tell "provider overloaded" from "provider broken". | →0 |
| **Retry rate** | How hard did we have to try? | Retries per request, and the share of requests needing at least one. | →0 |
| **Provider failover** | Did OpenRouter switch backends mid-run? | More than one provider served the run — latency and calibration coverage then mix two backends, and the run is no longer one clean measurement. | no |
| **Reproducibility score** | Same input, same output? | Combines seed spread with repeat stability. Below 1.0 at temperature 0 means the *serving stack* is non-deterministic, not the sampler. | →1 |

---

## 9. The parallelism degrees (TP, PP, DP, SP, CP, EP)

Six ways to split a model across devices. **All are configuration**, never scores.

| Degree | ELI5 | What it splits | Helps | Hurts | Comms |
|---|---|---|---|---|---|
| **TP** — Tensor Parallel | Six people each do part of *the same* multiplication. | Each weight matrix, across GPUs | Latency, fits big models | All-reduce every layer | **Heavy** — needs NVLink |
| **PP** — Pipeline Parallel | An assembly line: each station owns some layers. | Layers, across GPUs | Memory; cheap comms | Pipeline "bubbles" at low batch | Light (point-to-point) |
| **DP** — Data Parallel | Six identical shops serving different customers. | Requests, across replicas | Throughput | Memory ×N; no latency gain | None (inference) |
| **SP** — Sequence Parallel | Split the *sentence* in the parts TP left duplicated. | Sequence dim in norm/dropout regions | Activation memory | Only meaningful with TP | Moderate |
| **CP** — Context Parallel | Split a very long document across GPUs for attention itself. | Sequence dim in attention (ring/Ulysses) | Very long contexts | Complex; comms-heavy | Heavy |
| **EP** — Expert Parallel | Specialists in different buildings; each token visits a few. | MoE experts, across GPUs | MoE capacity | **No effect on dense models** | All-to-all |

> **All three models here are dense**, so EP is recorded as `1` — present in the
> schema to make that explicit rather than absent.

> **`world_size` = TP × PP × DP × CP.** SP and EP re-partition work already
> counted; they don't multiply device count.

### What a sweep measures

Vary **one** degree, hold everything else fixed, observe:

| Should move | Should **not** move |
|---|---|
| TTFT, TPOT, E2E | Overall accuracy |
| Prefill / decode throughput | Macro accuracy |
| VRAM, KV cache | NLL, ECE, Brier |
| Communication overhead, energy, cost | Last-word accuracy |

**Parallelism changes arithmetic order, not model semantics.** Quality *should*
be flat. `analyse_sweep()` flags any quality drift beyond 0.5 pp as
`unexpected` — because that's a bug or non-determinism worth investigating,
**not a result to report**.

Example output from a TP sweep:

```
TP1  speedup=1.00  ideal=1.0  efficiency=1.00  comm_overhead=0.00
TP2  speedup=1.73  ideal=2.0  efficiency=0.87  comm_overhead=0.13
TP4  speedup=2.80  ideal=4.0  efficiency=0.70  comm_overhead=0.30

quality_drift: overall_accuracy spread=0.004  → flagged: []
```

Reading it: doubling to TP2 bought 1.73× (87% efficient); going to TP4 bought
only 2.80× (70% efficient) — the missing 30% is all-reduce traffic. Accuracy
moved 0.4 pp, within noise, exactly as it should.

---

## 10. Benchmark process flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 0. DEFINE THE RUN                                               │
│    RunConfig: model, parallelism, serving, decoding, dataset,   │
│    evaluation. Validate it. Stamp it into every result file.    │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. PROBE CAPABILITIES                                           │
│    One trivial call per model. Which provider served it? Are    │
│    logprobs available? Streaming? Usage/cost accounting?        │
│    → decides which metric blocks are even computable            │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. LOAD DATA (cached, deterministic)                            │
│    MMLU    → HF datasets-server, N questions/subject            │
│    LAMBADA → local plain-text splits, seeded sample             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ESTIMATE COST → confirm before spending                      │
│    calls = items × models × (1 + calib + repeats + robustness   │
│                              + ablations + seeds)               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌───────────────────┐                   ┌──────────────────────┐
│ 4a. MAIN PASS     │                   │ 4b. EXTRA PASSES     │
│  streamed         │                   │  calibration (1 tok) │
│  → answer         │                   │  repeats  → stability│
│  → TTFT/E2E       │                   │  robustness variants │
│  → tokens, cost   │                   │  context ablations   │
│  → provider       │                   │  seed reruns         │
└─────────┬─────────┘                   └──────────┬───────────┘
          └──────────────────┬─────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. COMPUTE METRIC BLOCKS  (metrics/)                            │
│    task_quality · probability · consistency · robustness        │
│    context · tokenization · systems                             │
│    Each block: available=true/false + reason. Never fabricate.  │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. AGGREGATE & RANK                                             │
│    composite = 0.55 quality + 0.15 calibration                  │
│              + 0.15 robustness + 0.15 efficiency                │
│    renormalised over components actually available              │
│    REFUSES to rank if RunConfigs differ (confounded)            │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. (OPTIONAL) PARALLELISM SWEEP — local backend only            │
│    vary one degree → analyse_sweep()                            │
│    systems effects + scaling efficiency + quality-drift check   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
                    report.md · results/v2/*.json
```

### Commands

```bash
# quick sanity run
python run_benchmark.py --benchmark mmlu --subjects astronomy --questions 3 --yes

# MMLU with calibration (2× calls, gives ECE/Brier/NLL)
python run_benchmark.py --benchmark mmlu --subjects stem --questions 5 --calibration

# the full study (expensive — read the printed multiplier first)
python run_benchmark.py --benchmark both --calibration --repeats 3 \
                        --robustness --context --seeds 1,2,3

# LAMBADA with context ablations
python run_benchmark.py --benchmark lambada --samples 100 --context --robustness
```

---

## 11. What this setup can and cannot measure

This project talks to models over **OpenRouter — someone else's GPUs**. That
creates a hard boundary, and pretending otherwise would be the fastest way to
make the whole report untrustworthy.

| Stage | Status here | Why |
|---|---|---|
| 1 Task Quality | ✅ measured | needs only the answer |
| 3 Consistency, 4 Context, 5 Robustness | ✅ measured | prompt manipulation + repeat calls |
| 6 API Performance | ✅ measured | from the streamed response, client-side |
| 7 Token Efficiency | ✅ provider-reported | `usage`, incl. reasoning and cached tokens |
| 8 Economics | ✅ provider-reported | real per-request cost in the `usage` payload |
| 9 Reliability | ✅ measured | error class, retries and provider identity are all client-side |
| **2 Probabilistic Quality** | ⚠️ **provider-dependent** | only some providers return logprobs — measured at 1 of 3 models here |
| 1 · LAMBADA PPL / NLL | ⚠️ **same dependency** | derived from logprobs, so it inherits stage 2's condition |
| 10 · VRAM / KV cache | ⚠️ analytic only, self-hosted path | computed from published architecture, labelled `analytic_model` |
| 10 · Energy | ❌ unavailable | needs NVML/RAPL on the serving host |
| 10 · Communication overhead | ❌ unavailable | we don't own the interconnect |
| **10 · TP / PP / DP / SP / CP / EP effects** | ❌ unavailable | the provider chooses the layout; we can't set or see it |

Every field carries a `source` tag:

| Tag | Meaning |
|---|---|
| `measured` | observed by this client |
| `provider_reported` | from the API's usage payload |
| `analytic_model` | computed from architecture facts we supplied |
| `unavailable` | cannot be known in this deployment — with a reason string |

**To unlock the bottom rows:** re-run against a local vLLM backend
(`local_vllm_config()` in `benchmark_config.py`). There TP/PP/DP/SP/CP/EP become
real knobs, and VRAM, KV cache, energy and communication overhead become
directly measurable.

---

## 12. How to read a result file

`results/v2/<model>_<benchmark>_v2.json`:

```jsonc
{
  "model": "meta-llama/llama-3.2-3b-instruct",
  "benchmark": "mmlu",

  "task_quality":  { "overall_accuracy": 0.47, "macro_accuracy_subject": 0.44,
                     "overall_accuracy_ci95": [0.36, 0.60], ... },

  "probability":   { "available": true, "coverage": 1.0,
                     "nll": 0.118, "perplexity": 1.13, "ece": 0.105, ... },
  //  or, honestly:
  //               { "available": false,
  //                 "reason": "provider did not return token logprobs" }

  "consistency":   { "available": false, "reason": "repeat runs not enabled" },
  "robustness":    { ... },
  "systems":       { "latency": {...}, "throughput": {...}, "cost": {...},
                     "memory": { "source": "analytic_model", ... },
                     "energy": { "available": false, "reason": "..." },
                     "unobservable": { "fields": [...], "reason": "..." } },

  "config":        { "parallelism": { "tensor_parallel": 1, ...,
                                      "controlled": false },   // ← inputs
                     "decoding": {...}, "dataset": {...} }
}
```

**Reading checklist:**

1. Check `config` first — what was actually run?
2. Check `probability.available` before quoting any calibration number.
3. Check `systems.*.source` before quoting VRAM or energy.
4. Check `comparison.comparable` before quoting a ranking.
5. Check CI widths before believing a gap between two models.

---

## 13. Statistical honesty

| Practice | Why |
|---|---|
| **Wilson confidence intervals** | At 5 questions/subject, the point estimate is nearly meaningless. Wilson stays inside [0,1] even at n=1, unlike the normal approximation. |
| **Report macro *and* micro** | They can differ by several points on MMLU. |
| **Normalised accuracy** | 30% on a 4-way task is 6.7% above chance, not "30% good". |
| **Seed spread vs model gap** | If `accuracy_std` across seeds ≈ the gap between two models, the gap is noise. Say so. |
| **Renormalised composites** | A model isn't penalised for a study we didn't run — weights are renormalised over available components, and `components_used` is printed. |
| **Refuse confounded comparisons** | `build_comparison()` sets `comparable: false` when RunConfigs differ. |
| **`available: false` + reason** | Never a zero standing in for a missing measurement. |

### Sample-size reality check

| Setup | Questions | 95% CI half-width @ 50% |
|---|---|---|
| 3 subjects × 5 | 15 | ±25 pp |
| 18 STEM × 5 | 90 | ±10 pp |
| 57 × 10 | 570 | ±4 pp |
| 57 × 100 | 5 700 | ±1.3 pp |

At 90 questions you cannot distinguish a 45% model from a 52% model. Plan the
sample size around the difference you need to detect.

---

## 14. Glossary

| Term | Plain meaning |
|---|---|
| **SLM** | Small Language Model — roughly ≤10 B parameters; runs on modest hardware |
| **Logits / logprobs** | Raw scores the model assigns each possible next token, in log space |
| **Token** | A word-piece. "riverbank" might be `river` + `bank` |
| **BPE** | Byte-Pair Encoding — the algorithm that decides those pieces |
| **Prefill** | Processing your prompt (parallel, compute-bound) |
| **Decode** | Generating output one token at a time (sequential, memory-bound) |
| **KV cache** | Stored attention keys/values so past tokens aren't recomputed. Grows with sequence × batch |
| **GQA** | Grouped Query Attention — several query heads share one KV head, shrinking the cache |
| **Greedy decoding** | Always take the highest-probability token (temperature 0) |
| **Few-shot** | Worked examples included in the prompt |
| **CoT** | Chain of Thought — "think step by step" before answering |
| **Proper scoring rule** | A score you can't improve by lying about your confidence (Brier is one; accuracy is not) |
| **Micro vs macro** | Average over items vs average over per-group averages |
| **Ablation** | Deliberately removing something to see how much it mattered |
| **Confounded** | Two things changed at once, so you can't attribute the effect |

---

## Further reading

| Source | Why |
|---|---|
| Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021 | The MMLU paper — subject list and category definitions |
| Paperno et al., *The LAMBADA dataset*, ACL 2016 | Why last-word prediction tests discourse, not syntax |
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | Where ECE and reliability diagrams come from |
| Wang et al., *Self-Consistency Improves CoT Reasoning*, ICLR 2023 | The majority-vote method in §4 |
| EleutherAI `lm-evaluation-harness` | The reference implementation of letter-scoring |
| Kwon et al., *Efficient Memory Management … PagedAttention (vLLM)*, SOSP 2023 | KV cache mechanics and the local backend path |
| Shoeybi et al., *Megatron-LM*, 2019 | Tensor and pipeline parallelism |
