# Self-Study & Defence Guide

Everything you need to explain this project, defend the numbers, and answer the
questions an examiner is likely to ask.

Three layers, deliberately:

1. **ELI5** — the one-sentence version you can say without notes.
2. **The precise version** — what you say when pressed.
3. **The trap** — the follow-up question that catches people out, and the answer.

---

## Contents

1. [The 60-second pitch](#1-the-60-second-pitch)
2. [Workflow, step by step](#2-workflow-step-by-step)
3. [The ten stages](#3-the-ten-stages)
4. [Every metric, ELI5 + precise](#4-every-metric-eli5--precise)
5. [The results, and how to talk about them](#5-the-results-and-how-to-talk-about-them)
6. [Likely exam questions, with answers](#6-likely-exam-questions-with-answers)
7. [The hard questions](#7-the-hard-questions)
8. [Presentation walkthrough](#8-presentation-walkthrough)
9. [Numbers worth memorising](#9-numbers-worth-memorising)
10. [Glossary](#10-glossary)

---

## 1. The 60-second pitch

> We benchmarked three small language models on two tasks — MMLU (multiple-choice
> knowledge across 57 subjects) and LAMBADA (predict the last word of a passage).
> Rather than reporting a single accuracy number, we built a nine-stage
> evaluation that also measures how well-calibrated the models are, how stable
> their answers are, how much they depend on context, how robust they are to
> input changes, and what they cost in latency, tokens, money and reliability.
> The key design decision is that configuration — like parallelism degrees and
> decoding settings — is recorded as an *input* and never mixed into a score.
> Ministral-8B wins on quality, but the extra stages changed how we read the
> results in four separate places.

**If they only ask one thing, say this:** accuracy alone cannot distinguish a
wrong answer from an unparseable one, a confident-correct model from a
confident-wrong one, or a cheap model from a cheap-per-*correct-answer* model.
Each stage exists because it separates two things accuracy conflates.

---

## 2. Workflow, step by step

| # | Step | What happens | Why it matters |
|---|---|---|---|
| 1 | **Configure** | Build a `RunConfig`: models, decoding parameters, dataset, which optional passes to run, parallelism degrees | Stamped into every result file, so a number from March is provably comparable (or not) with one from September |
| 2 | **Pre-flight** | Probe each model with 4 cheap calls; measure its 429 rate | A 2,850-item run against a throttled provider takes hours. Four probes turn that into a ten-second decision |
| 3 | **Estimate** | Print the API-call multiplier (`2x calls ≈ 5,700 requests`) | It is easy to ask for a sweep that quietly costs 20× the base run |
| 4 | **Scored pass** | Chain-of-thought prompt, non-streamed → the headline accuracy | This is how the model would really be used |
| 5 | **Extended pass** | Identical prompt, streamed → TTFT, TPOT, tokens, cost, provider | Latency is unmeasurable without streaming |
| 6 | **Optional passes** | Calibration (1 token), repeats, robustness variants, context ablations | Each costs an extra pass, so each is opt-in |
| 7 | **Compute** | Build nine metric blocks; each marked `available` or not, with a reason | An absent measurement is reported as absent, never as zero |
| 8 | **Aggregate** | Rank models; renormalise weights over available components | A model is never penalised for a study that was not run |
| 9 | **Publish** | `results/v2/`, history, comparison, report, slides | The web view reads the stored blocks, so an old run keeps showing its own numbers |

### Why two passes?

This is the single most likely "wait, what?" question.

- The **scored pass** is not streamed, because the classic evaluator also parses
  and grades the reasoning text.
- The **extended pass** must be streamed, because TTFT — time to *first* token —
  cannot be observed from a single blocking response.

They are two separate API calls, so at temperature 0 they still differ slightly.
That residual is serving non-determinism, and the web view prints the delta
rather than showing two accuracies and leaving you to notice.

---

## 3. The ten stages

| # | Stage | ELI5 | Band | Observable here |
|---|---|---|---|---|
| 1 | Task Quality | Did it get the right answer? | P0 | yes |
| 2 | Probabilistic Quality | When it says "90% sure", is it right 90% of the time? | P1 | **only on some providers** |
| 3 | Reasoning & Consistency | Ask twice — same answer? | P1 | yes (needs repeats) |
| 4 | Context Behavior | Is it reading the whole passage, or just the last few words? | P1 | yes (needs ablation) |
| 5 | Robustness | Change the wording harmlessly — does the score survive? | P1 | yes (needs variants) |
| 6 | API Performance | How fast? | P0 | yes |
| 7 | Token Efficiency | How many tokens did it burn? | P0 | yes |
| 8 | Economics | What did it cost in real money? | P0 | yes |
| 9 | Reliability | Did the calls actually succeed? | P0 | yes |
| 10 | Hardware & Distributed | How is it sharded across GPUs? Power draw? | — | **no — excluded** |

**P0** = free with every run. **P1** = costs one extra pass over the dataset.

### Why stage 10 is excluded rather than estimated

We are a client of someone else's GPUs. The provider chooses the parallelism
layout, the batch composition and the hardware; none of it is exposed to the
caller. Any TP/PP/DP figure, VRAM number or joules-per-token we printed would be
invented. So the schema records those six degrees as **configuration variables**
with `controlled: false`, and the same suite can be re-run against a self-hosted
vLLM backend where they become real, settable knobs.

---

## 4. Every metric, ELI5 + precise

### Stage 1 — Task Quality

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **Overall accuracy** | How many did it get right? | correct ÷ questions *answered* | higher |
| **Macro accuracy** | Average the score per subject, then average those | Mean of per-subject accuracies, so a big subject can't dominate | higher |
| **Normalised accuracy** | How much better than a coin flip, honestly? | `(acc − 0.25) ÷ 0.75` for 4 options | higher |
| **Wilson 95% CI** | How much wiggle room does this number have? | Binomial interval that stays inside [0,1] at small n | narrower |
| **Error rate** | The share it got wrong | `1 − accuracy` | lower |
| **Parse-failure rate** | How often was the answer unreadable rather than wrong? | Responses with no extractable letter | lower |
| **Option-position bias** | Does it just love picking "C"? | Total variation distance between picked letters and correct letters | lower |

> **The trap:** *"Your accuracy denominator excludes API failures — isn't that
> flattering the model?"*
> **Answer:** It would be flattering if we hid them. We report `answered`,
> `error_rate` and `accuracy_including_errors` side by side. A request the
> provider refused with a 429 tells you nothing about what the model knows;
> scoring it as a wrong answer measures the provider's load, not the model.

### Stage 2 — Probabilistic Quality

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **P(correct option)** | How much belief did it put on the right answer? | Mean probability mass on the gold option | higher |
| **NLL** | How surprised was it by the truth? | `−mean(log P(correct))` | lower |
| **Perplexity** | How many options was it effectively torn between? | `exp(NLL)`. 1.0 = certain and right; 4.0 ≈ guessing on 4 | lower |
| **Entropy** | How spread out was its opinion? | Shannon entropy of the option distribution | context |
| **ECE** | Does its confidence match reality? | Bin by confidence; compare avg confidence vs actual accuracy per bin; weight by bin size | lower |
| **MCE** | The worst bin, not the average | Maximum calibration error — tail risk | lower |
| **Brier score** | Accurate *and* appropriately confident, in one number | Mean squared error between confidence and outcome. A *proper scoring rule* — cannot be gamed by always saying 50% | lower |
| **Confidence↔accuracy corr** | Does it know when it knows? | Point-biserial correlation. Near 0 = its confidence is noise | higher |

**How we get probabilities from a chat API** — worth knowing, it is a real finding:

```
max_tokens = 2  → logprobs describe '<|eot_id|>'   ← the EOS token. Useless.
max_tokens = 1  → logprobs describe 'A'            ← the answer. Usable.
```

Providers return log-probabilities for the **final** token only. With
`max_tokens=1` that final token *is* the answer letter, so its top-k gives
P(A), P(B), P(C), P(D) directly. This is the letter-scoring protocol used by
`lm-evaluation-harness`.

> **The trap:** *"Why is calibration missing for most of your models?"*
> **Answer:** Because it is a property of the **provider**, not the model.
> Llama returned log-probabilities via Parasail on one run and none via
> Cloudflare on the next — same model, same account, same day. We print the
> provider beside every row so an empty cell is interpretable, and we mark the
> block `available: false` with a reason rather than filling zeros.

### Stage 3 — Consistency

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **Answer stability** | Ask 3 times — same answer all 3? | Share of items where every repeat matched | higher |
| **Self-consistency gain** | Does majority-vote beat asking once? | Majority-vote accuracy − mean single-sample accuracy | higher |
| **Seed stability** | Re-run with a different seed — how much does the score wobble? | Std-dev of accuracy across seeds | lower |

> **The trap:** *"You use temperature 0 — shouldn't it be perfectly stable?"*
> **Answer:** It should, and it isn't. Any instability at temperature 0 comes
> from the **serving stack** — batching non-determinism, provider failover,
> kernel scheduling — not from sampling. That is itself a finding: if seed
> spread is comparable to the gap between two models, that gap is noise and we
> say so instead of ranking them.

### Stage 4 — Context Behavior (LAMBADA)

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **Context utilisation** | How much did the earlier text actually help? | `acc(full) − acc(last sentence only)` | higher |
| **Utilisation ratio** | What share of its skill needs the wider passage? | `gain ÷ full_accuracy`. 1.0 = entirely context-driven | higher |
| **Context ablation** | Feed it less and less, watch the score fall | Sweep: full → last 20 words → last 10 → last sentence → nothing | — |
| **Position sensitivity** | Does it lose the plot on long passages? | Accuracy bucketed by passage length | flat |

> **Why this matters:** LAMBADA is *designed* so the last word is guessable from
> the whole passage but not the final sentence. A model scoring 40% that still
> scores 38% on the last sentence alone isn't doing long-range comprehension —
> it's exploiting local n-grams. Without the ablation you cannot tell those two
> apart, and their headline numbers look identical.

### Stage 5 — Robustness

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **Accuracy drop** | How much did the score fall? | `baseline − variant` | ~0 |
| **Flip rate** | How many answers *changed at all*? | `(broke + fixed) ÷ n` | lower |
| **Broke / fixed** | Right→wrong, and wrong→right | Counted separately | lower |
| **Robustness score** | One 0–1 summary | `1 − mean(relative drop)`, clipped | →1 |

> **The trap:** *"Accuracy didn't change under perturbation — so it's robust?"*
> **Answer:** Not necessarily. Accuracy can be flat because equal numbers of
> answers broke and got accidentally fixed. That is instability wearing a
> robustness costume, which is why we report `broke` and `fixed` separately and
> track flip rate alongside the drop.

### Stages 6–8 — Serving and cost

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **TTFT** | How long until the first word appears? | Time to first *content* token (a role-only opening delta doesn't count) | lower |
| **TPOT** | How fast do words stream after that? | `(E2E − TTFT) ÷ (tokens − 1)` — excludes the first token deliberately | lower |
| **p95 / p99** | The bad days, not the average day | Tail percentiles. For serving, the tail *is* the user experience | lower |
| **Decode throughput** | How fast can it write? | Generated tokens per second | higher |
| **Reasoning tokens** | Tokens burned thinking, not answering | From `completion_tokens_details`. 0 is a real answer for a non-reasoning model | lower |
| **Cached prompt tokens** | Prompt tokens served from cache, so cheaper | From `prompt_tokens_details` | higher |
| **Cost per 1M tokens** | Blended price actually paid | `total_cost ÷ total_tokens × 1e6` | lower |
| **Cost per correct answer** | Money per *useful* answer | `total_cost ÷ n_correct` | lower |

> **The headline economic point:** cheapest-per-token is not cheapest-per-answer.
> A model with half the token price and two-thirds the accuracy is not cheaper.
> Cost per correct answer is the figure that should drive model selection.

### Stage 9 — Reliability

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **Success rate** | How many calls came back at all? | Requests returning without error | →1 |
| **Invalid output rate** | Came back, but said nothing usable | HTTP 200 whose body yielded no answer | →0 |
| **Timeout / 429 rate** | *Which kind* of failure? | Errors bucketed by class | →0 |
| **Retry rate** | How hard did we have to try? | Retries per request | →0 |
| **Provider failover** | Did the backend change mid-run? | More than one provider served the run | no |

> **Why this is P0 and not a footnote:** a run that silently retried a third of
> its requests and dropped four to timeouts produces the **same accuracy table**
> as a clean one. Without stage 9 the difference is unrecoverable after the fact.

### Tokenization (LAMBADA)

| Metric | ELI5 | Precise | Good |
|---|---|---|---|
| **Tokens per target word** | How many pieces does the answer get chopped into? | Mean token count for the gold word, *with its leading space* | lower |
| **Fragmentation rate** | What share of targets aren't one clean token? | Share needing >1 token | lower |
| **Accuracy by fragmentation** | Score split by 1 / 2 / 3+ token targets | The payoff metric | — |

> **Why the leading space matters:** BPE encodes `" dog"` and `"dog"` as
> different tokens. LAMBADA targets always follow a space; omitting it inflates
> every count and makes every model look worse than it is.

---

## 5. The results, and how to talk about them

### MMLU — 2,850 questions per model

| Model | Accuracy | 95% CI | Parse fail | TTFT | $ / correct |
|---|---|---|---|---|---|
| **Ministral-8B** | **79.3%** | 77.8–80.8% | 0.6% | 0.413 s | $0.000051 |
| Gemma-3-4B | 60.4% | 58.5–62.1% | 0.4% | 0.582 s | $0.000046 |
| Llama-3.2-3B | 54.7% | 52.8–56.5% | **11.7%** | 0.330 s | $0.000083 |

### LAMBADA — 1,000 passages per model

| Model | Accuracy | 95% CI | TTFT | $ / correct |
|---|---|---|---|---|
| **Ministral-8B** | **39.9%** | 36.9–43.0% | 0.537 s | $0.000096 |
| Llama-3.2-3B | 22.5% | 20.0–25.2% | 0.345 s | $0.000101 |
| Gemma-3-4B | 18.0% | 15.7–20.5% | 0.531 s | $0.000107 |

### The four findings that need the extra stages

**1. Llama's 11.7% parse-failure rate.** Roughly one MMLU question in nine
produced no readable answer. Its 54.7% understates what it knows — that is an
instruction-following failure, fixable with a better prompt or constrained
decoding. Being *wrong* is not fixable that way. Stage 1 separates them.

**2. Tokenization explains part of the LAMBADA gap.** Split by how many tokens
the gold word needs:

| Model | 1-token | 3+-token | Collapse |
|---|---|---|---|
| Ministral-8B | 44.3% | 34.5% | −9.8 pp |
| Llama-3.2-3B | 29.5% | 2.6% | −26.9 pp |
| Gemma-3-4B | 23.6% | 2.6% | −21.0 pp |

Ministral holds up on multi-token targets; the others collapse. Part of its
LAMBADA lead is the ability to produce multi-token continuations, not purely
better comprehension.

> **Caveat you must state:** Gemma and Llama fell back to the generic
> `cl100k_base` tokenizer because their Hugging Face repos are gated. Their
> fragmentation *rates* therefore describe a generic BPE vocabulary, not their
> own. The within-model trend is valid; cross-model rate comparison is not.

**3. Serving noise hidden in a clean-looking score.** Gemma needed 40 retries
and hit a 0.1% transport-failure rate. Llama's run was served by two different
providers in one earlier run. None of this appears in an accuracy table.

**4. Token price inverts.** Gemma has the cheapest tokens ($0.0821/1M) *and* the
cheapest correct answer ($0.000046) on MMLU — but Llama has cheaper tokens than
Ministral while costing **60% more per correct answer**, because its accuracy is
lower. Accuracy converts token price into value.

---

## 6. Likely exam questions, with answers

**Q: Why not just report accuracy?**
Because accuracy conflates things with different causes and different fixes.
Give the parse-failure example: Llama's 11.7%. Same score, completely different
remedy.

**Q: Why two benchmarks?**
They test different things. MMLU is knowledge and reasoning across 57 subjects;
LAMBADA is long-range reading comprehension with an open vocabulary. A model can
be good at one and poor at the other — and ours are: Ministral leads both, but
the gap between second and third *reverses* between them.

**Q: What is MMLU's chance score, and why does it matter?**
25% (four options). It's why we report normalised accuracy: 60.4% is 47.1% above
chance, not "60% good".

**Q: Why Wilson intervals rather than the normal approximation?**
At small n the normal approximation produces intervals outside [0,1] and behaves
badly. Wilson stays inside and stays sane at n=1. With 2,850 questions our
intervals are ±2 pp; at 32 questions they were ±15 pp — which is exactly why we
report them.

**Q: How do you know the models are comparable?**
Every result carries its `RunConfig`. `build_comparison()` checks that all
models share one configuration and sets `comparable: false` if not — refusing to
rank rather than publishing a confounded ranking.

**Q: What's the composite score, and isn't it arbitrary?**
`0.50·quality + 0.15·calibration + 0.15·robustness + 0.10·efficiency +
0.10·reliability`. The weights are a judgement, yes — so we expose every
component separately, renormalise over whichever were actually measured, and
print `components_used` on every row. The ranking is auditable, not a black box.

**Q: Why is calibration missing from your final run?**
The routed providers returned no log-probabilities. We mark the block
unavailable with its reason and redistribute the weight. Filling zeros would
have silently punished all three models for a provider's choice.

**Q: What would you do with more budget?**
Run the P1 stages at full scale — the final 2,850-question run only had the
scored, extended and calibration passes. Robustness, repeats and context
ablations were measured at smaller n. Also pin a logprob-capable provider so
stage 2 is populated.

**Q: What does TP/PP/DP have to do with anything if you can't measure it?**
It is in the schema as *configuration*, to make explicit what the run did not
control. Recording `1 / controlled: false` is the honest encoding of "unknown" —
and the same code runs a real parallelism sweep on a self-hosted backend.

---

## 7. The hard questions

**Q: Your two passes give different accuracies. Which is real?**
Both. They are two independent samples of the same model at temperature 0; the
delta is serving non-determinism. We display it explicitly rather than picking
one silently. It also bounds how much precision any single run deserves — if two
passes of the same model differ by 0.8 pp, a 0.5 pp gap between two models is
not a result.

**Q: Isn't excluding stage 10 just avoiding the hard part?**
The opposite. We could have printed a VRAM estimate from published architecture
numbers and called it a measurement. Instead the code computes it, labels it
`analytic_model`, and keeps it out of the OpenRouter path entirely. The hard
part is being disciplined about what you *don't* know.

**Q: You changed the accuracy denominator mid-project. Doesn't that invalidate
earlier numbers?**
It changes them, and that is the point — the earlier figures counted provider
429s as wrong answers. During a rate-limit streak that deflated the score
substantially. Both denominators are now reported, so old and new numbers are
reconcilable.

**Q: How do you know your robustness perturbations preserve meaning?**
They are deterministic and auditable: adjacent-key typos, casing, whitespace,
punctuation, and unicode homoglyphs. Option reordering *does* move the correct
answer, so it carries the gold-letter remapping with it. We deliberately avoided
a model-generated paraphraser, which would make the benchmark non-reproducible.

**Q: 11.7% parse failure — is that not just a bad prompt on your side?**
Partly, and we say so. It's the same prompt for all three models, and the other
two fail at 0.4% and 0.6%. So the prompt is serviceable; Llama-3.2-3B is
markedly worse at following the output format. Both readings are visible because
the metric is reported separately from accuracy.

---

## 8. Presentation walkthrough

22 slides. Suggested timing for a 15-minute slot:

| Slides | Content | Time | Say this |
|---|---|---|---|
| 1 | Title | 15 s | Scale: 8,550 MMLU + 3,000 LAMBADA items, 3 models, 9 stages |
| 2 | The question | 1.5 min | The table of what accuracy *cannot* distinguish — this frames everything |
| 3 | Architecture | 1.5 min | Nine measurable stages, one excluded by design |
| 4 | Config ≠ metric | 1.5 min | The design decision. "A parallelism degree is not a virtue" |
| 5 | Workflow | 1.5 min | Nine steps; highlight pre-flight and the cost estimate |
| 6–8 | MMLU results | 3 min | Accuracy + CI, then serving, then composite |
| 9–11 | LAMBADA results | 2 min | Faster — note the ranking *reverses* between benchmarks |
| 12 | Findings | 2 min | **The payoff slide.** One finding per stage |
| 13 | What we cannot measure | 1 min | The honesty slide. Land the closing line |
| 14–21 | Screenshots | 1 min | Flip through; pause on stage 2 unavailable and run config |
| 22 | Conclusions | 1 min | Four points, then stop |

**If you are short on time**, cut slides 9–11 to one and skip 14–21. Never cut
slide 12 — it is the argument for the whole project.

**Two lines worth delivering verbatim:**
- *"A parallelism degree is not a virtue."* (slide 4)
- *"A fabricated calibration figure would be worse than a missing one."* (slide 13)

### Screenshots to capture

`diagram-analysis/README.md` lists all 19 with what each should show. The eight
used in the deck:

```
analysis-01-history-overview.png       analysis-10-api-performance.png
analysis-03-stage-list.png             analysis-13-reliability.png
analysis-04-task-quality.png           analysis-14-run-config.png
analysis-06-calibration-unavailable.png  analysis-15-decoding-panel.png
```

Drop them into `diagram-analysis/` and re-run `python make_slides.py` — they are
embedded automatically, replacing the placeholder boxes.

---

## 9. Numbers worth memorising

| Fact | Value |
|---|---|
| MMLU items per model | 2,850 (57 subjects × 50) |
| LAMBADA passages per model | 1,000 |
| MMLU chance baseline | 25% |
| Best MMLU | Ministral-8B, 79.3% |
| Best LAMBADA | Ministral-8B, 39.9% |
| Llama parse-failure rate | 11.7% |
| Fastest TTFT | Llama-3.2-3B, 0.330 s |
| Cheapest per correct answer (MMLU) | Gemma-3-4B, $0.000046 |
| CI width at n=2,850 | ≈ ±2 pp |
| Stages measured / defined | 9 of 10 |
| Composite weights | 0.50 / 0.15 / 0.15 / 0.10 / 0.10 |

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| **SLM** | Small Language Model — roughly ≤10B parameters |
| **Logits / logprobs** | Raw scores the model gives each possible next token, in log space |
| **Token** | A word-piece. "riverbank" might be `river` + `bank` |
| **BPE** | Byte-Pair Encoding — the algorithm that decides those pieces |
| **Prefill** | Processing your prompt (parallel, compute-bound) |
| **Decode** | Generating output one token at a time (sequential, memory-bound) |
| **KV cache** | Stored attention keys/values so past tokens aren't recomputed |
| **GQA** | Grouped Query Attention — query heads share KV heads, shrinking the cache |
| **Greedy decoding** | Always take the most likely token (temperature 0) |
| **CoT** | Chain of Thought — "think step by step" before answering |
| **Proper scoring rule** | A score you can't improve by misstating your confidence (Brier is one; accuracy is not) |
| **Micro vs macro** | Average over items vs average over per-group averages |
| **Ablation** | Deliberately removing something to see how much it mattered |
| **Confounded** | Two things changed at once, so the effect can't be attributed |
| **TP / PP / DP / SP / CP / EP** | Tensor / Pipeline / Data / Sequence / Context / Expert parallelism |

---

## Sources

| Source | Used for |
|---|---|
| Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021 | MMLU subjects and categories |
| Paperno et al., *The LAMBADA dataset*, ACL 2016 | Why last-word prediction tests discourse |
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | ECE and reliability diagrams |
| Brier, *Verification of Forecasts Expressed in Terms of Probability*, 1950 | The proper scoring rule |
| Wilson, *Probable Inference…*, JASA 1927 | Small-sample confidence intervals |
| Wang et al., *Self-Consistency Improves CoT Reasoning*, ICLR 2023 | Majority-vote decoding |
| Gao et al., `lm-evaluation-harness` | Letter-scoring protocol |
| Shoeybi et al., *Megatron-LM*, 2019 | Tensor and pipeline parallelism |
| Kwon et al., *PagedAttention (vLLM)*, SOSP 2023 | KV-cache mechanics, self-hosted path |
