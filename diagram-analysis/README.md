# Analysis screenshots — placeholder manifest

`report.md` §7.6 references every file below by exact name. Drop a PNG in with
the matching filename and it appears in the report automatically — in the
rendered Markdown, in the exported `report.html`, and in the web app's **Docs**
tab (served by the `report_figures` route in `app.py`).

Nothing else needs editing. A missing file simply shows as a broken image until
you add it.

## Naming

```
diagram-analysis/analysis-<nn>-<slug>.png
```

Keep the number prefix — it is what keeps the report figures in order.

## Files the report expects

| # | Filename | What to capture | Where |
|---|---|---|---|
| 01 | `analysis-01-history-overview.png` | History page showing a run card: badge, family chips, param chips, ranking table | `/history` |
| 02 | `analysis-02-extended-collapsed.png` | The "Extended analysis" bar collapsed, with the per-model tabs visible underneath once opened | `/history` → run card |
| 03 | `analysis-03-stage-list.png` | All nine stage headers collapsed, showing the `n/a` badges on unavailable ones | expand **Extended analysis** |
| 04 | `analysis-04-task-quality.png` | Stage 1 open: accuracy cards, Wilson CI, subject/category tables, option bias | expand **1. Task Quality** |
| 05 | `analysis-05-calibration.png` | Stage 2 open on a model that *has* logprobs (Llama): NLL, PPL, ECE, Brier, reliability bins | expand **2. Probabilistic Quality** |
| 06 | `analysis-06-calibration-unavailable.png` | Stage 2 on a model without logprobs (Gemma/Ministral): the "Not available" note and its reason | switch model tab |
| 07 | `analysis-07-consistency.png` | Stage 3 open: answer stability, self-consistency gain, agreement | expand **3. Reasoning & Consistency** |
| 08 | `analysis-08-context.png` | Stage 4 open (LAMBADA run): context utilization, ablation table | LAMBADA run → **4. Context Behavior** |
| 09 | `analysis-09-robustness.png` | Stage 5 open: per-variant accuracy drop, flip rate, broke/fixed columns | expand **5. Robustness** |
| 10 | `analysis-10-api-performance.png` | Stage 6 open: TTFT/TPOT/E2E table with p50/p95/p99, throughput cards | expand **6. API Performance** |
| 11 | `analysis-11-token-efficiency.png` | Stage 7 open: prompt/completion/reasoning/cached tokens, tokens per correct answer | expand **7. Token Efficiency** |
| 12 | `analysis-12-economics.png` | Stage 8 open: cost per request / 1M tokens / correct answer | expand **8. Economics** |
| 13 | `analysis-13-reliability.png` | Stage 9 open: success/failure/timeout/429 rates, retries, provider failover | expand **9. Reliability** |
| 14 | `analysis-14-run-config.png` | Run Configuration open: the "inputs, not scores" notice and TP/PP/DP/SP/CP/EP chips | expand **Run Configuration** |
| 15 | `analysis-15-decoding-panel.png` | Decoding panel open: all nine parameters plus the extended-passes block and call estimate | `/mmlu` → **Decoding** |
| 16 | `analysis-16-mobile-subjects.png` | Phone width: two subject categories per row with their selected counts | `/mmlu` at ~390 px |
| 17 | `analysis-17-nav-modal.png` | The hamburger navigation modal open | any page → ☰ |

## Tips

- Use a consistent viewport (1280 px wide for desktop shots, ~390 px for the
  two mobile ones) so the figures sit at the same scale in the report.
- Capture the *element*, not the whole page, where a stage panel is tall —
  the report renders images at `max-width: 100%`, so a very tall screenshot
  becomes unreadably small.
- PNG preferred. JPEG works but text stays crisper in PNG.
