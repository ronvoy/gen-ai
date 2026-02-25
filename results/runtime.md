============================================================
  LAMBADA Benchmark Evaluation Pipeline
============================================================

[Step 1/4] Validating configuration...
  API Key: **********...870d
  Models:  x-ai/grok-3-mini, mistralai/ministral-14b-2512, deepseek/deepseek-r1-distill-qwen-32b
  Dataset [test]: OK  (_rsc/lambada-dataset\lambada_test_plain_text.txt)
  Dataset [development]: OK  (_rsc/lambada-dataset\lambada_development_plain_text.txt)
  Dataset [control_test]: OK  (_rsc/lambada-dataset\lambada_control_test_data_plain_text.txt)
  Dataset [rejected]: OK  (_rsc/rejected-data1/rejected\rejected_plain_text.txt)

[Step 2/4] Running LAMBADA evaluation...

============================================================
LAMBADA Evaluation - Split: test
============================================================
Loaded 100 passages from test split

Evaluating: x-ai/grok-3-mini
----------------------------------------
  [x-ai/grok-3-mini] 10/100 - Accuracy so far: 90.00%
  [x-ai/grok-3-mini] 20/100 - Accuracy so far: 85.00%
  [x-ai/grok-3-mini] 30/100 - Accuracy so far: 80.00%
  [x-ai/grok-3-mini] 40/100 - Accuracy so far: 80.00%
  [x-ai/grok-3-mini] 50/100 - Accuracy so far: 84.00%
  [x-ai/grok-3-mini] 60/100 - Accuracy so far: 86.67%
  [x-ai/grok-3-mini] 70/100 - Accuracy so far: 85.71%
  [x-ai/grok-3-mini] 80/100 - Accuracy so far: 83.75%
  [x-ai/grok-3-mini] 90/100 - Accuracy so far: 81.11%
  [x-ai/grok-3-mini] 100/100 - Accuracy so far: 81.00%
  Results saved to results\x-ai_grok-3-mini_lambada_test.json
  Accuracy: 81.00%
  Avg Response Time: 9.398s

Evaluating: mistralai/ministral-14b-2512
----------------------------------------
  [mistralai/ministral-14b-2512] 10/100 - Accuracy so far: 40.00%
  [mistralai/ministral-14b-2512] 20/100 - Accuracy so far: 25.00%
  [mistralai/ministral-14b-2512] 30/100 - Accuracy so far: 20.00%
  [mistralai/ministral-14b-2512] 40/100 - Accuracy so far: 17.50%
  [mistralai/ministral-14b-2512] 50/100 - Accuracy so far: 22.00%
  [mistralai/ministral-14b-2512] 60/100 - Accuracy so far: 21.67%
  [mistralai/ministral-14b-2512] 70/100 - Accuracy so far: 21.43%
  [mistralai/ministral-14b-2512] 80/100 - Accuracy so far: 21.25%
  [mistralai/ministral-14b-2512] 90/100 - Accuracy so far: 20.00%
  [mistralai/ministral-14b-2512] 100/100 - Accuracy so far: 20.00%
  Results saved to results\mistralai_ministral-14b-2512_lambada_test.json
  Accuracy: 20.00%
  Avg Response Time: 0.480s

Evaluating: deepseek/deepseek-r1-distill-qwen-32b
----------------------------------------
  [deepseek/deepseek-r1-distill-qwen-32b] 10/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 20/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 30/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 40/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 50/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 60/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 70/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 80/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 90/100 - Accuracy so far: 0.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 100/100 - Accuracy so far: 0.00%
  Results saved to results\deepseek_deepseek-r1-distill-qwen-32b_lambada_test.json
  Accuracy: 0.00%
  Avg Response Time: 0.826s

Summary saved to results\summary_test.json

[Step 3/4] Generating diagrams...

Generating diagrams...
========================================
  Saved: diagrams\project_workflow.png
  Saved: diagrams\architecture_comparison.png
  Saved: diagrams\evaluation_pipeline.png
  Saved: diagrams\lambada_task.png
  Saved: diagrams\benchmark_method.png
  Saved: diagrams\flow_grok3_mini.png
  Saved: diagrams\flow_ministral_14b.png
  Saved: diagrams\flow_deepseek_r1_distill.png
  Saved: diagrams\accuracy_comparison.png
  Saved: diagrams\response_time.png
  Saved: diagrams\combined_metrics.png
  Saved: diagrams\error_rate.png
  Saved: diagrams\radar_comparison.png
  Saved: diagrams\time_hist_x-ai_grok-3-mini.png
  Saved: diagrams\time_hist_mistralai_ministral-14b-2512.png
  Saved: diagrams\time_hist_deepseek_deepseek-r1-distill-qwen-32b.png

Diagram generation complete.

[Step 4/4] Generating reports...

Generating reports...
========================================
Report saved to report\report.md
Slides saved to report\slide.md

Report generation complete.

============================================================
  Pipeline Complete!
============================================================

Outputs:
  results/          — JSON evaluation results
  diagrams/         — PNG charts and workflow diagrams
  report/report.md  — Full evaluation report
  report/slide.md   — Presentation slides

To launch the interactive dashboard:
  streamlit run app.py

------------
Issue / Improvement:

DeepSeek-R1-Distill (0%) — This is a config issue. The model wraps all output in <think>...</think> reasoning tokens. With max_tokens=10, it exhausts its budget on thinking and never outputs the actual answer. Every prediction is "" (empty).
Ministral-14B (20%) — This is partly a prompt issue. The model does produce predictions, but they're often semantically plausible yet wrong (e.g., target "salzella" → prediction "begin"). A zero-shot single-line prompt doesn't give smaller models enough guidance.
------------

============================================================
  LAMBADA Benchmark Evaluation Pipeline
============================================================

[Step 1/4] Validating configuration...
  API Key: **********...870d
  Models:  x-ai/grok-3-mini, mistralai/ministral-14b-2512, deepseek/deepseek-r1-distill-qwen-32b
  Dataset [test]: OK  (_rsc/lambada-dataset\lambada_test_plain_text.txt)
  Dataset [development]: OK  (_rsc/lambada-dataset\lambada_development_plain_text.txt)
  Dataset [control_test]: OK  (_rsc/lambada-dataset\lambada_control_test_data_plain_text.txt)
  Dataset [rejected]: OK  (_rsc/rejected-data1/rejected\rejected_plain_text.txt)

[Step 2/4] Running LAMBADA evaluation...

============================================================
LAMBADA Evaluation - Split: test
============================================================
Loaded 100 passages from test split

Evaluating: x-ai/grok-3-mini
----------------------------------------
  [x-ai/grok-3-mini] 10/100 - Accuracy so far: 90.00%
  [x-ai/grok-3-mini] 20/100 - Accuracy so far: 85.00%
  [x-ai/grok-3-mini] 30/100 - Accuracy so far: 73.33%
  [x-ai/grok-3-mini] 40/100 - Accuracy so far: 72.50%
  [x-ai/grok-3-mini] 50/100 - Accuracy so far: 76.00%
  [x-ai/grok-3-mini] 60/100 - Accuracy so far: 76.67%
  [x-ai/grok-3-mini] 70/100 - Accuracy so far: 77.14%
  [x-ai/grok-3-mini] 80/100 - Accuracy so far: 76.25%
  [x-ai/grok-3-mini] 90/100 - Accuracy so far: 76.67%
  [x-ai/grok-3-mini] 100/100 - Accuracy so far: 76.00%
  Results saved to results\x-ai_grok-3-mini_lambada_test.json
  Accuracy: 76.00%
  Avg Response Time: 10.607s

Evaluating: mistralai/ministral-14b-2512
----------------------------------------
  [mistralai/ministral-14b-2512] 10/100 - Accuracy so far: 60.00%
  [mistralai/ministral-14b-2512] 20/100 - Accuracy so far: 60.00%
  [mistralai/ministral-14b-2512] 30/100 - Accuracy so far: 46.67%
  [mistralai/ministral-14b-2512] 40/100 - Accuracy so far: 47.50%
  [mistralai/ministral-14b-2512] 50/100 - Accuracy so far: 50.00%
  [mistralai/ministral-14b-2512] 60/100 - Accuracy so far: 50.00%
  [mistralai/ministral-14b-2512] 70/100 - Accuracy so far: 51.43%
  [mistralai/ministral-14b-2512] 80/100 - Accuracy so far: 52.50%
  [mistralai/ministral-14b-2512] 90/100 - Accuracy so far: 52.22%
  [mistralai/ministral-14b-2512] 100/100 - Accuracy so far: 51.00%
  Results saved to results\mistralai_ministral-14b-2512_lambada_test.json
  Accuracy: 51.00%
  Avg Response Time: 0.474s

Evaluating: deepseek/deepseek-r1-distill-qwen-32b
----------------------------------------
  [deepseek/deepseek-r1-distill-qwen-32b] 10/100 - Accuracy so far: 40.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 20/100 - Accuracy so far: 40.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 30/100 - Accuracy so far: 30.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 40/100 - Accuracy so far: 40.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 50/100 - Accuracy so far: 38.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 60/100 - Accuracy so far: 38.33%
  [deepseek/deepseek-r1-distill-qwen-32b] 70/100 - Accuracy so far: 41.43%
  [deepseek/deepseek-r1-distill-qwen-32b] 80/100 - Accuracy so far: 37.50%
  [deepseek/deepseek-r1-distill-qwen-32b] 90/100 - Accuracy so far: 40.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 100/100 - Accuracy so far: 40.00%
  Results saved to results\deepseek_deepseek-r1-distill-qwen-32b_lambada_test.json
  Accuracy: 40.00%
  Avg Response Time: 30.831s

Summary saved to results\summary_test.json

[Step 3/4] Generating diagrams...

Generating diagrams...
========================================
  Saved: diagrams\project_workflow.png
  Saved: diagrams\architecture_comparison.png
  Saved: diagrams\evaluation_pipeline.png
  Saved: diagrams\lambada_task.png
  Saved: diagrams\benchmark_method.png
  Saved: diagrams\flow_grok3_mini.png
  Saved: diagrams\flow_ministral_14b.png
  Saved: diagrams\flow_deepseek_r1_distill.png
  Saved: diagrams\accuracy_comparison.png
  Saved: diagrams\response_time.png
  Saved: diagrams\combined_metrics.png
  Saved: diagrams\error_rate.png
  Saved: diagrams\radar_comparison.png
  Saved: diagrams\time_hist_x-ai_grok-3-mini.png
  Saved: diagrams\time_hist_mistralai_ministral-14b-2512.png
  Saved: diagrams\time_hist_deepseek_deepseek-r1-distill-qwen-32b.png

Diagram generation complete.

[Step 4/4] Generating reports...

Generating reports...
========================================
Report saved to report\report.md
Slides saved to report\slide.md

Report generation complete.

============================================================
  Pipeline Complete!
============================================================

Outputs:
  results/          — JSON evaluation results
  diagrams/         — PNG charts and workflow diagrams
  report/report.md  — Full evaluation report
  report/slide.md   — Presentation slides

To launch the interactive dashboard:
  streamlit run app.py

--------------------
Issue / Improvements:
Increasing max_tokens: 100 and max_tokens_reasoning: 8096
--------------------

============================================================
  LAMBADA Benchmark Evaluation Pipeline
============================================================

[Step 1/4] Validating configuration...
  API Key: **********...870d
  Models:  x-ai/grok-3-mini, mistralai/ministral-14b-2512, deepseek/deepseek-r1-distill-qwen-32b
  Dataset [test]: OK  (_rsc/lambada-dataset\lambada_test_plain_text.txt)
  Dataset [development]: OK  (_rsc/lambada-dataset\lambada_development_plain_text.txt)
  Dataset [control_test]: OK  (_rsc/lambada-dataset\lambada_control_test_data_plain_text.txt)
  Dataset [rejected]: OK  (_rsc/rejected-data1/rejected\rejected_plain_text.txt)

[Step 2/4] Running LAMBADA evaluation...

============================================================
LAMBADA Evaluation - Split: test
============================================================
Loaded 100 passages from test split

Evaluating: x-ai/grok-3-mini
----------------------------------------
  [x-ai/grok-3-mini] 10/100 - Accuracy so far: 90.00%
  [x-ai/grok-3-mini] 20/100 - Accuracy so far: 90.00%
  [x-ai/grok-3-mini] 30/100 - Accuracy so far: 80.00%
  [x-ai/grok-3-mini] 40/100 - Accuracy so far: 82.50%
  [x-ai/grok-3-mini] 50/100 - Accuracy so far: 82.00%
  [x-ai/grok-3-mini] 60/100 - Accuracy so far: 80.00%
  [x-ai/grok-3-mini] 70/100 - Accuracy so far: 80.00%
  [x-ai/grok-3-mini] 80/100 - Accuracy so far: 78.75%
  [x-ai/grok-3-mini] 90/100 - Accuracy so far: 78.89%
  [x-ai/grok-3-mini] 100/100 - Accuracy so far: 78.00%
  Results saved to results\x-ai_grok-3-mini_lambada_test.json
  Accuracy: 78.00%
  Avg Response Time: 9.730s

Evaluating: mistralai/ministral-14b-2512
----------------------------------------
  [mistralai/ministral-14b-2512] 10/100 - Accuracy so far: 60.00%
  [mistralai/ministral-14b-2512] 20/100 - Accuracy so far: 65.00%
  [mistralai/ministral-14b-2512] 30/100 - Accuracy so far: 53.33%
  [mistralai/ministral-14b-2512] 40/100 - Accuracy so far: 52.50%
  [mistralai/ministral-14b-2512] 50/100 - Accuracy so far: 54.00%
  [mistralai/ministral-14b-2512] 60/100 - Accuracy so far: 53.33%
  [mistralai/ministral-14b-2512] 70/100 - Accuracy so far: 55.71%
  [mistralai/ministral-14b-2512] 80/100 - Accuracy so far: 56.25%
  [mistralai/ministral-14b-2512] 90/100 - Accuracy so far: 55.56%
  [mistralai/ministral-14b-2512] 100/100 - Accuracy so far: 54.00%
  Results saved to results\mistralai_ministral-14b-2512_lambada_test.json
  Accuracy: 54.00%
  Avg Response Time: 0.455s

Evaluating: deepseek/deepseek-r1-distill-qwen-32b
----------------------------------------
  [deepseek/deepseek-r1-distill-qwen-32b] 10/100 - Accuracy so far: 50.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 20/100 - Accuracy so far: 50.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 30/100 - Accuracy so far: 36.67%
  [deepseek/deepseek-r1-distill-qwen-32b] 40/100 - Accuracy so far: 42.50%
  [deepseek/deepseek-r1-distill-qwen-32b] 50/100 - Accuracy so far: 42.00%
  [deepseek/deepseek-r1-distill-qwen-32b] 60/100 - Accuracy so far: 43.33%
  [deepseek/deepseek-r1-distill-qwen-32b] 70/100 - Accuracy so far: 47.14%
  [deepseek/deepseek-r1-distill-qwen-32b] 80/100 - Accuracy so far: 43.75%
  [deepseek/deepseek-r1-distill-qwen-32b] 90/100 - Accuracy so far: 42.22%
  [deepseek/deepseek-r1-distill-qwen-32b] 100/100 - Accuracy so far: 43.00%
  Results saved to results\deepseek_deepseek-r1-distill-qwen-32b_lambada_test.json
  Accuracy: 43.00%
  Avg Response Time: 35.502s

Summary saved to results\summary_test.json

[Step 3/4] Generating diagrams...

Generating diagrams...
========================================
  Saved: diagrams\project_workflow.png
  Saved: diagrams\architecture_comparison.png
  Saved: diagrams\evaluation_pipeline.png
  Saved: diagrams\lambada_task.png
  Saved: diagrams\benchmark_method.png
  Saved: diagrams\flow_grok3_mini.png
  Saved: diagrams\flow_ministral_14b.png
  Saved: diagrams\flow_deepseek_r1_distill.png
  Saved: diagrams\accuracy_comparison.png
  Saved: diagrams\response_time.png
  Saved: diagrams\combined_metrics.png
  Saved: diagrams\error_rate.png
  Saved: diagrams\radar_comparison.png
  Saved: diagrams\time_hist_x-ai_grok-3-mini.png
  Saved: diagrams\time_hist_mistralai_ministral-14b-2512.png
  Saved: diagrams\time_hist_deepseek_deepseek-r1-distill-qwen-32b.png

Diagram generation complete.

[Step 4/4] Generating reports...

Generating reports...
========================================
Report saved to report\report.md
Slides saved to report\slide.md

Report generation complete.

============================================================
  Pipeline Complete!
============================================================

Outputs:
  results/          — JSON evaluation results
  diagrams/         — PNG charts and workflow diagrams
  report/report.md  — Full evaluation report
  report/slide.md   — Presentation slides

To launch the interactive dashboard:
  streamlit run app.py