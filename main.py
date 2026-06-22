"""
LAMBADA Benchmark Evaluation Pipeline

Orchestration script that runs every step in chronological order:
  1. Validate configuration
  2. Evaluate all three SLMs on LAMBADA
  3. Generate diagrams (charts + mermaid PNGs)
  4. Generate markdown report and slides
"""

import sys
import os


def main():
    print("=" * 60)
    print("  LAMBADA Benchmark Evaluation Pipeline")
    print("=" * 60)

    # config
    print("\n[Step 1/4] Validating configuration...")
    from config import OPENROUTER_API_KEY, MODELS, DATASET_FILES

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("       Create a .env file with your API key (see .env.example).")
        sys.exit(1)

    print(f"  API Key: {'*' * 10}...{OPENROUTER_API_KEY[-4:]}")
    print(f"  Models:  {', '.join(MODELS)}")

    for split, path in DATASET_FILES.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  Dataset [{split}]: {status}  ({path})")

    # evaluate
    print("\n[Step 2/4] Running LAMBADA evaluation...")
    from evaluate_lambada import run_evaluation

    results = run_evaluation("test")
    if not results:
        print("WARNING: No evaluation results were produced.")

    # diagrams
    print("\n[Step 3/4] Generating diagrams...")
    from generate_diagrams import generate_all_diagrams

    generate_all_diagrams("test")

    # reports
    print("\n[Step 4/4] Generating reports...")
    from generate_report import generate_all_reports

    generate_all_reports("test")

    print("\n[Step 5/5] Building presentation...")
    from make_pptx import build as build_pptx

    build_pptx("test")

    # results summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\nOutputs:")
    print("  results/          - JSON evaluation results")
    print("  diagrams/         - PNG charts and workflow diagrams")
    print("  report.md         - Full evaluation report")
    print("  report/slide.pptx - Presentation")
    print("\nTo launch the web app:")
    print("  ./run.sh")


if __name__ == "__main__":
    main()
