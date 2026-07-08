"""
LAMBADA Benchmark - Flask web application.

Lets you pick one of three small language models, run the LAMBADA
word-prediction benchmark against it through the OpenRouter API, and
view the resulting metrics as a table and charts. A Docs tab renders
the project report (report.md) with live Mermaid diagrams.

The module exposes `application` so it can be served under Phusion
Passenger; it also runs directly with `python passenger_wsgi.py`.
"""

import os
import re
import json
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory, abort

from config import (
    MODELS, MODEL_INFO, DATASET_FILES, RESULTS_DIR, DIAGRAMS_DIR,
    PRESETS, MIN_SAMPLES, MAX_SAMPLES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "report.md")

app = Flask(__name__)


# helpers

def _short(model_id):
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])


def _results_path(model_id, split):
    safe = model_id.replace("/", "_")
    return os.path.join(BASE_DIR, RESULTS_DIR, f"{safe}_lambada_{split}.json")


def _summary_path(split):
    return os.path.join(BASE_DIR, RESULTS_DIR, f"summary_{split}.json")


def _history_path():
    return os.path.join(BASE_DIR, RESULTS_DIR, "history.json")


def _metric_from_result(r):
    info = MODEL_INFO.get(r["model"], {})
    return {
        "model": r["model"],
        "name": _short(r["model"]),
        "developer": info.get("developer", ""),
        "params": info.get("params", ""),
        "accuracy": round(r["accuracy"] * 100, 1),
        "correct": r["correct"],
        "total": r["total"],
        "avg_response_time": r["avg_response_time"],
        "errors": r["errors"],
    }


def load_model_result(model_id, split="test"):
    path = _results_path(model_id, split)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_summary(split="test"):
    """Recompute summary_<split>.json from the per-model result files."""
    models = []
    for m in MODELS:
        r = load_model_result(m, split)
        if not r:
            continue
        models.append({
            "model": r["model"],
            "name": _short(r["model"]),
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "avg_response_time": r["avg_response_time"],
            "errors": r["errors"],
        })

    summary = {"split": split, "models": models}
    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_summary_path(split), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def collect_metrics(split="test"):
    """Return the current metrics for every model that has a result file."""
    models = []
    for m in MODELS:
        r = load_model_result(m, split)
        if not r:
            continue
        models.append(_metric_from_result(r))
    return models


def load_history():
    """Return the saved run history, most recent first."""
    path = _history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return []


def append_history(run_type, samples, model_results, params=None):
    """Append a run record and return the stored entry."""
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": run_type,
        "samples": samples,
        "params": params or {},
        "models": [_metric_from_result(r) for r in model_results],
    }
    history = load_history()
    history.insert(0, entry)
    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_history_path(), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return entry


# run benchmark for each model

def run_single_benchmark(model_id, num_samples, params=None, split="test"):
    """Run LAMBADA for one model, save its result file, refresh the summary."""
    from config import OPENROUTER_API_KEY
    from evaluate_lambada import load_dataset, evaluate_model

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    filepath = DATASET_FILES.get(split)
    if not filepath:
        raise RuntimeError(f"Unknown split: {split}")
    abs_path = filepath if os.path.isabs(filepath) else os.path.join(BASE_DIR, filepath)
    if not os.path.exists(abs_path):
        raise RuntimeError(f"Dataset file not found: {abs_path}")

    passages = load_dataset(abs_path, num_samples)
    result = evaluate_model(model_id, passages, OPENROUTER_API_KEY, params)

    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_results_path(model_id, split), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rebuild_summary(split)
    return result


def _clamp(value, low, high):
    return max(low, min(value, high))


def parse_run_request(data):
    """Pull sample count and decoding params out of a request body."""
    from evaluate_lambada import default_params

    try:
        num_samples = int(data.get("samples", 25))
    except (TypeError, ValueError):
        num_samples = 25
    num_samples = _clamp(num_samples, MIN_SAMPLES, MAX_SAMPLES)

    params = default_params()

    def _num(key, low, high, cast):
        if key in data and data.get(key) not in (None, ""):
            try:
                params[key] = _clamp(cast(data.get(key)), low, high)
            except (TypeError, ValueError):
                pass

    _num("temperature", 0.0, 2.0, float)
    _num("top_p", 0.0, 1.0, float)
    _num("max_tokens", 1, 512, int)
    _num("frequency_penalty", -2.0, 2.0, float)
    _num("presence_penalty", -2.0, 2.0, float)
    _num("few_shot", 0, 5, int)

    return num_samples, params


# MMLU benchmark

def _mmlu_results_path(model_id):
    safe = model_id.replace("/", "_")
    return os.path.join(BASE_DIR, RESULTS_DIR, f"{safe}_mmlu.json")


def _mmlu_summary_path():
    return os.path.join(BASE_DIR, RESULTS_DIR, "summary_mmlu.json")


def load_mmlu_result(model_id):
    path = _mmlu_results_path(model_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_mmlu_summary():
    """Recompute the MMLU ranking summary from the per-model result files."""
    from evaluate_slm_mmlu import build_mmlu_summary

    results = [r for r in (load_mmlu_result(m) for m in MODELS) if r]
    summary = build_mmlu_summary(results)
    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_mmlu_summary_path(), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def collect_mmlu_metrics():
    """Return the current MMLU ranking rows for the web page."""
    return rebuild_mmlu_summary().get("ranking", [])


def run_mmlu_benchmark(model_id, subjects, questions, params=None):
    """Run MMLU for one model, save its result file, refresh the summary."""
    from config import OPENROUTER_API_KEY
    from evaluate_slm_mmlu import (
        load_mmlu_tasks, evaluate_model_mmlu, mmlu_default_params,
    )

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    tasks = load_mmlu_tasks(subjects, questions, BASE_DIR)
    if not tasks:
        raise RuntimeError("No questions loaded for the selected subjects.")

    result = evaluate_model_mmlu(
        model_id, tasks, OPENROUTER_API_KEY, params or mmlu_default_params()
    )

    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_mmlu_results_path(model_id), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rebuild_mmlu_summary()
    return result


def parse_mmlu_request(data):
    """Pull subjects, question count and decoding params from a request body."""
    from evaluate_slm_mmlu import (
        resolve_subjects, mmlu_default_params, MIN_QUESTIONS,
    )

    subjects = resolve_subjects(data.get("subjects", "stem"))
    if not subjects:
        raise ValueError("No valid MMLU subjects selected.")

    try:
        questions = int(data.get("questions", 5))
    except (TypeError, ValueError):
        questions = 5
    # Web runs are synchronous, so cap per-subject questions at 50 to keep
    # request times manageable (the CLI script allows up to 100).
    questions = _clamp(questions, MIN_QUESTIONS, 50)

    params = mmlu_default_params()
    for key, low, high, cast in (
        ("temperature", 0.0, 2.0, float),
        ("top_p", 0.0, 1.0, float),
        ("max_tokens", 64, 1024, int),
    ):
        if data.get(key) not in (None, ""):
            try:
                params[key] = _clamp(cast(data.get(key)), low, high)
            except (TypeError, ValueError):
                pass

    return subjects, questions, params


# mermaid renderer

_MERMAID_BLOCK = re.compile(
    r"```mermaid\s*\n(.*?)\n```[ \t]*\n*"          # the fenced mermaid block
    r"(?:!\[[^\]]*\]\([^)]*\)[ \t]*\n?)?",          # an optional duplicate image line
    re.DOTALL,
)


def render_report_html(script_root=""):
    if not os.path.exists(REPORT_PATH):
        return "<p>report.md not found.</p>"

    import markdown

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # pull out mermaid blocks

    blocks = []

    def _stash(match):
        blocks.append(match.group(1).strip())
        return f"\n\nMERMAIDBLOCK{len(blocks) - 1}ENDBLOCK\n\n"

    text = _MERMAID_BLOCK.sub(_stash, text)

    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )

    # Point report images at the served diagrams route, honouring the
    # app's base URI (e.g. /gen-ai) when mounted under a sub-path.
    html = html.replace('src="../diagrams/', f'src="{script_root}/diagrams/')
    html = html.replace('src="diagrams/', f'src="{script_root}/diagrams/')

    # Style Markdown tables with Bootstrap and make them scroll on small screens.
    html = html.replace(
        "<table>",
        '<div class="table-responsive">'
        '<table class="table table-bordered table-sm align-middle">',
    )
    html = html.replace("</table>", "</table></div>")

    # put the mermaid blocks back as live diagrams.
    for i, code in enumerate(blocks):
        html = html.replace(
            f"<p>MERMAIDBLOCK{i}ENDBLOCK</p>",
            f'<div class="mermaid">{code}</div>',
        )

    return html


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    from evaluate_lambada import default_params

    model_choices = [{"id": m, "name": _short(m)} for m in MODELS]
    return render_template(
        "index.html",
        models=model_choices,
        metrics=collect_metrics(),
        presets=PRESETS,
        defaults=default_params(),
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
    )


@app.route("/metrics")
def metrics():
    return jsonify(collect_metrics())


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(silent=True) or request.form
    model_id = data.get("model")
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model selected."}), 400

    num_samples, params = parse_run_request(data)

    try:
        result = run_single_benchmark(model_id, num_samples, params)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    append_history("single", num_samples, [result], params)

    metric = _metric_from_result(result)
    metric["metrics"] = collect_metrics()
    return jsonify(metric)


@app.route("/run_all", methods=["POST"])
def run_all():
    data = request.get_json(silent=True) or request.form
    num_samples, params = parse_run_request(data)

    results = []
    try:
        for model_id in MODELS:
            results.append(run_single_benchmark(model_id, num_samples, params))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    append_history("all", num_samples, results, params)

    return jsonify({
        "ran": [_short(r["model"]) for r in results],
        "samples": num_samples,
        "metrics": collect_metrics(),
    })


@app.route("/mmlu")
def mmlu():
    from evaluate_slm_mmlu import (
        SUBJECT_GROUPS, CATEGORY_LABELS, mmlu_default_params, MIN_QUESTIONS,
    )

    model_choices = [{"id": m, "name": _short(m)} for m in MODELS]
    groups = {
        key: {"label": CATEGORY_LABELS[key], "subjects": subjects}
        for key, subjects in SUBJECT_GROUPS.items()
        if key != "all"
    }
    return render_template(
        "mmlu.html",
        models=model_choices,
        groups=groups,
        ranking=collect_mmlu_metrics(),
        defaults=mmlu_default_params(),
        min_questions=MIN_QUESTIONS,
        max_questions=50,
    )


@app.route("/mmlu/run", methods=["POST"])
def mmlu_run():
    data = request.get_json(silent=True) or request.form
    model_id = data.get("model")

    try:
        subjects, questions, params = parse_mmlu_request(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    targets = MODELS if model_id == "all" else [model_id]
    if any(m not in MODELS for m in targets):
        return jsonify({"error": "Unknown model selected."}), 400

    results = []
    try:
        for m in targets:
            results.append(run_mmlu_benchmark(m, subjects, questions, params))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    run_params = dict(params)
    run_params["benchmark"] = "mmlu"
    run_params["subjects"] = f"{len(subjects)} subject(s), {questions} q each"
    append_history(
        "mmlu-all" if model_id == "all" else "mmlu-single",
        results[0]["total"], results, run_params,
    )

    return jsonify({
        "ran": [_short(r["model"]) for r in results],
        "subjects": subjects,
        "questions": questions,
        "ranking": collect_mmlu_metrics(),
    })


@app.route("/mmlu/metrics")
def mmlu_metrics():
    return jsonify(collect_mmlu_metrics())


@app.route("/mmlu/details")
def mmlu_details():
    """Per-question record (Q/A, model answer, reasoning) for one model."""
    model_id = request.args.get("model")
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model."}), 400
    result = load_mmlu_result(model_id)
    if not result:
        return jsonify({"error": "No MMLU results for this model yet."}), 404
    return jsonify({
        "model": model_id,
        "name": _short(model_id),
        "subjects": result.get("subjects", []),
        "subject_accuracy": result.get("subject_accuracy", {}),
        "reasoning": result.get("reasoning", {}),
        "results": result.get("results", []),
    })


@app.route("/history")
def history():
    return render_template("history.html", runs=load_history())


@app.route("/history/delete", methods=["POST"])
def history_delete():
    data = request.get_json(silent=True) or request.form
    run_id = data.get("id")
    history = load_history()
    remaining = [h for h in history if h.get("id") != run_id]
    with open(_history_path(), "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2)
    return jsonify({"ok": True, "removed": len(history) - len(remaining)})


@app.route("/docs")
def docs():
    sr = request.script_root
    report_html = os.path.join(BASE_DIR, "report.html")

    # Prefer the exported report.html; fall back to rendering report.md.
    if not os.path.exists(report_html):
        return render_template("docs.html", content=render_report_html(sr))

    with open(report_html, "r", encoding="utf-8") as f:
        html = f.read()

    # The VS Code export points Mermaid and KaTeX at local file:// paths that
    # do not exist on the server. Swap Mermaid for a CDN and drop KaTeX (the
    # report has no math).
    html = re.sub(
        r"file:/+[^\"']*?mermaid[^\"']*?\.js",
        "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js",
        html,
    )
    html = re.sub(r"<link[^>]*katex[^>]*>", "", html)

    # Serve report images through the app, honouring the base URI (e.g. /gen-ai).
    html = html.replace('src="../diagrams/', f'src="{sr}/diagrams/')
    html = html.replace('src="diagrams/', f'src="{sr}/diagrams/')

    # Inject a slim nav bar so the app tabs stay reachable from the report.
    nav = (
        '<nav style="background:#212529;padding:10px 16px;'
        'font-family:Arial,Helvetica,sans-serif;font-size:15px;">'
        f'<a href="{sr}/" style="color:#fff;margin-right:18px;'
        'text-decoration:none;font-weight:600;">LAMBADA Benchmark</a>'
        f'<a href="{sr}/" style="color:#cbd3da;margin-right:14px;text-decoration:none;">LAMBADA</a>'
        f'<a href="{sr}/mmlu" style="color:#cbd3da;margin-right:14px;text-decoration:none;">MMLU</a>'
        f'<a href="{sr}/history" style="color:#cbd3da;margin-right:14px;text-decoration:none;">History</a>'
        f'<a href="{sr}/docs" style="color:#fff;text-decoration:none;">Docs</a>'
        "</nav>"
    )
    html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + nav, html, count=1)

    return html


@app.route("/diagrams/<path:filename>")
def diagrams(filename):
    directory = os.path.join(BASE_DIR, DIAGRAMS_DIR)
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename)


# Phusion Passenger entry point.
application = app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8008, debug=True)
