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
import uuid
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory, abort

from config import (
    MODELS, MODEL_INFO, DATASET_FILES, RESULTS_DIR, DIAGRAMS_DIR,
    PRESETS, MMLU_PRESETS, MIN_SAMPLES, MAX_SAMPLES,
    DECODING_PARAMS, EVAL_PASSES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "report.md")
RESULTS_V2_DIR = os.path.join(RESULTS_DIR, "v2")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Background jobs (live terminal output)
#
# Each run executes in a daemon thread and appends log lines to an in-memory
# job record. The browser polls /progress/<job_id> and renders the lines in
# a terminal panel while the benchmark is still running.
# ---------------------------------------------------------------------------

JOBS = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 20


def _start_job(task):
    """Run `task(log)` in a background thread; return its job id.

    `task` receives a `log(line)` callable and its return value becomes the
    job's final payload, delivered to the client on the last progress poll.
    """
    job_id = uuid.uuid4().hex
    job = {"lines": [], "done": False, "error": None, "payload": None}

    with _JOBS_LOCK:
        if len(JOBS) >= _MAX_JOBS:
            for old_id in [k for k, j in JOBS.items() if j["done"]]:
                JOBS.pop(old_id, None)
                if len(JOBS) < _MAX_JOBS:
                    break
        JOBS[job_id] = job

    def log(line):
        job["lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")

    def worker():
        try:
            job["payload"] = task(log)
        except Exception as exc:
            job["error"] = str(exc)
            log(f"ERROR: {exc}")
        finally:
            job["done"] = True

    threading.Thread(target=worker, daemon=True).start()
    return job_id


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


def _dig(data, dotted, default=None):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return default if cur is None else cur


def enrich_history_models(entry):
    """Fill the extended columns of an entry's model rows from its own blocks.

    The classic runner records only name/accuracy/time, so composite,
    cost-per-correct and provider render as "-". Every one of those values is
    already inside the stage blocks stored with the entry, so they are derived
    here rather than left blank or dropped from the table.
    """
    blocks_by_model = entry.get("metrics") or {}
    if not blocks_by_model:
        return entry

    # Composite comes from the comparison built over this entry's own blocks,
    # so it reflects the run being displayed rather than the newest one on disk.
    ranks = {}
    try:
        from metrics import aggregate
        comp = aggregate.build_comparison(
            list(blocks_by_model.values()), entry.get("benchmark", "mmlu"))
        ranks = {r["model"]: r for r in comp.get("ranking", [])}
        entry["comparable"] = comp.get("comparable")
        # The composite is renormalised over whatever was measured, so the
        # table needs to say which components that was - and warn when the
        # models in one run were not scored over the same set.
        entry["composite_components"] = comp.get("components_used")
        entry["composite_missing"] = comp.get("components_missing")
        entry["composite_uniform"] = comp.get("components_uniform")
    except Exception:
        ranks = {}

    for row in entry.get("models", []):
        blocks = blocks_by_model.get(row.get("model"))
        if not blocks:
            continue
        rank = ranks.get(row["model"], {})
        row.setdefault("rank", rank.get("rank"))
        row.setdefault("composite_score", rank.get("composite_score"))
        row.setdefault("components_used", rank.get("components_used", []))
        row.setdefault("cost_per_correct_usd",
                       _dig(blocks, "economics.cost_per_correct_answer_usd"))
        row.setdefault("cost_total_usd", _dig(blocks, "economics.total_usd"))
        row.setdefault("ttft_mean", _dig(blocks, "api_performance.latency.ttft.mean"))
        providers = _dig(blocks, "reliability.providers_used", {}) or {}
        row.setdefault("provider", ", ".join(providers) or None)

    # Present the table in rank order. Left in insertion order the # column
    # reads 3, 2, 1. Unranked rows (a classic run, or a model whose blocks are
    # missing) sort last instead of failing the comparison against None.
    if any(r.get("rank") for r in entry.get("models", [])):
        entry["models"].sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
    return entry


def load_history():
    """Return the saved run history, most recent first."""
    path = _history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (ValueError, OSError):
        return []
    return [enrich_history_models(e) for e in history]


def append_history(run_type, samples, model_results, params=None,
                   benchmark="lambada", passes=None):
    """Append a run record and return the stored entry.

    Extended stage 1-9 blocks travel with the entry, so the History tab can
    show the full breakdown per run without re-reading results/v2 - and so a
    later run cannot overwrite the numbers an older entry is displaying.
    """
    extended = {}
    for r in model_results:
        blocks = r.get("extended")
        if blocks:
            extended[r["model"]] = blocks

    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": run_type,
        "benchmark": benchmark,
        "samples": samples,
        "params": params or {},
        "passes": passes or {},
        "models": [_metric_from_result(r) for r in model_results],
    }
    if extended:
        entry["extended"] = True
        entry["metrics"] = extended
        first = next(iter(extended.values()))
        entry["families"] = {
            k: bool(first.get(k) and first[k].get("available") is not False)
            for k in EXTENDED_KEYS if k in first
        }
        entry["parallelism"] = (first.get("config") or {}).get("parallelism", {})

    history = load_history()

    # Collapse a re-entrant write of the same run. The classic pass and the
    # extended pass finish seconds apart and both want to record the run; two
    # entries with slightly different accuracies for one click is a
    # contradiction on screen, not two results. The newer entry wins and
    # inherits whatever blocks the older one had.
    now = datetime.now()
    for i, prev in enumerate(history[:3]):
        if (prev.get("type") == entry["type"]
                and prev.get("samples") == entry["samples"]
                and prev.get("params") == entry["params"]):
            try:
                age = (now - datetime.strptime(
                    prev["timestamp"], "%Y-%m-%d %H:%M:%S")).total_seconds()
            except (ValueError, KeyError):
                age = 1e9
            if age <= 120:
                if not entry.get("metrics") and prev.get("metrics"):
                    entry["metrics"] = prev["metrics"]
                    entry["extended"] = True
                    entry["families"] = prev.get("families", {})
                    entry["parallelism"] = prev.get("parallelism", {})
                entry["id"] = prev.get("id", entry["id"])
                history.pop(i)
                break

    history.insert(0, entry)
    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_history_path(), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return entry


# run benchmark for each model

def run_single_benchmark(model_id, num_samples, params=None, split="test", log=None,
                         passes=None):
    """Run LAMBADA for one model, save its result file, refresh the summary."""
    from config import OPENROUTER_API_KEY
    from evaluate_lambada import load_dataset, evaluate_model

    log = log or (lambda line: None)

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    filepath = DATASET_FILES.get(split)
    if not filepath:
        raise RuntimeError(f"Unknown split: {split}")
    abs_path = filepath if os.path.isabs(filepath) else os.path.join(BASE_DIR, filepath)
    if not os.path.exists(abs_path):
        raise RuntimeError(f"Dataset file not found: {abs_path}")

    log(f"Loading LAMBADA {split} split...")
    passages = load_dataset(abs_path, num_samples)
    log(f"Loaded {len(passages)} passages.")
    log(f"Evaluating {_short(model_id)} via OpenRouter...")

    def progress(done, total, sample, correct):
        mark = "✗"
        if sample["error"]:
            mark = "!"
        elif sample["correct"]:
            mark = "✓"
        line = (
            f"[{done}/{total}] {mark} target=\"{sample['target']}\" "
            f"predicted=\"{sample['prediction'] or '-'}\" {sample['time']}s "
            f"| accuracy {correct / done:.1%}"
        )
        if sample["error"]:
            line += f" | error: {sample['error']}"
        log(line)

    result = evaluate_model(model_id, passages, OPENROUTER_API_KEY, params, progress)

    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_results_path(model_id, split), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rebuild_summary(split)

    result["extended"] = build_extended_for_run(
        model_id, "lambada", passages, params or {}, passes or {}, log,
    )
    log(
        f"Done: {_short(model_id)} - accuracy {result['accuracy']:.1%} "
        f"({result['correct']}/{result['total']}), "
        f"avg {result['avg_response_time']}s, {result['errors']} error(s)."
    )
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
    _num("top_k", 0, 100, int)
    _num("min_p", 0.0, 1.0, float)
    _num("max_tokens", 1, 512, int)
    _num("frequency_penalty", -2.0, 2.0, float)
    _num("presence_penalty", -2.0, 2.0, float)
    _num("repetition_penalty", 0.5, 2.0, float)
    _num("seed", 0, 9999, int)
    _num("few_shot", 0, 5, int)

    return num_samples, params, parse_passes(data)


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


def run_mmlu_benchmark(model_id, subjects, questions, params=None, log=None,
                       passes=None):
    """Run MMLU for one model through the extended (stage 1-9) pipeline.

    Every web run now produces the full metric taxonomy, not just accuracy:
    the classic result file is still written so the ranking table and Q/A
    viewer keep working, and the extended block is written alongside it and
    attached to the history entry.
    """
    from config import OPENROUTER_API_KEY
    from evaluate_slm_mmlu import (
        load_mmlu_tasks, evaluate_model_mmlu, mmlu_default_params,
    )

    log = log or (lambda line: None)
    passes = passes or {}

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    log(f"Fetching MMLU questions: {len(subjects)} subject(s) x {questions} each...")
    tasks = load_mmlu_tasks(subjects, questions, BASE_DIR)
    if not tasks:
        raise RuntimeError("No questions loaded for the selected subjects.")
    log(f"Loaded {len(tasks)} questions.")
    log(f"Evaluating {_short(model_id)} via OpenRouter (reasoning enabled)...")

    def progress(model, done, total, correct, question=None):
        if not question:
            return
        mark = "!" if question["error"] else ("✓" if question["correct"] else "✗")
        line = (
            f"[{done}/{total}] {mark} {question['subject'].replace('_', ' ')} "
            f"Q{question['index'] + 1}: picked {question['predicted_letter'] or '?'} "
            f"(correct {question['correct_letter']}) {question['time']}s "
            f"| accuracy {correct / done:.1%}"
        )
        if question["error"]:
            line += f" | error: {question['error']}"
        log(line)

    result = evaluate_model_mmlu(
        model_id, tasks, OPENROUTER_API_KEY,
        params or mmlu_default_params(), progress,
    )

    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR), exist_ok=True)
    with open(_mmlu_results_path(model_id), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    rebuild_mmlu_summary()
    log(
        f"Done: {_short(model_id)} - accuracy {result['accuracy']:.1%}, "
        f"avg {result['avg_response_time']}s, {result['errors']} error(s)."
    )

    result["extended"] = build_extended_for_run(
        model_id, "mmlu", tasks, params or mmlu_default_params(),
        passes, log,
    )
    return result


def build_extended_for_run(model_id, benchmark, items, params, passes, log):
    """Run the extended stage 1-9 pipeline for one model and return its blocks.

    Deliberately a second pass rather than a rewrite of the classic evaluator:
    the extended pipeline needs streaming (for TTFT/TPOT) and an optional
    single-token scoring call, neither of which the original evaluator does.
    Failures here are logged and swallowed - an extended-metrics problem must
    never lose a completed accuracy run.
    """
    try:
        import time as _time
        from benchmark_config import DecodingConfig, EvaluationConfig, hosted_api_config
        from metrics import aggregate
        from run_benchmark import run_lambada_item, run_mmlu_item
        from config import OPENROUTER_API_KEY

        want_calibration = bool(passes.get("calibration", True))
        repeats = int(passes.get("repeats", 1) or 1)

        log(f"Extended metrics: streaming pass over {len(items)} item(s)"
            + (" + calibration" if want_calibration else "") + "...")

        cfg = hosted_api_config(model_id, benchmark)
        cfg.decoding = DecodingConfig(
            temperature=params.get("temperature", 0.0),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_tokens", 384),
            seed=params.get("seed"),
            few_shot=params.get("few_shot", 0),
        )
        cfg.evaluation = EvaluationConfig(
            request_logprobs=want_calibration,
            repeats_for_consistency=repeats,
        )
        cfg.dataset.n_items = len(items)

        start = _time.perf_counter()
        records = []
        for i, item in enumerate(items, 1):
            if benchmark == "mmlu":
                rec = run_mmlu_item(model_id, item, OPENROUTER_API_KEY, params,
                                    with_calibration=want_calibration)
            else:
                rec = run_lambada_item(model_id, item, OPENROUTER_API_KEY, params,
                                       with_calibration=want_calibration)
            records.append(rec)
            if i % 10 == 0 or i == len(items):
                log(f"  extended {i}/{len(items)}")

        answer_sets = correct_answers = None
        if repeats > 1:
            key = "predicted_letter" if benchmark == "mmlu" else "prediction"
            log(f"  consistency: {repeats - 1} extra repeat(s)")
            answer_sets = [[r.get(key)] for r in records]
            for _ in range(repeats - 1):
                for j, item in enumerate(items):
                    rec = (run_mmlu_item(model_id, item, OPENROUTER_API_KEY, params)
                           if benchmark == "mmlu" else
                           run_lambada_item(model_id, item, OPENROUTER_API_KEY, params))
                    answer_sets[j].append(rec.get(key))
            correct_answers = [
                r.get("correct_letter") if benchmark == "mmlu"
                else _norm_word(r.get("target", "")) for r in records
            ]

        wall = _time.perf_counter() - start
        builder = (aggregate.build_mmlu_metrics if benchmark == "mmlu"
                   else aggregate.build_lambada_metrics)
        blocks = builder(model_id, records, run_config=cfg, wall_seconds=wall,
                         answer_sets=answer_sets, correct_answers=correct_answers)

        os.makedirs(os.path.join(BASE_DIR, RESULTS_V2_DIR), exist_ok=True)
        out = os.path.join(BASE_DIR, RESULTS_V2_DIR,
                           f"{model_id.replace('/', '_')}_{benchmark}_v2.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2, ensure_ascii=False)

        log(f"Extended metrics saved: stages "
            f"{', '.join(k for k in blocks if k in EXTENDED_KEYS)}")
        return blocks
    except Exception as exc:                       # never lose the main run
        log(f"! extended metrics unavailable: {exc}")
        return None


EXTENDED_KEYS = ("task_quality", "probability", "consistency", "context",
                 "robustness", "api_performance", "token_efficiency",
                 "economics", "reliability", "tokenization")


def _norm_word(word):
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", (word or "").lower())


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
    # Cap per-subject questions at 50 for web runs to keep run times
    # manageable (the CLI script allows up to 100).
    questions = _clamp(questions, MIN_QUESTIONS, 50)

    params = mmlu_default_params()
    # Full decoding surface. Ranges mirror config.DECODING_PARAMS so the UI and
    # the server agree on what is acceptable.
    for key, low, high, cast in (
        ("temperature", 0.0, 2.0, float),
        ("top_p", 0.0, 1.0, float),
        ("top_k", 0, 100, int),
        ("min_p", 0.0, 1.0, float),
        ("max_tokens", 8, 1024, int),
        ("frequency_penalty", -2.0, 2.0, float),
        ("presence_penalty", -2.0, 2.0, float),
        ("repetition_penalty", 0.5, 2.0, float),
        ("seed", 0, 9999, int),
    ):
        if data.get(key) not in (None, ""):
            try:
                params[key] = _clamp(cast(data.get(key)), low, high)
            except (TypeError, ValueError):
                pass

    passes = parse_passes(data)
    return subjects, questions, params, passes


def parse_passes(data):
    """Which extended evaluation passes to run. Calibration is on by default."""
    def flag(name, default):
        v = data.get(name, default)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)

    try:
        repeats = _clamp(int(data.get("repeats", 1) or 1), 1, 5)
    except (TypeError, ValueError):
        repeats = 1
    return {
        "calibration": flag("calibration", True),
        "robustness": flag("robustness", False),
        "context": flag("context", False),
        "repeats": repeats,
    }


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
        decoding_params=DECODING_PARAMS,
        eval_passes=EVAL_PASSES,
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

    num_samples, params, passes = parse_run_request(data)

    def task(log):
        result = run_single_benchmark(model_id, num_samples, params, log=log,
                                      passes=passes)
        append_history("single", num_samples, [result], params,
                       benchmark="lambada", passes=passes)
        metric = _metric_from_result(result)
        metric["metrics"] = collect_metrics()
        return metric

    return jsonify({"job_id": _start_job(task)})


@app.route("/run_all", methods=["POST"])
def run_all():
    data = request.get_json(silent=True) or request.form
    num_samples, params, passes = parse_run_request(data)

    def task(log):
        results = []
        for i, model_id in enumerate(MODELS):
            log(f"--- Model {i + 1}/{len(MODELS)}: {_short(model_id)} ---")
            results.append(
                run_single_benchmark(model_id, num_samples, params, log=log,
                                     passes=passes)
            )
        append_history("all", num_samples, results, params,
                       benchmark="lambada", passes=passes)
        return {
            "ran": [_short(r["model"]) for r in results],
            "samples": num_samples,
            "metrics": collect_metrics(),
        }

    return jsonify({"job_id": _start_job(task)})


@app.route("/progress/<job_id>")
def job_progress(job_id):
    """Incremental log lines (and, once done, the payload) for a run job."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown or expired job."}), 404
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0
    return jsonify({
        "lines": job["lines"][offset:],
        "next_offset": len(job["lines"]),
        "done": job["done"],
        "error": job["error"],
        "payload": job["payload"] if job["done"] else None,
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
        presets=MMLU_PRESETS,
        decoding_params=DECODING_PARAMS,
        eval_passes=EVAL_PASSES,
        defaults=mmlu_default_params(),
        min_questions=MIN_QUESTIONS,
        max_questions=50,
    )


@app.route("/mmlu/run", methods=["POST"])
def mmlu_run():
    data = request.get_json(silent=True) or request.form
    model_id = data.get("model")

    try:
        subjects, questions, params, passes = parse_mmlu_request(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    targets = MODELS if model_id == "all" else [model_id]
    if any(m not in MODELS for m in targets):
        return jsonify({"error": "Unknown model selected."}), 400

    def task(log):
        results = []
        for i, m in enumerate(targets):
            if len(targets) > 1:
                log(f"--- Model {i + 1}/{len(targets)}: {_short(m)} ---")
            results.append(run_mmlu_benchmark(
                m, subjects, questions, params, log=log, passes=passes))

        run_params = dict(params)
        run_params["benchmark"] = "mmlu"
        run_params["subjects"] = f"{len(subjects)} subject(s), {questions} q each"
        append_history(
            "mmlu-all" if model_id == "all" else "mmlu-single",
            results[0]["total"], results, run_params,
            benchmark="mmlu", passes=passes,
        )
        return {
            "ran": [_short(r["model"]) for r in results],
            "subjects": subjects,
            "questions": questions,
            "ranking": collect_mmlu_metrics(),
        }

    return jsonify({"job_id": _start_job(task)})


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


@app.route("/history/<run_id>/secondary", methods=["POST"])
def history_secondary(run_id):
    """Fill a saved run's empty stages by running the passes it skipped.

    Stages 2 (probability), 3 (consistency) and 5 (robustness) are empty in most
    runs not because the metrics are unobtainable but because their extra passes
    were not requested. This re-runs exactly those passes over a *sample* of the
    original dataset and merges the resulting blocks back into the stored entry.

    A sample rather than the full set: robustness alone is four extra calls per
    item, so re-running 2,850 questions would cost more than the original
    benchmark. The sample size is recorded in each block it produces so nobody
    mistakes it for full-dataset coverage.
    """
    data = request.get_json(silent=True) or {}
    history = load_history()
    entry = next((e for e in history if e.get("id") == run_id), None)
    if not entry:
        return jsonify({"error": "Unknown run."}), 404
    if not entry.get("metrics"):
        return jsonify({"error": "This run has no extended blocks to extend."}), 400

    benchmark = entry.get("benchmark") or "mmlu"
    try:
        sample = _clamp(int(data.get("sample", 25)), 5, 200)
        repeats = _clamp(int(data.get("repeats", 3)), 1, 5)
    except (TypeError, ValueError):
        sample, repeats = 25, 3
    want = {
        "calibration": bool(data.get("calibration", True)),
        "repeats": repeats,
        "robustness": bool(data.get("robustness", True)),
        "context": bool(data.get("context", benchmark == "lambada")),
    }
    models = [m["model"] for m in entry.get("models", [])
              if m.get("model") in entry["metrics"]]

    def task(log):
        merged = run_secondary_analysis(entry, models, benchmark, sample, want, log)
        # Persist against the raw file so the enrichment pass does not
        # overwrite what we just computed.
        with open(_history_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        for e in raw:
            if e.get("id") == run_id:
                for model, blocks in merged.items():
                    e.setdefault("metrics", {})[model] = blocks
                first = next(iter(merged.values()), {})
                e["families"] = {
                    k: bool(first.get(k) and first[k].get("available") is not False)
                    for k in EXTENDED_KEYS if k in first
                }
                e["secondary_analysis"] = {
                    "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sample": sample, "passes": want,
                }
        with open(_history_path(), "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        log("Secondary analysis merged into history.")
        return {"run_id": run_id, "sample": sample,
                "models": list(merged), "passes": want}

    return jsonify({"job_id": _start_job(task)})


def run_secondary_analysis(entry, models, benchmark, sample, want, log):
    """Run the skipped passes on `sample` items and return merged blocks."""
    import random
    import time as _time

    from config import OPENROUTER_API_KEY
    from metrics import consistency as consistency_mod
    from metrics import robustness as robustness_mod
    from metrics import calibration as calibration_mod
    from metrics import context as context_mod
    from run_benchmark import run_lambada_item, run_mmlu_item

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    # Rebuild the original item set, then take a deterministic sample so the
    # study is repeatable.
    if benchmark == "mmlu":
        from evaluate_slm_mmlu import load_mmlu_tasks, resolve_subjects
        subjects = resolve_subjects("all")
        per = max(1, sample // max(len(subjects), 1) + 1)
        items = load_mmlu_tasks(subjects, per, BASE_DIR)
    else:
        from evaluate_lambada import load_dataset
        path = DATASET_FILES.get("test")
        abs_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        items = load_dataset(abs_path, sample * 3)
    random.Random(42).shuffle(items)
    items = items[:sample]
    log(f"Secondary analysis on {len(items)} sampled item(s), "
        f"{len(models)} model(s).")

    params = {"temperature": 0.0, "top_p": 1.0,
              "max_tokens": 384 if benchmark == "mmlu" else 32, "few_shot": 3}
    run_item = run_mmlu_item if benchmark == "mmlu" else run_lambada_item
    answer_key = "predicted_letter" if benchmark == "mmlu" else "prediction"
    merged = {}

    for model in models:
        blocks = dict(entry["metrics"][model])
        log(f"--- {_short(model)} ---")
        t0 = _time.perf_counter()

        base = [run_item(model, it, OPENROUTER_API_KEY, params,
                         with_calibration=want["calibration"]) for it in items]
        log(f"  base pass: {len(base)} item(s)")

        # Stage 2 — probability, from the single-token scoring pass.
        if want["calibration"]:
            probs = [r.get("correct_option_prob") for r in base]
            dists = [r.get("option_probs") or [] for r in base]
            correct = [bool(r.get("correct")) for r in base]
            idx = None
            if benchmark == "mmlu":
                idx = ["ABCD".index(r["correct_letter"])
                       if r.get("correct_letter") in "ABCD" else None for r in base]
            ranks = [r.get("target_rank") for r in base]
            block = calibration_mod.summarise_probability_block(
                probs, dists, correct, correct_indices=idx,
                ranks=ranks if any(x is not None for x in ranks) else None)
            if block.get("available"):
                block["sampled"] = len(items)
                blocks["probability"] = block
                log(f"  stage 2 probability: coverage {block['coverage']:.0%}")
            else:
                log("  stage 2 probability: provider returned no logprobs")

        # Stage 3 — consistency, from repeated asks.
        if want["repeats"] > 1:
            sets = [[r.get(answer_key)] for r in base]
            for k in range(want["repeats"] - 1):
                log(f"  repeat {k + 2}/{want['repeats']}")
                for j, it in enumerate(items):
                    sets[j].append(
                        run_item(model, it, OPENROUTER_API_KEY, params).get(answer_key))
            gold = [r.get("correct_letter") for r in base] if benchmark == "mmlu" \
                else [_norm_word(r.get("target", "")) for r in base]
            block = consistency_mod.build_consistency_block(sets, gold)
            block["sampled"] = len(items)
            blocks["consistency"] = block
            log(f"  stage 3 consistency: stability {block.get('answer_stability')}")

        # Stage 5 — robustness, from the perturbation sweep.
        if want["robustness"]:
            baseline = [bool(r.get("correct")) for r in base]
            variants = {}
            if benchmark == "mmlu":
                for tpl in ("terse", "verbose"):
                    log(f"  robustness: prompt_{tpl}")
                    got = [run_item(model, it, OPENROUTER_API_KEY, params,
                                    template=tpl) for it in items]
                    variants[f"prompt_{tpl}"] = robustness_mod.robustness_delta(
                        baseline, [bool(r.get("correct")) for r in got])
                log("  robustness: option_reorder")
                got = []
                for it in items:
                    ch, gold_i = robustness_mod.reorder_options(
                        it["choices"], it["answer"], "reverse")
                    got.append(run_item(model, {**it, "choices": ch, "answer": gold_i},
                                        OPENROUTER_API_KEY, params))
                variants["option_reorder"] = robustness_mod.robustness_delta(
                    baseline, [bool(r.get("correct")) for r in got])
                log("  robustness: typo_noise")
                got = [run_item(model,
                                {**it, "question": robustness_mod.typo_perturb(
                                    it["question"], 0.08, 7)},
                                OPENROUTER_API_KEY, params) for it in items]
                variants["typo_noise"] = robustness_mod.robustness_delta(
                    baseline, [bool(r.get("correct")) for r in got])
            else:
                for nm, fn in (("typo", robustness_mod.typo_perturb),
                               ("casing", robustness_mod.casing_noise)):
                    log(f"  robustness: {nm}")
                    got = [run_item(model, it, OPENROUTER_API_KEY, params,
                                    context_override=fn(it["context"], seed=11))
                           for it in items]
                    variants[nm] = robustness_mod.robustness_delta(
                        baseline, [bool(r.get("correct")) for r in got])
            block = robustness_mod.build_robustness_block(variants)
            block["sampled"] = len(items)
            blocks["robustness"] = block
            log(f"  stage 5 robustness: score {block.get('robustness_score')}")

        # Stage 4 — context ablation, LAMBADA only.
        if want["context"] and benchmark == "lambada":
            abl = {"full": [bool(r.get("correct")) for r in base]}
            for nm in ("last_sentence", "last_10_words", "no_context"):
                log(f"  context ablation: {nm}")
                fn = context_mod.CONTEXT_ABLATIONS[nm]
                abl[nm] = [bool(run_item(model, it, OPENROUTER_API_KEY, params,
                                         context_override=fn(it["context"])).get("correct"))
                           for it in items]
            block = context_mod.build_context_block(abl, base)
            block["sampled"] = len(items)
            blocks["context"] = block
            log(f"  stage 4 context: utilisation {block.get('context_utilization')}")

        blocks["secondary_analysis"] = {
            "sampled_items": len(items),
            "wall_seconds": round(_time.perf_counter() - t0, 1),
            "note": ("computed on a sample of the dataset, not the full run — "
                     "treat as indicative"),
        }
        merged[model] = blocks
    return merged


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

    # Both files come from `make_docs.py`, which writes the markdown first and
    # the HTML second. Falling back to a live markdown render whenever the HTML
    # is the older of the two keeps the tab correct if report.md is hand-edited.
    if (not os.path.exists(report_html)
            or os.path.getmtime(REPORT_PATH) > os.path.getmtime(report_html)):
        return render_template("docs.html", content=render_report_html(sr))

    with open(report_html, "r", encoding="utf-8") as f:
        html = f.read()

    # A no-op for the generated page, which already points at a CDN — but a
    # hand-exported report.html (e.g. from VS Code) hard-codes file:// paths
    # for Mermaid and KaTeX that do not resolve on the server.
    html = re.sub(
        r"file:/+[^\"']*?mermaid[^\"']*?\.js",
        "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js",
        html,
    )
    html = re.sub(r"<link[^>]*katex[^>]*>", "", html)

    # Serve report images through the app, honouring the base URI (e.g. /gen-ai).
    html = html.replace('src="../diagrams/', f'src="{sr}/diagrams/')
    html = html.replace('src="diagrams/', f'src="{sr}/diagrams/')

    # This branch serves the *exported* report.html, which does not extend
    # base.html — so the sticky nav and back-to-top button have to be injected
    # here too, or the Docs tab would be the one page without them.
    # nowrap + overflow-x keeps this to a single row on a narrow phone; left to
    # wrap it becomes two lines and the sticky bar eats a third of the screen.
    nav = (
        '<nav style="background:#212529;padding:10px 16px;position:sticky;top:0;'
        'z-index:1030;font-family:Arial,Helvetica,sans-serif;font-size:15px;'
        'white-space:nowrap;overflow-x:auto;">'
        f'<a href="{sr}/" style="color:#fff;margin-right:18px;'
        'text-decoration:none;font-weight:600;">SLM Benchmark</a>'
        f'<a href="{sr}/" style="color:#cbd3da;margin-right:14px;text-decoration:none;">LAMBADA</a>'
        f'<a href="{sr}/mmlu" style="color:#cbd3da;margin-right:14px;text-decoration:none;">MMLU</a>'
        f'<a href="{sr}/history" style="color:#cbd3da;margin-right:14px;text-decoration:none;">History</a>'
        f'<a href="{sr}/docs" style="color:#fff;text-decoration:none;">Docs</a>'
        "</nav>"
    )
    html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + nav, html, count=1)

    to_top = (
        '<button id="to-top" type="button" aria-label="Back to top" title="Back to top" '
        'style="position:fixed;right:1rem;bottom:1rem;z-index:1020;width:42px;height:42px;'
        'border-radius:50%;display:none;align-items:center;justify-content:center;'
        'border:1px solid #ced4da;background:rgba(255,255,255,.94);color:#495057;'
        'box-shadow:0 2px 10px rgba(0,0,0,.12);cursor:pointer;padding:0;">'
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        'viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" '
        'd="M8 15a.5.5 0 0 0 .5-.5V2.707l3.146 3.147a.5.5 0 0 0 .708-.708l-4-4a.5.5 0 0 '
        '0-.708 0l-4 4a.5.5 0 1 0 .708.708L7.5 2.707V14.5a.5.5 0 0 0 .5.5"/></svg></button>'
        '<script>(function(){var b=document.getElementById("to-top");if(!b)return;'
        'var t=function(){b.style.display=window.scrollY>300?"flex":"none";};'
        'window.addEventListener("scroll",t,{passive:true});t();'
        'b.addEventListener("click",function(){window.scrollTo({top:0,behavior:"smooth"});});'
        "})();</script>"
    )
    if "</body>" in html:
        html = html.replace("</body>", to_top + "</body>", 1)
    else:
        html += to_top

    return html


@app.route("/diagrams/<path:filename>")
def diagrams(filename):
    directory = os.path.join(BASE_DIR, DIAGRAMS_DIR)
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename)


# Figure folders referenced from report.md by relative path
# (diagram-lambada/, diagram-mmlu/, diagram-analysis/). Without this the
# report renders correctly as a file but every image 404s in the Docs tab.
FIGURE_DIRS = ("diagram-lambada", "diagram-mmlu", "diagram-analysis")


@app.route("/<folder>/<path:filename>")
def report_figures(folder, filename):
    # Whitelisted rather than routed by converter: the folder names contain
    # hyphens, which werkzeug's any() converter cannot parse unquoted.
    if folder not in FIGURE_DIRS:
        abort(404)
    directory = os.path.join(BASE_DIR, folder)
    if not os.path.exists(os.path.join(directory, filename)):
        abort(404)
    return send_from_directory(directory, filename)


# ---------------------------------------------------------------------------
# Extended metric analysis (results/v2)
#
# Serves the full metric taxonomy produced by run_benchmark.py. The templates
# render whatever blocks are present and show an explicit "not available"
# notice with the recorded reason for the rest, so the UI can never imply a
# measurement that was not taken.
# ---------------------------------------------------------------------------


def _v2_dir():
    return os.path.join(BASE_DIR, RESULTS_V2_DIR)


def load_v2_results(benchmark="mmlu"):
    """Every per-model v2 result file for a benchmark, newest run first."""
    directory = _v2_dir()
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(f"_{benchmark}_v2.json"):
            continue
        try:
            with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError):
            continue
        info = MODEL_INFO.get(data.get("model"), {})
        data["display_name"] = info.get("name", _short(data.get("model", "")))
        data["developer"] = info.get("developer", "")
        data["param_count"] = info.get("params", "")
        data["colour"] = info.get("color", "#6c757d")
        out.append(data)
    out.sort(key=lambda d: (d.get("config") or {}).get("timestamp", ""), reverse=True)
    return out


def load_v2_comparison(benchmark="mmlu"):
    path = os.path.join(_v2_dir(), f"comparison_{benchmark}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def v2_available_benchmarks():
    """Which benchmarks actually have extended results on disk."""
    directory = _v2_dir()
    if not os.path.isdir(directory):
        return []
    found = []
    for benchmark in ("mmlu", "lambada"):
        if any(n.endswith(f"_{benchmark}_v2.json") for n in os.listdir(directory)):
            found.append(benchmark)
    return found


@app.route("/analysis/data")
def analysis_data():
    """JSON API behind the analysis view - also what the History tab links to."""
    benchmark = request.args.get("benchmark", "mmlu")
    if benchmark not in ("mmlu", "lambada"):
        return jsonify({"error": "unknown benchmark"}), 400
    return jsonify({
        "benchmark": benchmark,
        "available_benchmarks": v2_available_benchmarks(),
        "results": load_v2_results(benchmark),
        "comparison": load_v2_comparison(benchmark),
    })


@app.route("/analysis/export/<benchmark>")
def analysis_export(benchmark):
    """Download the full extended result set as one JSON file."""
    if benchmark not in ("mmlu", "lambada"):
        abort(404)
    payload = {
        "benchmark": benchmark,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": load_v2_results(benchmark),
        "comparison": load_v2_comparison(benchmark),
    }
    return app.response_class(
        json.dumps(payload, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={
            "Content-Disposition":
                f"attachment; filename=slm-analysis-{benchmark}.json"
        },
    )


# Phusion Passenger entry point.
application = app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8008, debug=True)
