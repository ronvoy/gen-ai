#!/usr/bin/env bash
# Run the app locally with the Flask dev server - one command does it all:
# creates .venv, installs every dependency (runtime + report/slide tooling)
# and starts the server.
#
#     ./run_local.sh             install everything, then serve on :8008
#     ./run_local.sh check       install everything, then run check_deploy.py
#     ./run_local.sh install     install everything only
#
#     PORT=5000 ./run_local.sh   serve on another port
#     SKIP_INSTALL=1 ./run_local.sh   start without touching pip
#     VENV=/path/to/venv ./run_local.sh   use an existing virtualenv
#
# The production entry point is app.py under Passenger (see run_cpanel.sh and
# DEPLOYMENT.md); this script only drives `python app.py`, which runs the same
# Flask app with debug + auto-reload.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
MODE="${1:-serve}"
VENV_DIR="${VENV:-$APP_DIR/.venv}"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    say "creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR" || fail "python3 -m venv failed - is python3 (>= 3.9) installed?"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
say "using $($PY --version) from $VENV_DIR"

# requirements-dev.txt pulls in requirements.txt, so one line installs the
# runtime deps (flask, requests, python-dotenv, markdown) and the report/slide
# tooling (python-pptx, scipy, statsmodels). pip is a fast no-op once they
# are all present.
install_all() {
    if [ "${SKIP_INSTALL:-0}" = "1" ]; then
        say "SKIP_INSTALL=1 - not running pip"
        return
    fi
    say "installing all dependencies (requirements.txt + requirements-dev.txt)"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements-dev.txt
}

prepare() {
    mkdir -p results/v2
    if [ ! -f .env ]; then
        if [ -n "${OPENROUTER_API_KEY:-}" ]; then
            return
        fi
        if [ -f .env.example ]; then
            cp .env.example .env
            echo "    created .env from .env.example - put your OPENROUTER_API_KEY in it;"
        else
            echo "    no .env and OPENROUTER_API_KEY not set;"
        fi
        echo "    pages render without it, but no benchmark run will start."
    fi
}

case "$MODE" in
    serve)
        install_all
        prepare
        say "serving on http://127.0.0.1:${PORT:-8008} (Ctrl-C to stop)"
        exec "$PY" app.py
        ;;
    check)
        install_all
        prepare
        exec "$PY" check_deploy.py
        ;;
    install)
        install_all
        say "done"
        ;;
    *)
        fail "unknown mode '$MODE' (serve | check | install)"
        ;;
esac
