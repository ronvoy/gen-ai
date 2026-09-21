#!/usr/bin/env bash
# Deploy / restart this app on cPanel shared hosting (Passenger).
#
# Run it on the server, from the application root, in the cPanel Terminal or
# over SSH:
#
#     ./run_cpanel.sh            install deps, verify, restart Passenger
#     ./run_cpanel.sh check      verify only (no install, no restart)
#     ./run_cpanel.sh restart    restart Passenger only
#     ./run_cpanel.sh serve      run the Flask dev server on 127.0.0.1:8008
#                                (for an SSH-tunnelled smoke test; Passenger
#                                serves the real site)
#
# Set VENV to override the virtualenv auto-detection, e.g.
#     VENV=/home/user/virtualenv/gen-ai/3.12 ./run_cpanel.sh
#
# cPanel "Setup Python App" settings this script assumes:
#     Application root          <this directory>
#     Application startup file  app.py          <- NOT passenger_wsgi.py
#     Application Entry point   application
#
# With the startup file set to passenger_wsgi.py, cPanel's generated stub
# (also named passenger_wsgi.py) loads itself until RecursionError. That is
# the one deployment mistake this script actively repairs.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
MODE="${1:-deploy}"

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Virtualenv. cPanel creates it at /home/<user>/virtualenv/<app path>/<ver>,
# where <app path> is the application root relative to $HOME.
# --------------------------------------------------------------------------
find_venv() {
    if [ -n "${VENV:-}" ]; then
        echo "$VENV"; return
    fi
    local rel="${APP_DIR#"$HOME"/}"
    local base="$HOME/virtualenv/$rel"
    if [ -d "$base" ]; then
        # Newest Python version wins when several exist.
        local v
        v="$(ls -1d "$base"/*/ 2>/dev/null | sort -V | tail -n 1)"
        [ -n "$v" ] && { echo "${v%/}"; return; }
    fi
    [ -d "$APP_DIR/.venv" ] && { echo "$APP_DIR/.venv"; return; }
    echo ""
}

VENV_DIR="$(find_venv)"
if [ -n "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    say "virtualenv: $VENV_DIR"
else
    say "no virtualenv found - using $(command -v python3)"
    echo "    (create the app in cPanel > Setup Python App first, or set VENV=...)"
fi
PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || fail "no python interpreter on PATH"

# --------------------------------------------------------------------------
# passenger_wsgi.py must point at app.py, never at itself.
# --------------------------------------------------------------------------
fix_passenger_stub() {
    [ -f app.py ] || fail "app.py is missing - this is not the application root"
    if [ -f passenger_wsgi.py ] \
        && grep -q "load_source(" passenger_wsgi.py \
        && grep -q "passenger_wsgi.py['\"]" passenger_wsgi.py; then
        say "passenger_wsgi.py is cPanel's stub and it loads ITSELF - rewriting"
        cp passenger_wsgi.py passenger_wsgi.py.cpanel.bak
        cat > passenger_wsgi.py <<'PYEOF'
"""
Phusion Passenger entry point (shim). The Flask app lives in app.py.

Rewritten by run_cpanel.sh: cPanel had generated a stub here that loaded
passenger_wsgi.py (itself). Set "Application startup file" to app.py in
Setup Python App so the next save does not recreate that stub.
"""

from app import application  # noqa: F401
PYEOF
        echo "    Now set 'Application startup file' = app.py in Setup Python App,"
        echo "    or cPanel will write the broken stub back on the next save."
    elif [ ! -f passenger_wsgi.py ]; then
        say "passenger_wsgi.py missing - writing shim"
        printf 'from app import application  # noqa: F401\n' > passenger_wsgi.py
    fi
}

install_deps() {
    say "installing requirements"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet -r requirements.txt
}

prepare_dirs() {
    say "preparing writable directories"
    mkdir -p results/v2 tmp
    chmod 755 results results/v2
    if [ ! -f .env ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "    WARNING: no .env and OPENROUTER_API_KEY not set - pages will"
        echo "    render but no benchmark run will start (see DEPLOYMENT.md, 3)."
    fi
}

run_checks() {
    say "pre-flight (check_deploy.py)"
    "$PY" check_deploy.py
}

restart_passenger() {
    # Passenger reloads the app on the next request when tmp/restart.txt is
    # touched; this is what the cPanel "Restart" button does too.
    say "restarting Passenger"
    mkdir -p tmp
    touch tmp/restart.txt
    echo "    touched tmp/restart.txt - the next request loads the new code"
}

case "$MODE" in
    deploy)
        fix_passenger_stub
        install_deps
        prepare_dirs
        run_checks
        restart_passenger
        ;;
    check)
        fix_passenger_stub
        run_checks
        ;;
    restart)
        restart_passenger
        ;;
    serve)
        fix_passenger_stub
        say "dev server on http://127.0.0.1:8008 (Ctrl-C to stop)"
        exec "$PY" app.py
        ;;
    *)
        fail "unknown mode '$MODE' (deploy | check | restart | serve)"
        ;;
esac

say "done"
