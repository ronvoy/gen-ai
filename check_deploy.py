"""Pre-flight check for a shared-hosting (cPanel / Passenger) deployment.

Run it with the SAME interpreter the server will use — that is the whole point:

    /home/<user>/virtualenv/<app>/3.12/bin/python check_deploy.py

Every check prints why it matters, so a failure on the server is actionable
without a local reproduction. Exit status is the number of failures.
"""

import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

failures = 0
warnings = 0


def check(label, ok, detail="", ok_detail=""):
    """Report one check. `detail` explains a failure, `ok_detail` a pass."""
    global failures
    if ok:
        print(f"  PASS  {label}" + (f" - {ok_detail}" if ok_detail else ""))
    else:
        failures += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))
    return ok


def warn(label, ok, detail=""):
    global warnings
    if ok:
        print(f"  PASS  {label}")
    else:
        warnings += 1
        print(f"  WARN  {label}" + (f"\n        {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

def check_interpreter():
    print("\nInterpreter")
    print(f"  exe        {sys.executable}")
    print(f"  version    {sys.version.split()[0]}")
    print(f"  prefix     {sys.prefix}")

    check("Python >= 3.9", sys.version_info >= (3, 9),
          f"found {sys.version.split()[0]}; f-strings and dict ordering aside, "
          f"the type hints in metrics/ need 3.9+",
          ok_detail=sys.version.split()[0])

    # cPanel writes PYTHONHOME into the Passenger config, pinned to the Python
    # version chosen in the UI. Rebuild the virtualenv on a different version
    # and the path still points at the old one, so the interpreter cannot find
    # its own stdlib and dies with "ModuleNotFoundError: No module named
    # 'encodings'" before any app code runs - which is why that traceback never
    # mentions this project.
    #
    # Note what this check can and cannot do: if PYTHONHOME is badly wrong the
    # interpreter dies before it can run THIS script, so nothing below prints.
    # That silence is itself the diagnosis, and DEPLOYMENT.md documents it.
    # What is caught here is the survivable variant - a PYTHONHOME that exists
    # but points at the wrong version.
    home = os.environ.get("PYTHONHOME")
    if home:
        ok = os.path.isdir(home) and os.path.isdir(os.path.join(home, "lib"))
        check("PYTHONHOME resolves", ok,
              f"PYTHONHOME={home} does not exist. This is the cause of "
              f"'No module named encodings'. Either unset PYTHONHOME or point "
              f"the app's Python version in cPanel at the version actually "
              f"installed, then rebuild the virtualenv.",
              ok_detail=home)
    else:
        print("  PASS  PYTHONHOME unset - the interpreter finds its own stdlib")

    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    warn("running inside a virtualenv", venv,
         "not in a virtualenv; on cPanel the app must run from the virtualenv "
         "that has the dependencies, not the system Python")

    # A virtualenv built against a different minor version fails the same way.
    if venv:
        tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
        libdir = os.path.join(sys.prefix, "lib")
        if os.path.isdir(libdir):
            present = [d for d in os.listdir(libdir) if d.startswith("python")]
            check("virtualenv matches the interpreter version",
                  not present or tag in present,
                  f"virtualenv contains {present} but the interpreter is {tag}. "
                  f"Delete and rebuild the virtualenv on {tag}.",
                  ok_detail=tag)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

RUNTIME = [
    ("flask", "serves the web view"),
    ("requests", "every OpenRouter call"),
    ("dotenv", "reads OPENROUTER_API_KEY from .env"),
    ("markdown", "renders report.md at /docs"),
]

DEV_ONLY = [("pptx", "make_slides.py")]


def check_imports():
    print("\nRuntime dependencies")
    for mod, why in RUNTIME:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True, ok_detail=why)
        except Exception as exc:
            check(f"import {mod}", False,
                  f"{why}; install with: pip install -r requirements.txt ({exc})")

    print("\nDevelopment-only dependencies (not needed on the server)")
    for mod, why in DEV_ONLY:
        try:
            importlib.import_module(mod)
            print(f"  PASS  import {mod} - {why}")
        except Exception:
            print(f"  SKIP  import {mod} - absent, fine on the server ({why})")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    ("passenger_wsgi.py", "Passenger's entry point; must be at the app root"),
    ("config.py", "model list and API configuration"),
    ("templates/base.html", "every page extends it"),
    ("report.md", "the /docs view; regenerate with make_docs.py"),
    ("report.html", "the rendered /docs page"),
]


def check_files():
    print("\nApplication files")
    for rel, why in REQUIRED_FILES:
        path = os.path.join(BASE_DIR, rel)
        check(rel, os.path.exists(path), f"missing - {why}", ok_detail=why)

    results = os.path.join(BASE_DIR, "results")
    ok = os.path.isdir(results) and os.access(results, os.W_OK)
    check("results/ is writable", ok,
          "history and run output are written here; on cPanel set it to 755 "
          "and make sure it is owned by the application user")


def check_config():
    print("\nConfiguration")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env):
            with open(env, encoding="utf-8") as f:
                key = "OPENROUTER_API_KEY" in f.read()
    warn("OPENROUTER_API_KEY is set", bool(key),
         "absent - the web view loads and history renders, but no run will "
         "start. Set it in .env or in the cPanel environment variables.")

    try:
        sys.path.insert(0, BASE_DIR)
        import config
        check("config imports", True,
              ok_detail=f"{len(getattr(config, 'MODELS', []))} model(s) configured")
    except Exception as exc:
        check("config imports", False, str(exc))


def check_app():
    print("\nWSGI application")
    try:
        sys.path.insert(0, BASE_DIR)
        import passenger_wsgi
        app = getattr(passenger_wsgi, "application", None) or getattr(
            passenger_wsgi, "app", None)
        if not check("passenger_wsgi exposes an application", app is not None,
                     "Passenger imports passenger_wsgi and looks for a module-"
                     "level 'application'; without it the request 500s"):
            return
        client = app.test_client()
        for route in ("/", "/mmlu", "/history", "/docs"):
            r = client.get(route)
            check(f"GET {route}", r.status_code == 200,
                  f"returned {r.status_code}", ok_detail=f"{len(r.data):,} bytes")
    except Exception as exc:
        check("passenger_wsgi imports", False, f"{type(exc).__name__}: {exc}")


def main():
    print("=" * 66)
    print("Deployment pre-flight")
    print(f"  app root   {BASE_DIR}")
    print("=" * 66)

    check_interpreter()
    check_imports()
    check_files()
    check_config()
    check_app()

    print("\n" + "=" * 66)
    if failures:
        print(f"{failures} failure(s), {warnings} warning(s) - not ready to deploy")
    else:
        print(f"all checks passed, {warnings} warning(s)")
    print("=" * 66)
    return failures


if __name__ == "__main__":
    sys.exit(main())
