# Deployment

Target: cPanel shared hosting with Passenger (LiteSpeed). The app is a plain
Flask WSGI application, so anything that can serve WSGI will run it.

## 1. Pre-flight

Run the checker **with the interpreter the server will use**, not your local one:

```bash
/home/<user>/virtualenv/<app>/3.12/bin/python check_deploy.py
```

It verifies the interpreter, the four runtime dependencies, the files Passenger
needs, and that `/`, `/mmlu`, `/history` and `/docs` all return 200. Exit status
is the number of failures.

## 2. Install

```bash
source /home/<user>/virtualenv/<app>/3.12/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is deliberately minimal — `flask`, `requests`,
`python-dotenv`, `markdown`. Report and slide generation needs
`requirements-dev.txt` (adds `python-pptx`, `scipy`, `statsmodels`), and none of
that belongs on the server: **generate `report.md`, `report.html` and the deck
locally and upload the output.** `metrics/significance.py` is pure stdlib for
exactly this reason, so the statistics in the report can be regenerated on the
server if you ever need to.

## 3. Configure

Set `OPENROUTER_API_KEY`, either in cPanel's environment-variable UI or in a
`.env` file at the app root. Without it the site still serves — history and docs
render fine — but no run will start.

`results/` must be writable by the application user (755).

## 4. Passenger

The Flask app is `app.py`, which defines a module-level `application`.
`passenger_wsgi.py` is a two-line shim that re-exports it for plain Passenger.
No `app.run()` is involved in production.

In cPanel → *Setup Python App*:

| Setting                  | Value            |
| ------------------------ | ---------------- |
| Application root         | this directory   |
| Application startup file | `app.py`         |
| Application Entry point  | `application`    |

**The startup file must not be `passenger_wsgi.py`.** cPanel overwrites
`passenger_wsgi.py` with its own stub that loads whatever the startup file is;
pointed at itself, the stub recurses until Python gives up (see
Troubleshooting).

## 5. One-shot deploy: `run_cpanel.sh`

From the application root in the cPanel Terminal (or over SSH):

```bash
./run_cpanel.sh            # install deps, repair the stub if needed, verify, restart
./run_cpanel.sh check      # verify only
./run_cpanel.sh restart    # touch tmp/restart.txt
./run_cpanel.sh serve      # Flask dev server on 127.0.0.1:8008 for a tunnelled smoke test
```

It finds the cPanel virtualenv from the app path (`VENV=...` overrides), runs
`check_deploy.py` with that interpreter, and rewrites a self-loading
`passenger_wsgi.py` stub into the shim (keeping a `.cpanel.bak`).

For local development use `./run_local.sh` instead: one run creates `.venv`,
installs everything (`requirements.txt` + `requirements-dev.txt`, so report
and slide generation work too), creates `.env` from `.env.example` if missing,
and serves `app.py` on `127.0.0.1:8008` (`PORT=...` to change,
`SKIP_INSTALL=1` to skip pip). `./run_local.sh check` runs the pre-flight with
that venv.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'encodings'`

```
Fatal Python error: init_fs_encoding: failed to get the Python codec
    of the filesystem encoding
ModuleNotFoundError: No module named 'encodings'
```

**This is not an application error.** The interpreter dies during start-up,
before a single line of this project is imported — which is why the traceback
never mentions it.

**Cause.** `PYTHONHOME` in the Passenger configuration points at a Python
version that is not on disk. cPanel writes that path when you pick a Python
version in the UI; if the virtualenv is later rebuilt on a different version
(or the host removes an old one), the path goes stale. The interpreter then
looks for its own standard library in a directory that does not exist and
cannot even load `encodings`.

**Confirm it.** Reproduce locally with any working interpreter:

```bash
PYTHONHOME=/nonexistent/virtualenv/app/3.14 python -c "print('hi')"
```

You get the identical fatal error. Note that this also means `check_deploy.py`
cannot report the problem — the interpreter cannot start to run it. If that
script produces no output at all, this is why.

**Fix.** Make the configured version and the installed version agree:

1. In cPanel → Setup Python App, read the Python version the app is set to.
2. On the server, check what is actually installed:
   `ls /home/<user>/virtualenv/<app>/`
3. If they differ, either set the app to the installed version, or destroy and
   recreate the virtualenv on the configured one.
4. Restart the app (`touch tmp/restart.txt`) and re-run `check_deploy.py`.

### `RecursionError` / traceback looping through `load_source` and `<module>`

```
  File "passenger_wsgi.py", line 13, in load_source
    loader.exec_module(module)
  File "passenger_wsgi.py", line 16, in <module>
  File "passenger_wsgi.py", line 13, in load_source
  ...
RecursionError: maximum recursion depth exceeded
```

**This is not this project's `passenger_wsgi.py`.** Those line numbers belong to
the stub cPanel writes into the application root when you save *Setup Python
App*:

```python
wsgi = load_source('wsgi', 'passenger_wsgi.py')   # the "Application startup file"
application = wsgi.application
```

With the startup file set to `passenger_wsgi.py` the stub loads itself,
forever. Nothing from `app.py` is ever imported, so no application error can
appear in the log.

**Fix.** Set *Application startup file* to `app.py` and *Application Entry
point* to `application`, save (cPanel rewrites the stub to point at `app.py`),
then restart. `./run_cpanel.sh` detects the self-loading stub and replaces it
with the shim in the meantime; `check_deploy.py` reports it as
`passenger_wsgi.py does not load itself`.

### A route 500s but the site loads

Check `stderr.log` in the app root. The usual cause is a missing runtime
dependency — `check_deploy.py` reports each one and what it is needed for.

### `/docs` is stale

`report.md` and `report.html` are build artefacts. Regenerate locally with
`python make_docs.py` and upload both. The route prefers `report.html` and
falls back to rendering `report.md` when the markdown is the newer of the two.
