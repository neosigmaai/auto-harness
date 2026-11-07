"""Inspect a recorded terminal job without a worker, submission, or provider calls.

Example: python -m scripts.demo_recorded_review --report docs/live_calibrated_benchmark_review.json
The report locates its ignored private runs/.../control.json. Credentials never
appear in command arguments or printed configuration. The temporary API rejects
all HTTP methods except GET and HEAD, and only its own process is stopped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from uuid import UUID

import httpx

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
PRIVATE_ENVIRONMENT = {"DATABASE_URL", "AOS_API_KEY", "NEOSIGMA_OPEN_API_KEY",
                       "OPENAI_API_KEY", "OPEN_API_KEY", "E2B_API_KEY", "BOOTSTRAP_ADMIN_TOKEN"}
CHILD_ENVIRONMENT_ALLOWLIST = {"PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
                               "SSL_CERT_FILE", "SSL_CERT_DIR", "SYSTEMROOT"}


class InspectionError(Exception):
    """Only fixed, credential-free messages are exposed to the terminal."""


def load_recorded_context(report_path: Path):
    try:
        report = json.loads(report_path.read_text())
        job = report["job"]
        job_id = str(UUID(job["job_id"]))
        if job["status"] not in TERMINAL:
            raise InspectionError("The supplied report must describe a terminal job.")
        directory = (ROOT / report["run_directory"]).resolve()
        directory.relative_to((ROOT / "runs").resolve())
        control_path = (directory / "control.json").resolve()
        control_path.relative_to((ROOT / "runs").resolve())
        control = json.loads(control_path.read_text())
        if str(UUID(control["job_id"])) != job_id:
            raise InspectionError("The report and private control identify different jobs.")
        dsn, key = control["database_url"], control["api_key"]
        if (not isinstance(dsn, str) or urlsplit(dsn).scheme not in {"postgresql", "postgres"}
                or not isinstance(key, str) or not key.startswith("aos_") or len(key) < 12):
            raise InspectionError("The private control has an unsupported credential schema.")
        return {"job_id": job_id, "status": job["status"], "database_url": dsn, "api_key": key}
    except InspectionError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise InspectionError("The report or its private local control is missing or invalid.") from None


def child_environments(context):
    clean = {key: value for key, value in os.environ.items() if key in CHILD_ENVIRONMENT_ALLOWLIST}
    # Empty overrides also prevent service.config from loading these values from .env.
    api = {**clean, **{key: "" for key in PRIVATE_ENVIRONMENT},
           "DATABASE_URL": context["database_url"], "EXECUTOR": "simulated", "OPTIMIZER": "simulated"}
    client = {**clean, "AOS_API_KEY": context["api_key"]}
    return api, client


def create_inspection_app():
    from starlette.responses import JSONResponse
    from service.api.app import create_app

    app = create_app()

    @app.middleware("http")
    async def read_only(request, call_next):
        if request.method not in {"GET", "HEAD"}:
            return JSONResponse({"detail": "Recorded-history inspection is read-only."}, status_code=405)
        return await call_next(request)

    return app


def inspect(args):
    context = load_recorded_context(args.report)
    api_env, client_env = child_environments(context)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    process = None
    with tempfile.TemporaryDirectory(prefix="aos-history-inspection-") as folder:
        with (Path(folder) / "api.log").open("w") as log:
            try:
                process = subprocess.Popen([sys.executable, "-m", "uvicorn",
                    "scripts.demo_recorded_review:create_inspection_app", "--factory", "--host", "127.0.0.1",
                    "--port", str(port), "--no-access-log"], cwd=ROOT, env=api_env,
                    stdout=log, stderr=subprocess.STDOUT)
                with httpx.Client(base_url=base_url, timeout=5, trust_env=False) as client:
                    ready = False
                    readiness_deadline = time.monotonic() + 10
                    for _ in range(100):
                        if process.poll() is not None or time.monotonic() >= readiness_deadline:
                            break
                        try:
                            ready = client.get("/v1/readyz", timeout=1).status_code == 200
                        except httpx.TransportError:
                            pass
                        if ready:
                            break
                        time.sleep(.1)
                    if not ready:
                        raise InspectionError("The temporary inspection API did not become ready.")
                    result = client.get(f"/v1/jobs/{context['job_id']}",
                        headers={"Authorization": "Bearer " + context["api_key"]})
                    if result.status_code != 200:
                        raise InspectionError("The recorded job is unavailable to this service credential.")
                    observed = result.json()
                    if (observed.get("job_id") != context["job_id"] or
                            observed.get("status") != context["status"] or observed.get("status") not in TERMINAL):
                        raise InspectionError("The database job does not match the recorded terminal state.")
                command = [sys.executable, str(ROOT / "test_client.py"), "--base-url", base_url,
                           "--job-id", context["job_id"], "--quiet"]
                if args.json:
                    command.append("--json")
                completed = subprocess.run(command, cwd=ROOT, env=client_env, capture_output=True,
                                           text=True, timeout=120)
                # The existing client exits 2 for a terminal cancelled/failed job.
                expected_exit = 0 if context["status"] == "SUCCEEDED" else 2
                if completed.returncode != expected_exit:
                    raise InspectionError("The history client could not complete its read-only inspection.")
                output = completed.stdout
                for secret in (context["api_key"], context["database_url"]):
                    output = output.replace(secret, "[redacted]")
                print("Recorded-history replay only: no worker, job submission, or model calls.", flush=True)
                print(output, end="" if output.endswith("\n") else "\n", flush=True)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    print("Temporary inspection API stopped.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Local terminal controller report")
    parser.add_argument("--json", action="store_true", help="Include full history/traces; compact output is the default")
    args = parser.parse_args()
    try:
        inspect(args)
    except InspectionError as exc:
        print(f"Inspection stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Inspection stopped ({type(exc).__name__}); no private configuration is printed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
