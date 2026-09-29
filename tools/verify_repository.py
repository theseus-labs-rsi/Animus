"""Run repository suites in isolated processes, retaining logs and source hashes.

Install requirements-test.txt, then: python -B tools/verify_repository.py
Optional positional substrings select selftest names for a focused regression.
External sockets are denied; loopback is needed for transport server fixtures.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
folder = ROOT / "output" / "repository_verification" / stamp
folder.mkdir(parents=True)
entry = Path(__file__).with_name("offline_test_entry.py")
SUITE_TIMEOUT_S = 300  # The complete native process suite exceeds 180s on Windows.


def source_manifest():
    listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
    paths = sorted({name for name in listing.split("\0") if name and
                    Path(name).suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".json", ".txt"}
                    and (ROOT / name).is_file()})
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


before = source_manifest()
suites = sorted(ROOT.glob("tests/*_selftest.py"))
if len(sys.argv) > 1:
    suites = [path for path in suites if any(part in path.name for part in sys.argv[1:])]
jobs = [(str(path.relative_to(ROOT)), [str(path)]) for path in suites]
if len(sys.argv) == 1:
    jobs.append(("pytest_harness", ["pytest", "-q", "tests/harness"]))
if not jobs:
    raise SystemExit("No matching test suites")
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHON_DOTENV_DISABLED="1",
    OPENAI_API_KEY="offline-disabled", MODEL="offline-disabled", OPENAI_BASE_URL="http://127.0.0.1:9/v1",
    PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tests"))


def run(job):
    name, arguments = job
    try:
        result = subprocess.run([sys.executable, "-B", str(entry), *arguments], cwd=ROOT,
            env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=SUITE_TIMEOUT_S)
        row = {"suite": name, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        row = {"suite": name, "returncode": "timeout", "stdout": str(error.stdout), "stderr": str(error.stderr)}
    row["log_sha256"] = hashlib.sha256((row["stdout"] + row["stderr"]).encode()).hexdigest()
    path = folder / (name.replace("/", "_").replace("\\", "_") + ".json")
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"suite": name, "returncode": row["returncode"], "log": str(path), "log_sha256": row["log_sha256"]}


results = []
with ThreadPoolExecutor(max_workers=3) as pool:
    for future in as_completed([pool.submit(run, job) for job in jobs]):
        row = future.result()
        results.append(row)
        if row["returncode"] != 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
after = source_manifest()
drift = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
summary = {"utc": stamp, "suites": len(results), "passed_suites": sum(row["returncode"] == 0 for row in results),
    "failed_suites": [row for row in results if row["returncode"] != 0],
    "results": sorted(results, key=lambda row: row["suite"]), "source_drift": drift,
    "external_network": "forbidden", "real_model_calls": 0, "source_files": len(before)}
(folder / "source_manifest.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
(folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False), flush=True)
print(str(folder / "summary.json"), flush=True)
raise SystemExit(bool(summary["failed_suites"] or drift))
