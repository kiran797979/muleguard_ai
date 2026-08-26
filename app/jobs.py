"""
Upload a dataset from the browser and run the pipeline against it.

This exists for one moment: a judge hands over a file and you have minutes, not
an afternoon. Drag it into the page, watch the stages tick past, read the result.

SECURITY POSTURE
----------------
File upload is the largest attack surface in this project, so the controls are
deliberate rather than incidental:

  * Only a fixed allow-list of extensions is accepted. Nothing is ever executed,
    only parsed by pandas.
  * The client's filename is discarded except for its extension. A new safe name
    is generated, so `../../etc/passwd` and friends have nothing to traverse.
  * Uploads land in `runs/<job>/data/` and nowhere else. No request supplies a
    path, and the destination is derived, never accepted.
  * Streamed to disk in chunks with a hard size cap, so a large file cannot
    exhaust memory and an endless one cannot fill the disk.
  * The pipeline runs as a SUBPROCESS with an explicit environment. A crash or a
    hang cannot take the API down with it, and the job can be killed.
  * Local tool bound to 127.0.0.1. This is not hardened for a shared network and
    should not be exposed on one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RUNS = ROOT / "runs"

ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024      # 500 MB
CHUNK = 1024 * 1024

# Stage markers the pipeline prints, used to drive the progress bar.
STAGE_ORDER = [
    ("Stage 0", "Integrity audit"),
    ("Stage 1", "Cleaning and leak removal"),
    ("Stage 2/3", "Feature engineering"),
    ("Stage 4/5", "Training the ensemble"),
    ("Stage 6", "Graph stage"),
    ("Stage 7/8", "Risk score and SHAP"),
    ("Stage 9", "AML rule layer"),
    ("Stage 11", "Operating metrics"),
    ("Stage 12", "Export bundle"),
    ("Reporting", "Plots"),
]


class UploadRejected(ValueError):
    """The uploaded file failed a safety or format check."""


@dataclass
class Job:
    job_id: str
    original_name: str
    stored_path: Path
    workdir: Path
    status: str = "PENDING"          # PENDING RUNNING DONE FAILED CANCELLED
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    log_path: Path | None = None
    proc: subprocess.Popen | None = None
    error: str | None = None
    target_override: str | None = None

    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _safe_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_stem(name: str) -> str:
    """Keep a readable hint from the client's filename, discard everything else.

    Only used for display and for naming the run folder. The stored file gets a
    generated name regardless, so nothing here can influence where bytes land.
    """
    stem = Path(str(name)).stem
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:40]
    return stem or "dataset"


def accept_upload(filename: str, stream, target: str | None = None) -> Job:
    """Stream an uploaded file to disk under a fresh run directory."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(
            f"'{suffix or 'no extension'}' is not an accepted format. "
            f"Accepted: {', '.join(sorted(ALLOWED_SUFFIXES))}")

    job_id = _safe_job_id()
    stem = _safe_stem(filename)
    workdir = RUNS / f"{stem}_{job_id}"
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generated name. The client's filename never reaches the filesystem.
    stored = data_dir / f"uploaded{suffix}"

    written = 0
    try:
        with open(stored, "wb") as out:
            while True:
                chunk = stream.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise UploadRejected(
                        f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.")
                out.write(chunk)
    except UploadRejected:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(workdir, ignore_errors=True)
        raise UploadRejected(f"Could not save the upload: {exc}") from exc

    if written == 0:
        shutil.rmtree(workdir, ignore_errors=True)
        raise UploadRejected("The uploaded file is empty.")

    job = Job(job_id=job_id, original_name=str(filename)[:120],
              stored_path=stored, workdir=workdir,
              target_override=(target or None))
    job.log_path = workdir / "run.log"
    with _LOCK:
        _JOBS[job_id] = job
    return job


def start(job: Job, fast: bool = True) -> Job:
    """Run the pipeline against the uploaded file, in its own process."""
    env = os.environ.copy()
    env["MULEGUARD_DATA"] = str(job.stored_path)
    env["MULEGUARD_WORKDIR"] = str(job.workdir)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(SRC)
    if fast:
        env["MULEGUARD_FAST"] = "1"
    else:
        env.pop("MULEGUARD_FAST", None)
    if job.target_override:
        env["MULEGUARD_TARGET"] = job.target_override
    # A dictionary belonging to another dataset would be actively misleading.
    env["MULEGUARD_DICT"] = str(job.workdir / "data" / "__none__.xlsx")

    log = open(job.log_path, "w", encoding="utf-8", errors="replace")
    job.proc = subprocess.Popen(
        [sys.executable, str(SRC / "pipeline.py")],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    job.status = "RUNNING"
    job.started_at = time.time()

    def _watch() -> None:
        code = job.proc.wait()
        job.finished_at = time.time()
        if job.status == "CANCELLED":
            pass
        elif code == 0:
            job.status = "DONE"
        else:
            job.status = "FAILED"
            job.error = _last_error(job)
        log.close()

    threading.Thread(target=_watch, daemon=True).start()
    return job


def _last_error(job: Job) -> str:
    """Pull something useful out of the log rather than just an exit code."""
    if not job.log_path or not job.log_path.exists():
        return "The pipeline exited non-zero and wrote no log."
    lines = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if "[ERROR]" in line:
            return line.split("[ERROR]", 1)[-1].strip()
    tail = [l for l in lines if l.strip()][-4:]
    return " / ".join(tail) if tail else "Unknown failure."


def progress(job: Job) -> dict:
    """Stage progress, parsed from the log the pipeline is writing."""
    done, current = [], None
    if job.log_path and job.log_path.exists():
        text = job.log_path.read_text(encoding="utf-8", errors="replace")
        for marker, label in STAGE_ORDER:
            if f"{marker} " in text or f"{marker}—" in text:
                done.append(label)
        current = done[-1] if done else "Starting"
    pct = int(100 * len(done) / len(STAGE_ORDER)) if done else 0
    if job.status == "DONE":
        pct = 100
    return {
        "job_id": job.job_id,
        "status": job.status,
        "original_name": job.original_name,
        "workdir": str(job.workdir.relative_to(ROOT)),
        "elapsed_seconds": round(job.elapsed(), 1),
        "stages_complete": len(done),
        "stages_total": len(STAGE_ORDER),
        "percent": pct,
        "current_stage": current,
        "error": job.error,
    }


def get(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(str(job_id))


def listing(limit: int = 20) -> list[dict]:
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: -j.started_at)[:limit]
    return [progress(j) for j in jobs]


def cancel(job: Job) -> dict:
    if job.proc and job.proc.poll() is None:
        job.status = "CANCELLED"
        job.proc.terminate()
        try:
            job.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            job.proc.kill()
    job.finished_at = time.time()
    return progress(job)


def tail(job: Job, lines: int = 40) -> list[str]:
    if not job.log_path or not job.log_path.exists():
        return []
    text = job.log_path.read_text(encoding="utf-8", errors="replace")
    return [l for l in text.splitlines() if l.strip()][-lines:]


def results(job: Job) -> dict:
    """Headline numbers from a finished run, read from its own workdir."""
    import json

    if job.status != "DONE":
        return {"available": False, "status": job.status}
    rep = job.workdir / "reports"

    def read(name):
        p = rep / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    m, sc, cl, ig = (read("03_metrics.json"), read("05_scoring_report.json"),
                     read("01_clean_report.json"), read("06_integrity_audit.json"))
    out = {"available": True, "workdir": str(job.workdir.relative_to(ROOT)),
           "elapsed_seconds": round(job.elapsed(), 1)}
    if cl:
        out["schema"] = cl.get("schema", {})
        out["leaks_removed"] = cl.get("removed_semantic_leaks", {})
        out["partition_audit"] = cl.get("structural_leak_audit", {})
    if ig:
        out["integrity"] = {
            "contaminated": ig.get("verdict", {}).get("contaminated"),
            "summary": ig.get("verdict", {}).get("summary"),
            "test_a": ig.get("test_A_missingness_only"),
            "test_c": ig.get("test_C_shuffled_labels"),
            "baseline": ig.get("auprc_random_baseline"),
        }
    if m:
        e = m.get("ensemble_precision_first", {})
        out["metrics"] = {k: e.get(k) for k in
                          ("precision", "recall", "auprc", "auroc",
                           "lift_over_prevalence")}
        out["validation"] = m.get("validation", {}).get("scheme")
        # A fitted threshold needs positives to fit on. When the upload is too
        # small to supply them the fitted precision and recall collapse to zero
        # and say nothing about the model, so the review-budget point is sent
        # as the headline instead and the fitted one is marked unusable.
        rb = m.get("review_budget") or {}
        if rb:
            out["review_budget"] = rb
            out["threshold_estimable"] = rb.get(
                "fitted_threshold_is_estimable", True)
            out["positives_per_fit"] = rb.get("positives_per_threshold_fit")
            out["n_mules"] = m.get("n_mules")
    if sc:
        out["bands"] = sc.get("band_stats", {})
    return out
