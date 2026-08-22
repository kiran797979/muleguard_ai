"""
MuleGuard AI — HTTP API and static host for the command-center UI.

Run:
    python -m uvicorn app.server:app --reload
    -> http://127.0.0.1:8000

Every endpoint returns real pipeline output or an explicit error. Nothing here
synthesises a number when an artefact is missing; a 503 with the name of the
stage to run is the correct answer, because a demo that invents plausible
metrics is the single worst failure mode this project could have.

Security posture (this is a local analyst tool, not a public service):
  * Binds to 127.0.0.1 by default; CORS is not enabled at all, because the API
    and the UI are served from the same origin.
  * No path, filename, or module name is ever taken from a request. The model is
    loaded from one fixed location, so untrusted pickles cannot be introduced
    through the API.
  * Request bodies are size-capped and schema-validated before reaching numpy.
  * Errors return a message and no traceback; the traceback goes to the log.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import jobs, service
from .service import ArtefactMissing, jsonable

log = logging.getLogger("muleguard")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

MAX_BODY_BYTES = 256 * 1024

app = FastAPI(
    title="MuleGuard AI",
    description="Network-aware money-mule detection with a dataset integrity audit.",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


# --------------------------------------------------------------------------
# Error handling — never leak a traceback, always say what to do
# --------------------------------------------------------------------------
@app.exception_handler(ArtefactMissing)
async def _artefact_missing(_: Request, exc: ArtefactMissing) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "ARTEFACT_MISSING",
            "detail": str(exc),
            "artefact": exc.what,
            "produced_by": exc.stage,
            "fix": "Run  .\\run.ps1  (Windows) or  ./run.sh  (macOS/Linux) to "
                   "regenerate every pipeline artefact.",
        },
    )


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL", "detail": type(exc).__name__,
                 "fix": "Check the server log. GET /api/health reports which "
                        "pipeline artefacts are present."},
    )


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    """Normalise FastAPI's {detail: ...} into the shape the UI renders."""
    return JSONResponse(status_code=exc.status_code,
                        content={"error": f"HTTP_{exc.status_code}",
                                 "detail": exc.detail if isinstance(exc.detail, str)
                                           else "Invalid request."})


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn pydantic's nested error list into one readable sentence."""
    parts = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(x) for x in e.get("loc", []) if x not in ("body", "query"))
        parts.append(f"{loc or 'input'}: {e.get('msg', 'invalid')}")
    return JSONResponse(status_code=422,
                        content={"error": "INVALID_INPUT", "detail": "; ".join(parts),
                                 "fix": "Feature values must be numbers. See /api/docs."})


@app.middleware("http")
async def _cap_body(request: Request, call_next):
    """Reject oversized bodies before they are parsed."""
    length = request.headers.get("content-length")
    # Uploads have their own, much larger cap enforced while streaming to disk.
    limit = jobs.MAX_UPLOAD_BYTES if request.url.path.startswith(
        "/api/jobs/upload") else MAX_BODY_BYTES
    if length and length.isdigit() and int(length) > limit:
        return JSONResponse(status_code=413,
                            content={"error": "BODY_TOO_LARGE",
                                     "detail": f"limit {limit} bytes"})
    return await call_next(request)


# --------------------------------------------------------------------------
# Read-only panels
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Never raises — this is what the UI polls to decide what it can show."""
    try:
        return jsonable(service.health())
    except Exception as exc:  # noqa: BLE001
        log.exception("health check failed")
        return {"status": "DOWN", "model_loaded": False,
                "model_error": type(exc).__name__, "artefacts": [], "missing": []}


@app.get("/api/overview")
def overview() -> dict:
    return jsonable(service.overview())


@app.get("/api/integrity")
def integrity() -> dict:
    return jsonable(service.integrity())


@app.get("/api/clean")
def clean() -> dict:
    """The Stage 1 report verbatim — encoding, imputation and drop ledgers."""
    return jsonable(service.clean_report())


@app.get("/api/schema")
def schema() -> dict:
    """How this dataset was interpreted — all of it discovered, none configured."""
    return jsonable(service.schema_report())


@app.get("/api/leakage")
def leakage() -> dict:
    return jsonable(service.leakage_defence())


@app.get("/api/features")
def features() -> dict:
    return jsonable(service.mule_features())


@app.get("/api/models")
def models() -> dict:
    return jsonable(service.model_comparison())


@app.get("/api/metrics")
def metrics() -> dict:
    return jsonable(service.metrics())


@app.get("/api/shap")
def shap_global() -> dict:
    return jsonable(service.shap_global())


@app.get("/api/rules")
def rules() -> dict:
    """The deterministic AML rule layer, measured against the base rate."""
    return jsonable(service.rules_report())


@app.get("/api/ablation")
def ablation() -> dict:
    """How much of the score survives removing the behavioural features."""
    return jsonable(service.ablation_report())


@app.get("/api/operating")
def operating() -> dict:
    """Precision@K, investigator load, review budget, extract drift."""
    return jsonable(service.operating_metrics())


@app.get("/api/decisions")
def decisions(limit: int = 100) -> dict:
    """The investigator audit trail."""
    return jsonable(service.decision_log(max(1, min(int(limit), 500))))


@app.get("/api/bands")
def bands() -> dict:
    return jsonable(service.scoring_report())


# --------------------------------------------------------------------------
# Accounts
# --------------------------------------------------------------------------
@app.get("/api/accounts")
def accounts(band: str | None = None, limit: int = 50, offset: int = 0,
             mules_only: bool = False) -> dict:
    if band is not None and band.upper() not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(400, "band must be LOW, MEDIUM or HIGH")
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    return jsonable(service.list_accounts(band, limit, offset, bool(mules_only)))


@app.get("/api/account/{idx}")
def account(idx: int) -> dict:
    try:
        return jsonable(service.analyse_account(int(idx)))
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc


class ScoreRequest(BaseModel):
    features: dict[str, float | int | None] = Field(
        default_factory=dict,
        description="Feature values keyed by F-code (F2506) or by the real "
                    "banking variable name (UPI_AMT_L7D). Anything you omit is "
                    "imputed with the median learned during training.",
    )


class DecisionRequest(BaseModel):
    account_idx: int = Field(..., ge=0, description="Row index of the account reviewed")
    decision: str = Field(..., description="CONFIRMED_MULE, DISMISSED or NEEDS_REVIEW")
    note: str = Field("", max_length=500, description="Free-text investigator note")
    actor: str = Field("demo-analyst", max_length=64)


@app.post("/api/decision")
def decision(req: DecisionRequest) -> dict:
    """Record an investigator decision into the append-only audit trail.

    This is the feedback loop: in a deployment these rows become the labels the
    next model retrains on. Writes to a fixed local file; no path, filename or
    column comes from the request.
    """
    try:
        return jsonable(service.record_decision(
            int(req.account_idx), req.decision, req.note, req.actor))
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/score")
def score(req: ScoreRequest) -> dict:
    """Score an account supplied at runtime with the deployed ensemble."""
    if len(req.features) > 5000:
        raise HTTPException(400, "too many features (limit 5000)")
    clean = {k: v for k, v in req.features.items() if v is not None}
    try:
        return jsonable(service.score_features(clean))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------
# EFRMS / AML integration exports
# --------------------------------------------------------------------------
@app.get("/api/export/alerts")
def export_alerts(min_band: str = "MEDIUM", limit: int = 500,
                  format: str = "json"):
    """Alerts in the documented vendor-neutral schema.

    `format=csv` returns a delimited file, which is how most bank systems
    actually exchange data. This is NOT certified against any named EFRMS; see
    the compatibility_statement in the payload.
    """
    import sys
    from pathlib import Path as _P
    src = _P(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import integration

    if min_band.upper() not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(400, "min_band must be LOW, MEDIUM or HIGH")
    try:
        payload = integration.build_alerts(min_band, max(1, min(int(limit), 5000)))
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    if format.lower() == "csv":
        from fastapi.responses import PlainTextResponse
        csv = integration.alerts_to_dataframe(payload).to_csv(index=False)
        return PlainTextResponse(csv, media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="muleguard_alerts.csv"'})
    return jsonable(payload)


@app.get("/api/export/casepack/{idx}")
def export_casepack(idx: int) -> dict:
    """The full investigator case pack for one alert."""
    import sys
    from pathlib import Path as _P
    src = _P(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import integration
    try:
        return jsonable(integration.case_pack(int(idx)))
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/export/contract")
def export_contract() -> dict:
    """The integration contract: field mapping, taxonomy, and what we do NOT claim."""
    import sys
    from pathlib import Path as _P
    src = _P(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import integration
    return jsonable({
        "schema_version": integration.SCHEMA_VERSION,
        "field_mapping": integration.FIELD_MAP,
        "typology_taxonomy": integration.TYPOLOGY,
        "formats": ["JSON", "CSV", "OpenAPI 3 at /api/openapi.json"],
        "compatibility_statement":
            "Vendor-neutral export in a documented schema. NOT certified against "
            "Oracle FCCM, SAS, NICE Actimize, Clari5, Amlock or any other named "
            "platform; no such testing has been performed. Integration is a "
            "field-mapping exercise, not a rebuild.",
        "what_a_named_platform_would_need": [
            "That platform's integration specification.",
            "A scenario catalogue to map scenario_codes onto.",
            "An agreed entity key; this dataset is anonymised.",
            "A test environment to validate ingestion against.",
        ],
    })


# --------------------------------------------------------------------------
# Upload a dataset and run against it
# --------------------------------------------------------------------------
@app.post("/api/jobs/upload")
async def upload_dataset(file: UploadFile = File(...),
                         target: str = Form(""),
                         fast: bool = Form(True),
                         mode: str = Form("auto")) -> dict:
    """Accept a dataset from the browser and start the pipeline against it.

    The client's filename is used only for display; the stored file gets a
    generated name in a fresh run directory, so nothing in the request can
    influence where bytes are written. See app/jobs.py for the full posture.
    """
    try:
        job = jobs.accept_upload(file.filename or "dataset.csv", file.file,
                                 target.strip() or None)
    except jobs.UploadRejected as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await file.close()

    # Nobody handing over a file will say whether it is labelled, and it is not
    # their job to. In "auto" the system reads the schema and decides: a label
    # means it can retrain and MEASURE, no label means it can still DETECT with
    # the deployed model. Failing and asking the operator to pick a button was
    # the wrong answer to a question the system can answer itself.
    chosen = str(mode).lower()
    detected = None
    if chosen == "auto":
        detected = service.detect_target(job.stored_path)
        chosen = "train" if detected["labelled"] else "score"
        log.info("auto mode: %s -> %s", job.original_name, chosen)

    # score does NOT retrain: it applies the deployed model to an unlabelled
    # extract. Training needs a target column; detection does not.
    if chosen == "score":
        try:
            out = service.score_file(job.stored_path)
            out["decided"] = ("No label column found in this file, so it was "
                              "scored with the deployed model rather than "
                              "retrained. Detection needs no labels."
                              ) if detected else "Scoring was requested explicitly."
            out["target_search"] = detected
            # This job never enters the pipeline, so leave no PENDING ghost
            # sitting in the run list pretending work is queued.
            job.status = "SCORED"
            job.finished_at = time.time()
            return jsonable({**out, "job_id": job.job_id})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("score-only run failed")
            raise HTTPException(500, f"Scoring failed: {exc}") from exc

    jobs.start(job, fast=bool(fast))
    log.info("started job %s for %s", job.job_id, job.original_name)
    return jsonable({**jobs.progress(job), "target_search": detected})


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> dict:
    return jsonable({"jobs": jobs.listing(max(1, min(int(limit), 100)))})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job. It may have expired with a restart.")
    return jsonable({**jobs.progress(job), "log_tail": jobs.tail(job, 25)})


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return jsonable(jobs.results(job))


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return jsonable(jobs.cancel(job))


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------
class _NoCacheStatic(StaticFiles):
    """Serve the UI with caching disabled.

    A browser holding an old app.js against a new index.html produces symptoms
    that look like backend bugs: buttons that exist but do nothing, or that do
    the wrong thing because the handler wiring them up is missing. On a local
    single-operator server there is nothing to gain from caching.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp


if STATIC.exists():
    app.mount("/static", _NoCacheStatic(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(500, "UI not found — app/static/index.html is missing")
    return FileResponse(page, headers={"Cache-Control": "no-store, must-revalidate"})
