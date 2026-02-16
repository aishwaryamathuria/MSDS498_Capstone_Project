import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator import run_patient_workflow  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SUBMISSIONS_DIR = DATA_DIR / "submissions"
DB_PATH = DATA_DIR / "claims_db.json"
POLL_INTERVAL_SECONDS = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}

db_lock = threading.Lock()

app = Flask(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text(json.dumps({"claims": []}, indent=2), encoding="utf-8")


def load_db() -> dict:
    ensure_storage()
    with db_lock:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))


def save_db(db_data: dict) -> None:
    with db_lock:
        DB_PATH.write_text(json.dumps(db_data, indent=2), encoding="utf-8")


def generate_submission_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"CLM-{ts}-{suffix}"


def run_orchestrator_for_claim(claim: dict) -> dict:
    image_path = None
    hematology_path = None
    for report in claim.get("reports", []):
        report_path = Path(report["stored_path"])
        if report_path.suffix.lower() in IMAGE_EXTENSIONS:
            if image_path is None:
                image_path = str(report_path)
        else:
            if hematology_path is None:
                hematology_path = str(report_path)

    return run_patient_workflow(
        image_path=image_path,
        hematology_report_path=hematology_path,
    )


def process_submission(submission_id: str) -> None:
    db_data = load_db()
    claim = next((c for c in db_data["claims"] if c["submission_id"] == submission_id), None)
    if not claim:
        return
    if claim["status"] != "Under Review":
        return

    try:
        orchestrator_result = run_orchestrator_for_claim(claim)
    except Exception as exc:  # pragma: no cover - defensive fallback
        orchestrator_result = {
            "status": "uncertain",
            "agent_results": [],
            "message": f"Orchestrator failed: {exc}",
        }

    agent_result_by_name = {
        item.get("agent"): item for item in orchestrator_result.get("agent_results", [])
    }
    for report in claim["reports"]:
        report_path = Path(report["stored_path"])
        agent_name = "imaging" if report_path.suffix.lower() in IMAGE_EXTENSIONS else "hematology"
        agent_result = agent_result_by_name.get(agent_name)
        if agent_result:
            report["explanation"] = agent_result.get("explanation", "")
            report["report_evaluation"] = agent_result.get("decision", "uncertain")
        else:
            report["explanation"] = (
                f"No {agent_name} agent result produced for this report in current submission."
            )
            report["report_evaluation"] = "uncertain"

    claim["status"] = orchestrator_result.get("status", "uncertain")
    claim["final_evaluation"] = claim["status"]
    claim["updated_at"] = now_iso()
    save_db(db_data)


class SubmissionListener(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)

    def run(self) -> None:
        while True:
            db_data = load_db()
            pending = [
                item["submission_id"]
                for item in db_data["claims"]
                if item["status"] == "Under Review"
            ]
            for submission_id in pending:
                process_submission(submission_id)
            time.sleep(POLL_INTERVAL_SECONDS)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PATCH,OPTIONS"
    return response


@app.route("/api/claims", methods=["GET"])
def list_claims():
    db_data = load_db()
    claims_sorted = sorted(db_data["claims"], key=lambda c: c["created_at"], reverse=True)
    return jsonify({"claims": claims_sorted})


@app.route("/api/claims/uncertain", methods=["GET"])
def list_uncertain_claims():
    db_data = load_db()
    uncertain_claims = [claim for claim in db_data["claims"] if claim["status"] == "uncertain"]
    uncertain_claims.sort(key=lambda c: c["created_at"], reverse=True)
    return jsonify({"claims": uncertain_claims})


@app.route("/api/claims/<submission_id>", methods=["GET"])
def get_claim(submission_id: str):
    db_data = load_db()
    claim = next((c for c in db_data["claims"] if c["submission_id"] == submission_id), None)
    if not claim:
        return jsonify({"error": "Claim not found."}), 404
    return jsonify({"claim": claim})


@app.route("/api/claims/<submission_id>/reports/<path:filename>", methods=["GET"])
def get_report_file(submission_id: str, filename: str):
    db_data = load_db()
    claim = next((c for c in db_data["claims"] if c["submission_id"] == submission_id), None)
    if not claim:
        return jsonify({"error": "Claim not found."}), 404

    report = next((r for r in claim["reports"] if r["filename"] == filename), None)
    if not report:
        return jsonify({"error": "Report not found for this claim."}), 404

    report_path = Path(report["stored_path"]).resolve()
    try:
        report_path.relative_to(SUBMISSIONS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid report path."}), 400

    if not report_path.exists():
        return jsonify({"error": "Report file does not exist."}), 404

    return send_file(report_path, as_attachment=False)


@app.route("/api/claims", methods=["POST"])
def submit_claim():
    if "reports" not in request.files:
        return jsonify({"error": "At least one report file is required under 'reports'."}), 400

    files = request.files.getlist("reports")
    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "Please upload at least one report file."}), 400

    comments = request.form.get("comments", "")
    submission_id = generate_submission_id()

    claim_root = SUBMISSIONS_DIR / submission_id
    reports_dir = claim_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    reports_payload = []
    for uploaded_file in files:
        if not uploaded_file.filename:
            continue
        safe_name = Path(uploaded_file.filename).name
        destination = reports_dir / safe_name
        uploaded_file.save(destination)
        reports_payload.append(
            {
                "filename": safe_name,
                "stored_path": str(destination.resolve()),
                "explanation": "",
                "report_evaluation": "pending",
            }
        )

    if not reports_payload:
        return jsonify({"error": "No valid report files received."}), 400

    claim = {
        "submission_id": submission_id,
        "comments": comments,
        "status": "Under Review",
        "final_evaluation": None,
        "practitioner_comment": "",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "reports": reports_payload,
    }

    db_data = load_db()
    db_data["claims"].append(claim)
    save_db(db_data)

    return jsonify({"submission_id": submission_id, "status": claim["status"]}), 201


@app.route("/api/claims/<submission_id>/practitioner-review", methods=["PATCH"])
def practitioner_review(submission_id: str):
    payload = request.get_json(silent=True) or {}
    status = payload.get("status")
    comment = payload.get("comment", "")
    if status not in {"accept", "reject", "uncertain"}:
        return jsonify({"error": "Status must be one of: accept, reject, uncertain."}), 400

    db_data = load_db()
    claim = next((c for c in db_data["claims"] if c["submission_id"] == submission_id), None)
    if not claim:
        return jsonify({"error": "Claim not found."}), 404
    if claim["status"] != "uncertain":
        return jsonify({"error": "Only claims with uncertain status can be updated by practitioner."}), 400

    claim["status"] = status
    claim["final_evaluation"] = status
    claim["practitioner_comment"] = comment
    claim["updated_at"] = now_iso()
    save_db(db_data)
    return jsonify({"submission_id": submission_id, "status": status}), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def start_listener() -> None:
    listener = SubmissionListener()
    listener.start()


if __name__ == "__main__":
    ensure_storage()
    start_listener()
    app.run(host="0.0.0.0", port=8081, debug=False)
