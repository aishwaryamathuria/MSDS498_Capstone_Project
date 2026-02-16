import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  fetchClaimById,
  fetchUncertainClaims,
  getReportPreviewUrl,
  updatePractitionerReview
} from "../api";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]);

function isImageReport(filename) {
  const extension = filename?.includes(".") ? `.${filename.split(".").pop().toLowerCase()}` : "";
  return Boolean(extension) && IMAGE_EXTENSIONS.has(extension);
}

function formatDate(dateString) {
  if (!dateString) {
    return "-";
  }
  const parsed = new Date(dateString);
  if (Number.isNaN(parsed.getTime())) {
    return dateString;
  }
  return parsed.toLocaleString();
}

function PractitionerListView() {
  const navigate = useNavigate();
  const [claims, setClaims] = useState([]);
  const [error, setError] = useState("");

  const loadClaims = async () => {
    try {
      setError("");
      const result = await fetchUncertainClaims();
      setClaims(result.claims || []);
    } catch (loadError) {
      setError(loadError.message);
    }
  };

  useEffect(() => {
    loadClaims();
  }, []);

  return (
    <section>
      <div className="page-header">
        <h2>Practitioner View (Uncertain Claims)</h2>
        <button onClick={loadClaims} type="button">
          Refresh
        </button>
      </div>
      {error && <p className="status error">Failed to load claims: {error}</p>}
      <div className="card practitioner-list-header">
        <span>Claim ID</span>
        <span>Submission Date</span>
        <span>Status (click to open)</span>
      </div>
      <div className="claims-list">
        {claims.map((claim) => (
          <article className="card practitioner-list-item" key={claim.submission_id}>
            <button
              className="practitioner-list-row"
              onClick={() => navigate(`/practitioner/${claim.submission_id}`)}
              type="button"
            >
              <span>{claim.submission_id}</span>
              <span>{formatDate(claim.created_at)}</span>
              <span>
                <span className={`badge ${claim.status}`}>{claim.status}</span>
              </span>
            </button>
            <p className="practitioner-reports-line">
              <strong>Reports:</strong>{" "}
              {claim.reports?.length
                ? claim.reports.map((report) => report.filename).join(", ")
                : "No reports uploaded"}
            </p>
          </article>
        ))}
        {!claims.length && <p>No claims available.</p>}
      </div>
    </section>
  );
}

function PractitionerDetailView({ submissionId }) {
  const [claim, setClaim] = useState(null);
  const [selectedReportName, setSelectedReportName] = useState("");
  const [status, setStatus] = useState("accept");
  const [comment, setComment] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const loadClaim = async () => {
    try {
      setError("");
      const result = await fetchClaimById(submissionId);
      if (result.error) {
        setError(result.error);
        setClaim(null);
        return;
      }
      setClaim(result.claim);
      setStatus(result.claim.status === "uncertain" ? "accept" : result.claim.status);
      setComment(result.claim.practitioner_comment || "");
      if (!selectedReportName && result.claim.reports?.length) {
        setSelectedReportName(result.claim.reports[0].filename);
      }
    } catch (loadError) {
      setError(loadError.message);
    }
  };

  useEffect(() => {
    loadClaim();
  }, [submissionId]);

  const selectedReport = useMemo(() => {
    if (!claim?.reports?.length || !selectedReportName) {
      return null;
    }
    return claim.reports.find((report) => report.filename === selectedReportName) || null;
  }, [claim, selectedReportName]);

  const onSubmit = async (event) => {
    event.preventDefault();
    setMessage("");
    const result = await updatePractitionerReview(submissionId, { status, comment });
    if (result.error) {
      setMessage(result.error);
      return;
    }
    setMessage(`Updated to ${result.status}.`);
    await loadClaim();
  };

  if (error) {
    return (
      <section>
        <div className="page-header">
          <h2>Claim Details</h2>
          <Link className="report-link-button" to="/practitioner">
            Back to list
          </Link>
        </div>
        <p className="status error">{error}</p>
      </section>
    );
  }

  if (!claim) {
    return (
      <section>
        <div className="page-header">
          <h2>Claim Details</h2>
          <Link className="report-link-button" to="/practitioner">
            Back to list
          </Link>
        </div>
        <p>Loading claim details...</p>
      </section>
    );
  }

  return (
    <section>
      <div className="page-header">
        <h2>Claim Details</h2>
        <Link className="report-link-button" to="/practitioner">
          Back to list
        </Link>
      </div>
      <div className="claims-layout split">
        <div className="claims-list">
          <article className="card">
            <p>
              <strong>Claim ID:</strong> {claim.submission_id}
            </p>
            <p>
              <strong>Submission Date:</strong> {formatDate(claim.created_at)}
            </p>
            <p>
              <strong>Status:</strong> <span className={`badge ${claim.status}`}>{claim.status}</span>
            </p>
            <p>
              <strong>User Comments:</strong> {claim.comments || "None"}
            </p>
          </article>

          <article className="card">
            <p>
              <strong>Reports</strong>
            </p>
            <ul className="inline-list">
              {claim.reports?.map((report) => (
                <li key={`${claim.submission_id}-${report.filename}`}>
                  <button
                    className="report-link-button"
                    onClick={() => setSelectedReportName(report.filename)}
                    type="button"
                  >
                    {report.filename}
                  </button>
                </li>
              ))}
            </ul>
          </article>

          <article className="card">
            {claim.status !== "uncertain" ? (
              <p className="status">This claim is not uncertain. Practitioner update is disabled.</p>
            ) : (
              <form className="form-grid" onSubmit={onSubmit}>
                <label>
                  Final Decision
                  <select value={status} onChange={(event) => setStatus(event.target.value)}>
                    <option value="accept">accept</option>
                    <option value="reject">reject</option>
                    <option value="uncertain">uncertain</option>
                  </select>
                </label>
                <label>
                  Practitioner Comment
                  <textarea
                    rows={3}
                    value={comment}
                    onChange={(event) => setComment(event.target.value)}
                    placeholder="Add final notes..."
                  />
                </label>
                <button type="submit">Save Review</button>
              </form>
            )}
            {message && <p className="status">{message}</p>}
          </article>
        </div>

        <aside className="report-preview-panel">
          <div className="report-preview-header">
            <h3>Report Preview</h3>
          </div>
          {!selectedReport ? (
            <p className="preview-placeholder">Select a report to preview.</p>
          ) : (
            <>
              <p className="preview-meta">
                <strong>File:</strong> {selectedReport.filename}
              </p>
              {isImageReport(selectedReport.filename) ? (
                <img
                  className="report-preview-image"
                  src={getReportPreviewUrl(claim.submission_id, selectedReport.filename)}
                  alt={`Preview of ${selectedReport.filename}`}
                />
              ) : (
                <iframe
                  className="report-preview-iframe"
                  src={getReportPreviewUrl(claim.submission_id, selectedReport.filename)}
                  title={selectedReport.filename}
                />
              )}
              <p className="explanation-text">
                <strong>Explanation:</strong> {selectedReport.explanation || "Pending"}
              </p>
            </>
          )}
        </aside>
      </div>
    </section>
  );
}

function PractitionerPage() {
  const { submissionId } = useParams();
  if (submissionId) {
    return <PractitionerDetailView submissionId={submissionId} />;
  }
  return <PractitionerListView />;
}

export default PractitionerPage;
