import { useEffect, useMemo, useState } from "react";
import { fetchClaims, getReportPreviewUrl } from "../api";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]);

function getEvaluationClass(evaluation) {
  const normalized = (evaluation || "").toLowerCase();
  if (normalized === "accept") {
    return "evaluation-accept";
  }
  if (normalized === "reject") {
    return "evaluation-reject";
  }
  if (normalized === "uncertain") {
    return "evaluation-uncertain";
  }
  return "evaluation-pending";
}

function getReportType(filename) {
  const extension = filename?.includes(".") ? `.${filename.split(".").pop().toLowerCase()}` : "";
  if (!extension) {
    return "unknown";
  }
  return IMAGE_EXTENSIONS.has(extension) ? "imaging" : "hematology";
}

function getClaimDate(createdAt) {
  if (!createdAt || typeof createdAt !== "string") {
    return "";
  }
  return createdAt.slice(0, 10);
}

function ClaimsListPage() {
  const [claims, setClaims] = useState([]);
  const [error, setError] = useState("");
  const [selectedReport, setSelectedReport] = useState(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [dateFilter, setDateFilter] = useState("");
  const [claimIdFilter, setClaimIdFilter] = useState("");

  const loadClaims = async () => {
    try {
      setError("");
      const result = await fetchClaims();
      setClaims(result.claims || []);
    } catch (loadError) {
      setError(loadError.message);
    }
  };

  useEffect(() => {
    loadClaims();
  }, []);

  const filteredClaims = useMemo(() => {
    const normalizedClaimId = claimIdFilter.trim().toLowerCase();
    return claims.filter((claim) => {
      const matchesClaimId =
        !normalizedClaimId || claim.submission_id.toLowerCase().includes(normalizedClaimId);
      const matchesDate = !dateFilter || getClaimDate(claim.created_at) === dateFilter;
      const matchesStatus = statusFilter === "all" || claim.status.toLowerCase() === statusFilter;
      return matchesClaimId && matchesDate && matchesStatus;
    });
  }, [claims, claimIdFilter, dateFilter, statusFilter]);

  const availableStatuses = useMemo(() => {
    const unique = new Set(
      claims
        .map((claim) => (claim.status || "").trim())
        .filter(Boolean)
        .map((status) => status.toLowerCase())
    );
    return Array.from(unique).sort();
  }, [claims]);

  useEffect(() => {
    if (!selectedReport) {
      return;
    }
    const selectedClaim = filteredClaims.find((claim) => claim.submission_id === selectedReport.submissionId);
    const reportStillExists = selectedClaim?.reports?.some(
      (report) => report.filename === selectedReport.filename
    );
    if (!selectedClaim || !reportStillExists) {
      setSelectedReport(null);
    }
  }, [filteredClaims, selectedReport]);

  const clearFilters = () => {
    setStatusFilter("all");
    setDateFilter("");
    setClaimIdFilter("");
  };

  const hasActiveFilters = statusFilter !== "all" || Boolean(dateFilter) || Boolean(claimIdFilter.trim());

  return (
    <section>
      <div className="page-header">
        <h2>All Claims</h2>
        <button onClick={loadClaims} type="button">
          Refresh
        </button>
      </div>
      {error && <p className="status error">Failed to load claims: {error}</p>}
      <div className="claims-filters card">
        <label className="claims-filter-field">
          Status
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
            <option value="all">All</option>
            {availableStatuses.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </label>
        <label className="claims-filter-field">
          Date
          <input type="date" value={dateFilter} onChange={(event) => setDateFilter(event.target.value)} />
        </label>
        <label className="claims-filter-field">
          Claim ID
          <input
            type="text"
            value={claimIdFilter}
            onChange={(event) => setClaimIdFilter(event.target.value)}
            placeholder="e.g. CLM-2026..."
          />
        </label>
        <button className="secondary-button" disabled={!hasActiveFilters} onClick={clearFilters} type="button">
          Clear Filters
        </button>
      </div>
      <p className="claims-filter-summary">
        Showing {filteredClaims.length} of {claims.length} claims
      </p>
      <div className={selectedReport ? "claims-layout split" : "claims-layout full"}>
        <div className="claims-list">
          {filteredClaims.map((claim) => {
            const visibleReports = claim.reports || [];
            return (
              <article className="card" key={claim.submission_id}>
                <p>
                  <strong>Submission ID:</strong> {claim.submission_id}
                </p>
                <p>
                  <strong>Status:</strong> <span className={`badge ${claim.status}`}>{claim.status}</span>
                </p>
                <p>
                  <strong>Comments:</strong> {claim.comments || "None"}
                </p>
                <p>
                  <strong>Reports:</strong> {visibleReports.length}
                </p>
                {visibleReports.map((report) => (
                  <div className="report-block" key={`${claim.submission_id}-${report.filename}`}>
                    <div className="report-evaluation-card">
                      <div className="report-evaluation-row">
                        <span className="report-evaluation-label">Report</span>
                        <button
                          className="report-link-button"
                          onClick={() =>
                            setSelectedReport({
                              submissionId: claim.submission_id,
                              filename: report.filename,
                              url: getReportPreviewUrl(claim.submission_id, report.filename)
                            })
                          }
                          type="button"
                        >
                          {report.filename}
                        </button>
                      </div>
                      <div className="report-evaluation-row">
                        <span className="report-evaluation-label">Report Type</span>
                        <span className="report-type-chip">{getReportType(report.filename)}</span>
                      </div>
                      <div className="report-evaluation-row">
                        <span className="report-evaluation-label">Report Evaluation</span>
                        <span className={`report-evaluation-pill ${getEvaluationClass(report.report_evaluation)}`}>
                          {report.report_evaluation || "pending"}
                        </span>
                      </div>
                      <div className="report-evaluation-explanation">
                        <span className="report-evaluation-label">Explanation</span>
                        <p className="explanation-text report-evaluation-explanation-text">
                          {report.explanation || "Pending"}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
                {claim.practitioner_comment && (
                  <p>
                    <strong>Practitioner Comment:</strong> {claim.practitioner_comment}
                  </p>
                )}
              </article>
            );
          })}
          {!filteredClaims.length && <p>No claims match the selected filters.</p>}
        </div>

        {selectedReport && (
          <aside className="report-preview-panel">
            <div className="report-preview-header">
              <h3>Report Preview</h3>
              <button onClick={() => setSelectedReport(null)} type="button">
                Close
              </button>
            </div>
            <p className="preview-meta">
              <strong>Claim:</strong> {selectedReport.submissionId}
            </p>
            <p className="preview-meta">
              <strong>File:</strong> {selectedReport.filename}
            </p>
            <iframe className="report-preview-iframe" src={selectedReport.url} title={selectedReport.filename} />
          </aside>
        )}
      </div>
    </section>
  );
}

export default ClaimsListPage;
