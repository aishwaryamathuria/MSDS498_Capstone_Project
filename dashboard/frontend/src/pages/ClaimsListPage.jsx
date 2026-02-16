import { useEffect, useState } from "react";
import { fetchClaims, getReportPreviewUrl } from "../api";

function ClaimsListPage() {
  const [claims, setClaims] = useState([]);
  const [error, setError] = useState("");
  const [selectedReport, setSelectedReport] = useState(null);

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

  return (
    <section>
      <div className="page-header">
        <h2>All Claims</h2>
        <button onClick={loadClaims} type="button">
          Refresh
        </button>
      </div>
      {error && <p className="status error">Failed to load claims: {error}</p>}
      <div className={selectedReport ? "claims-layout split" : "claims-layout full"}>
        <div className="claims-list">
          {claims.map((claim) => (
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
                <strong>Reports:</strong> {claim.reports?.length || 0}
              </p>
              {claim.reports?.map((report) => (
                <div className="report-block" key={`${claim.submission_id}-${report.filename}`}>
                  <p>
                    <strong>Report:</strong>{" "}
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
                  </p>
                  <p>
                    <strong>Report Evaluation:</strong> {report.report_evaluation}
                  </p>
                  <p className="explanation-text">
                    <strong>Explanation:</strong> {report.explanation || "Pending"}
                  </p>
                </div>
              ))}
              {claim.practitioner_comment && (
                <p>
                  <strong>Practitioner Comment:</strong> {claim.practitioner_comment}
                </p>
              )}
            </article>
          ))}
          {!claims.length && <p>No claims found yet.</p>}
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
