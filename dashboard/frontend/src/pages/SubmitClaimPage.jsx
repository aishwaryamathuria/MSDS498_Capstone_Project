import { useState } from "react";
import { submitClaim } from "../api";

function SubmitClaimPage() {
  const [comments, setComments] = useState("");
  const [files, setFiles] = useState([]);
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const addFiles = (event) => {
    const nextFiles = Array.from(event.target.files || []);
    if (!nextFiles.length) {
      return;
    }
    setFiles((current) => {
      const seen = new Set(current.map((file) => `${file.name}-${file.size}-${file.lastModified}`));
      const merged = [...current];
      nextFiles.forEach((file) => {
        const key = `${file.name}-${file.size}-${file.lastModified}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(file);
        }
      });
      return merged;
    });
    event.target.value = "";
  };

  const removeFile = (fileToRemove) => {
    setFiles((current) =>
      current.filter(
        (file) =>
          !(
            file.name === fileToRemove.name &&
            file.size === fileToRemove.size &&
            file.lastModified === fileToRemove.lastModified
          )
      )
    );
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!files.length) {
      setMessage("Please upload at least one report.");
      return;
    }
    setIsSubmitting(true);
    setMessage("");
    try {
      const formData = new FormData();
      formData.append("comments", comments);
      files.forEach((file) => formData.append("reports", file));
      const result = await submitClaim(formData);
      if (result.error) {
        setMessage(result.error);
      } else {
        setMessage(`Claim submitted: ${result.submission_id}. Current status: ${result.status}.`);
        setComments("");
        setFiles([]);
      }
    } catch (error) {
      setMessage(`Submission failed: ${error.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section>
      <h2>User Submission</h2>
      <form className="card form-grid" onSubmit={onSubmit}>
        <label>
          Comments
          <textarea
            rows={4}
            value={comments}
            onChange={(event) => setComments(event.target.value)}
            placeholder="Add claim comments..."
          />
        </label>
        <label>
          Reports
          <input type="file" multiple onChange={addFiles} />
        </label>
        {!!files.length && (
          <div className="selected-files">
            <p>
              <strong>Selected reports:</strong> {files.length}
            </p>
            <ul className="inline-list">
              {files.map((file) => (
                <li className="selected-file-item" key={`${file.name}-${file.size}-${file.lastModified}`}>
                  <span>{file.name}</span>
                  <button className="secondary-button" onClick={() => removeFile(file)} type="button">
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        <button disabled={isSubmitting} type="submit">
          {isSubmitting ? "Submitting..." : "Submit Claim"}
        </button>
      </form>
      {message && <p className="status">{message}</p>}
    </section>
  );
}

export default SubmitClaimPage;
