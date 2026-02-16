const API_BASE = "http://localhost:8081/api";

async function parseApiResponse(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Request failed with status ${response.status}`);
    }
    return data;
  }

  const rawText = await response.text();
  const looksLikeHtml = rawText.trim().startsWith("<!doctype") || rawText.trim().startsWith("<html");
  if (looksLikeHtml) {
    throw new Error("API returned HTML instead of JSON. Try restarting the backend.");
  }

  throw new Error(rawText || `Request failed with status ${response.status}`);
}

export async function submitClaim(formData) {
  const response = await fetch(`${API_BASE}/claims`, {
    method: "POST",
    body: formData
  });
  return parseApiResponse(response);
}

export async function fetchClaims() {
  const response = await fetch(`${API_BASE}/claims`);
  return parseApiResponse(response);
}

export async function fetchUncertainClaims() {
  const response = await fetch(`${API_BASE}/claims/uncertain`);
  return parseApiResponse(response);
}

export async function fetchClaimById(submissionId) {
  const response = await fetch(`${API_BASE}/claims/${submissionId}`);
  return parseApiResponse(response);
}

export async function updatePractitionerReview(submissionId, payload) {
  const response = await fetch(`${API_BASE}/claims/${submissionId}/practitioner-review`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  return parseApiResponse(response);
}

export function getReportPreviewUrl(submissionId, filename) {
  const encodedFilename = encodeURIComponent(filename);
  return `${API_BASE}/claims/${submissionId}/reports/${encodedFilename}`;
}
