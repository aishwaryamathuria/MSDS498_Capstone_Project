import os
import re
from pathlib import Path

try:
    import litellm
except ImportError:
    litellm = None

_BASE = None

def _base():
    global _BASE
    if _BASE is None:
        _BASE = Path(__file__).resolve().parent.parent
    return _BASE

def load_rag_examples(n_positive=5, n_negative=5):
    golden_dir = _base() / "dataset" / "hematology_golden"
    positive_dir = golden_dir / "positive"
    negative_dir = golden_dir / "negative"
    positive_texts = []
    negative_texts = []
    if positive_dir.exists():
        files = sorted(positive_dir.glob("*.txt"))[:n_positive]
        for f in files:
            positive_texts.append(f.read_text(encoding="utf-8", errors="replace").strip())
    if negative_dir.exists():
        files = sorted(negative_dir.glob("*.txt"))[:n_negative]
        for f in files:
            negative_texts.append(f.read_text(encoding="utf-8", errors="replace").strip())
    return positive_texts, negative_texts

def _parse_float_from_text(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except (ValueError, IndexError):
        return None

def parse_hematology_report(report_text):
    wbc = _parse_float_from_text(report_text, r"WBC\s+Count:\s*([\d.]+)")
    if wbc is None:
        wbc = _parse_float_from_text(report_text, r"WBC[^:]*:\s*([\d.]+)")
    crp = _parse_float_from_text(report_text, r"CRP\)?:\s*([\d.]+)")
    if crp is None:
        crp = _parse_float_from_text(report_text, r"C-reactive protein[^:]*:\s*([\d.]+)\s*mg")
    neutrophils = _parse_float_from_text(report_text, r"Neutrophils:\s*([\d.]+)\s*%")
    return {"wbc": wbc, "crp": crp, "neutrophils": neutrophils}

def _elevated_from_values(values):
    wbc, crp, neut = values.get("wbc"), values.get("crp"), values.get("neutrophils")
    elevated = []
    if wbc is not None and wbc > 11.0:
        elevated.append("wbc")
    if crp is not None and crp >= 50.0:
        elevated.append("crp")
    if neut is not None and neut > 75.0:
        elevated.append("neutrophils")
    return elevated

# Rule-based verdict from values
def _verdict_from_values(values):
    elevated = _elevated_from_values(values)
    wbc, crp, neut = values.get("wbc"), values.get("crp"), values.get("neutrophils")
    all_normal = (
        (wbc is None or wbc <= 11.0)
        and (crp is None or crp <= 10.0)
        and (neut is None or neut <= 75.0)
    )
    if len(elevated) >= 2:
        return "true"
    if all_normal:
        return "false"
    return "uncertain"


def check_pneumonia_thresholds(values, report_text=None):
    wbc = values.get("wbc")
    crp = values.get("crp")
    neutrophils = values.get("neutrophils")
    elevated = _elevated_from_values(values)
    details = {"wbc": wbc, "crp": crp, "neutrophils": neutrophils, "elevated": elevated}

    positive_examples, negative_examples = load_rag_examples()
    if not positive_examples and not negative_examples:
        return _check_pneumonia_fallback(values)

    if litellm is None:
        return _check_pneumonia_fallback(values)

    rule_verdict = _verdict_from_values(values)
    if rule_verdict in ("true", "false"):
        return rule_verdict, details

    model = os.environ.get("OLLAMA_MODEL", "phi3:latest")
    model_id = f"ollama/{model}"

    def block(label, texts):
        if not texts:
            return ""
        parts = [f"--- {label} ---"]
        for i, t in enumerate(texts, 1):
            parts.append(f"[Example {i}]\n{t}")
        return "\n\n".join(parts)

    positive_block = block("Golden examples labeled POSITIVE (pneumonia)", positive_examples)
    negative_block = block("Golden examples labeled NEGATIVE (no pneumonia)", negative_examples)
    current = (report_text or "").strip()
    if not current:
        current = f"WBC: {wbc}, CRP (mg/L): {crp}, Neutrophils (%): {neutrophils}"

    prompt = f"""You are classifying a hematology report for pneumonia. Use the golden examples below as reference.

{positive_block}

{negative_block}

--- Report to classify ---
{current}

Based on the golden examples, is this report more like POSITIVE (pneumonia) or NEGATIVE (no pneumonia)? Reply with exactly one word: true, false, or uncertain."""

    try:
        out = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        text = (out.choices[0].message.content or "").strip().lower()
        verdict = _parse_verdict(text)
        if verdict and verdict not in ("true", "false", "uncertain"):
            verdict = "uncertain"
        if not verdict:
            verdict = "uncertain"
        return verdict, details
    except Exception:
        return _check_pneumonia_fallback(values)


def _parse_verdict(text):
    if not text:
        return None
    for word in ("true", "false", "uncertain"):
        if word in text.split() or text.startswith(word):
            return word
    return None


def _check_pneumonia_fallback(values):
    wbc, crp, neut = values.get("wbc"), values.get("crp"), values.get("neutrophils")
    details = {"wbc": wbc, "crp": crp, "neutrophils": neut, "elevated": []}

    wbc_elevated = wbc is not None and wbc > 11.0
    crp_high = crp is not None and crp >= 50.0
    crp_mid = crp is not None and 10.0 < crp < 50.0
    crp_normal = crp is not None and crp <= 10.0
    neut_elevated = neut is not None and neut > 75.0

    if wbc_elevated:
        details["elevated"].append("wbc")
    if crp_high:
        details["elevated"].append("crp")
    if crp_mid:
        details["elevated"].append("crp_mid")
    if neut_elevated:
        details["elevated"].append("neutrophils")

    strong = sum([wbc_elevated, crp_high, neut_elevated]) >= 2
    if strong:
        return "true", details
    if crp_normal and not wbc_elevated and not neut_elevated:
        return "false", details
    return "uncertain", details


def _resolve_report_path(report_path):
    path = Path(report_path)
    base = _base()
    dataset_reports = base / "dataset" / "heamatology_reports"
    if not path.is_absolute():
        candidate = dataset_reports / path
        if candidate.exists():
            return candidate
        if (base / path).exists():
            return base / path
        return candidate
    return path

# Turn verdict and values into a short human-readable interpretation.
def _interpretation_text(verdict, values, elevated):
    wbc = values.get("wbc")
    crp = values.get("crp")
    neut = values.get("neutrophils")
    parts = []

    if verdict == "true":
        parts.append("The hematology results suggest that pneumonia is likely.")
        if elevated:
            reasons = []
            if "wbc" in elevated and wbc is not None:
                reasons.append(f"white blood count is elevated ({wbc})")
            if "crp" in elevated and crp is not None:
                reasons.append(f"CRP is markedly raised ({crp} mg/L)")
            if "neutrophils" in elevated and neut is not None:
                reasons.append(f"neutrophils are high ({neut}%)")
            if reasons:
                parts.append(" This is because " + ", ".join(reasons) + ", which together point to a bacterial infection such as pneumonia.")
        else:
            parts.append(" Inflammatory markers and blood counts are consistent with infection.")
    elif verdict == "false":
        parts.append("The hematology results do not suggest pneumonia.")
        parts.append(" White blood count, CRP, and neutrophils are within or near normal limits, with no significant inflammatory pattern to support a bacterial lung infection.")
    else:
        parts.append("The results are inconclusive for pneumonia.")
        if elevated:
            reasons = []
            if "wbc" in elevated and wbc is not None:
                reasons.append(f"WBC is elevated ({wbc})")
            if "crp" in elevated and crp is not None:
                reasons.append(f"CRP is {crp} mg/L")
            if "neutrophils" in elevated and neut is not None:
                reasons.append(f"neutrophils are {neut}%")
            if reasons:
                parts.append(" Some markers are raised (" + ", ".join(reasons) + "), but the overall picture is borderline.")
        else:
            parts.append(" Values are in a grey zone; clinical and imaging correlation are needed to decide.")

    return "".join(parts)

# Run hematology agent and return a human-readable interpretation
def run(report_path=None, report_text=None):
    if report_text is None and report_path is None:
        return "Hematology: No report provided."
    if report_text is None:
        path = _resolve_report_path(report_path)
        if not path.exists():
            return f"Hematology: Report file not found: {report_path}"
        report_text = path.read_text(encoding="utf-8", errors="replace")
    values = parse_hematology_report(report_text)
    verdict, details = check_pneumonia_thresholds(values, report_text=report_text)
    elevated = details.get("elevated", [])
    interpretation = _interpretation_text(verdict, values, elevated)
    return f"Hematology: {interpretation}"
