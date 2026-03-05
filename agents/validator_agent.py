import json
import logging
import os

try:
    import litellm
except ImportError:
    litellm = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


def _normalize_decision(value):
    if value is None:
        return "uncertain"
    value = str(value).strip().lower()

    if value in {"accept", "approved", "approve", "true", "positive"}:
        return "accept"
    if value in {"reject", "rejected", "false", "negative"}:
        return "reject"
    return "uncertain"


def _deterministic_validator(imaging_result=None, hematology_result=None):
    imaging_decision = _normalize_decision(
        imaging_result.get("decision") if imaging_result else None
    )
    hema_decision = _normalize_decision(
        hematology_result.get("decision") if hematology_result else None
    )

    # If one side is missing, use the other side conservatively
    if imaging_result and not hematology_result:
        final_decision = imaging_decision
        decision_rule = "imaging_only"
    elif hematology_result and not imaging_result:
        final_decision = hema_decision
        decision_rule = "hematology_only"
    else:
        if imaging_decision == "uncertain" or hema_decision == "uncertain":
            final_decision = "uncertain"
            decision_rule = "uncertain_input"
        elif imaging_decision == "accept" and hema_decision == "accept":
            final_decision = "accept"
            decision_rule = "both_accept"
        elif imaging_decision == "reject" and hema_decision == "reject":
            final_decision = "reject"
            decision_rule = "both_reject"
        else:
            final_decision = "uncertain"
            decision_rule = "agent_mismatch"

    parts = []

    if imaging_result:
        band = imaging_result.get("band_label")
        decile = imaging_result.get("decile")
        prob = imaging_result.get("probability")
        expl = imaging_result.get("explanation", "")
        prob_text = f"{prob:.4f}" if isinstance(prob, (int, float)) else "NA"
        parts.append(
            f"Imaging agent decision={imaging_decision}, band={band}, decile={decile}, probability={prob_text}. {expl}"
        )

    if hematology_result:
        verdict = hematology_result.get("verdict")
        expl = hematology_result.get("explanation", "")
        parts.append(
            f"Hematology agent decision={hema_decision}, verdict={verdict}. {expl}"
        )

    if decision_rule == "both_accept":
        rationale = "Both agents support pneumonia-related evidence."
    elif decision_rule == "both_reject":
        rationale = "Both agents do not support pneumonia-related evidence."
    elif decision_rule == "agent_mismatch":
        rationale = "The agents disagree, so the case should remain uncertain for review."
    elif decision_rule == "imaging_only":
        rationale = "Only imaging evidence is available, so the decision follows the imaging agent."
    elif decision_rule == "hematology_only":
        rationale = "Only hematology evidence is available, so the decision follows the hematology agent."
    else:
        rationale = "At least one agent is uncertain, so the case should remain uncertain for review."

    parts.append(f"Validator rationale: {rationale}")

    return {
        "decision": final_decision,
        "decision_rule": decision_rule,
        "explanation": " ".join(parts),
        "source": "deterministic",
    }


def _build_validator_prompt(imaging_result=None, hematology_result=None):
    imaging_decision = _normalize_decision(
        imaging_result.get("decision") if imaging_result else None
    )
    hema_decision = _normalize_decision(
        hematology_result.get("decision") if hematology_result else None
    )

    payload = {
        "imaging_result": {
            "decision": imaging_decision,
            "band_label": imaging_result.get("band_label") if imaging_result else None,
            "decile": imaging_result.get("decile") if imaging_result else None,
            "probability": imaging_result.get("probability") if imaging_result else None,
            "explanation": imaging_result.get("explanation") if imaging_result else None,
        } if imaging_result else None,
        "hematology_result": {
            "decision": hema_decision,
            "verdict": hematology_result.get("verdict") if hematology_result else None,
            "values": hematology_result.get("values") if hematology_result else None,
            "explanation": hematology_result.get("explanation") if hematology_result else None,
        } if hematology_result else None,
    }

    return f"""
You are a validator agent for an AI-assisted insurance claim review workflow.

Your job is to combine the structured outputs from both agents:
1. an imaging agent
2. a hematology agent

Allowed final decisions:
- accept
- reject
- uncertain

Rules:
- If all agents support pneumonia-related evidence -> accept
- If all agents do not support pneumonia-related evidence -> reject
- If any agent is is uncertain -> uncertain
- If both agents are present and disagree -> uncertain

Return ONLY valid JSON in this exact schema:
{{
  "decision": "accept|reject|uncertain",
  "decision_rule": "short_rule_name",
  "explanation": "brief plain-English rationale"
}}

Structured agent outputs:
{json.dumps(payload, indent=2)}
""".strip()


def _llm_validator(imaging_result=None, hematology_result=None):
    if litellm is None:
        raise RuntimeError("litellm is not installed")

    model = os.environ.get("OLLAMA_MODEL", "phi3:latest")
    model_id = f"ollama/{model}"
    prompt = _build_validator_prompt(imaging_result=imaging_result, hematology_result=hematology_result)

    out = litellm.completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )

    text = (out.choices[0].message.content or "").strip()
    logger.info("RAW LLM OUTPUT: %s", text)
    print(text)
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Validator LLM did not return parseable JSON: {text}")
        parsed = json.loads(text[start:end + 1])

    decision = _normalize_decision(parsed.get("decision"))
    decision_rule = parsed.get("decision_rule", "llm_rule")
    explanation = parsed.get("explanation", "Validator provided no explanation.")

    return {
        "decision": decision,
        "decision_rule": decision_rule,
        "explanation": explanation,
        "source": "llm",
    }


def validate_claim(imaging_result=None, hematology_result=None):
    """
    Main validator entry point.
    Uses litellm if available; falls back to deterministic rules if LLM fails.
    """
    try:
        if litellm is not None:
            logger.info("Validator agent using litellm.")
            return _llm_validator(
                imaging_result=imaging_result,
                hematology_result=hematology_result,
            )
    except Exception as exc:
        logger.warning("Validator LLM failed, falling back to deterministic logic: %s", exc)

    logger.info("Validator agent using deterministic fallback.")
    return _deterministic_validator(
        imaging_result=imaging_result,
        hematology_result=hematology_result,
    )