import os
import logging
from pathlib import Path
from PIL import Image
import torch
import transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoProcessor


CLASSIFIER_MODEL_ID = "lxyuan/vit-xray-pneumonia-classification"
EXPLAINER_MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

_MODELS_READY = False
_MODEL_INIT_ERROR = None
_classifier_processor = None
_classifier_model = None
_explainer_processor = None
_explainer_model = None
_ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_env_file():
    env_path = _ROOT_DIR / ".env"
    if not env_path.exists():
        logger.info("No .env found at %s; continuing without file-based env vars.", env_path)
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    logger.info("Loaded environment variables from %s", env_path)


_load_env_file()
_HF_TOKEN = os.getenv("HF_TOKEN")
logger.info("HF token detected: %s", "yes" if _HF_TOKEN else "no")


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


_DEVICE = _resolve_device()
if _DEVICE == "mps":
    # Gracefully fall back to CPU for ops not yet implemented on MPS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _log_device_context() -> None:
    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    logger.info(
        "Imaging runtime device selected: %s (cuda_available=%s, mps_available=%s)",
        _DEVICE,
        cuda_available,
        mps_available,
    )


def _from_pretrained_with_auth(model_cls, model_id, **kwargs):
    if _HF_TOKEN:
        try:
            return model_cls.from_pretrained(model_id, token=_HF_TOKEN, **kwargs)
        except TypeError:
            return model_cls.from_pretrained(
                model_id, use_auth_token=_HF_TOKEN, **kwargs
            )
    return model_cls.from_pretrained(model_id, **kwargs)


def _build_multimodal_prompt(user_text):
    if hasattr(_explainer_processor, "apply_chat_template"):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            return _explainer_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            # Fall back to explicit image token prompt format.
            pass

    return f"<image>\n{user_text}"


def _load_explainer_model(model_id):
    preferred_classes = [
        "AutoModelForVision2Seq",
        "LlavaForConditionalGeneration",
        "LlavaNextForConditionalGeneration",
        "AutoModelForCausalLM",
    ]

    kwargs = {"torch_dtype": torch.float16} if _DEVICE in {"cuda", "mps"} else {}
    last_error = None

    for class_name in preferred_classes:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            logger.info("Trying explainer loader class: %s", class_name)
            return _from_pretrained_with_auth(model_cls, model_id, **kwargs).to(_DEVICE)
        except Exception as exc:
            last_error = exc
            logger.warning("Explainer loader %s failed: %s", class_name, exc)

    raise RuntimeError(
        "Unable to load explainer model with available transformers classes. "
        f"Tried: {', '.join(preferred_classes)}"
    ) from last_error


def initialize_imaging_models():
    # Load classifier + VLM models once during script lifecycle
    global _MODELS_READY
    global _MODEL_INIT_ERROR
    global _classifier_processor, _classifier_model
    global _explainer_processor, _explainer_model

    if _MODELS_READY:
        logger.info("Imaging models already initialized.")
        return
    if _MODEL_INIT_ERROR is not None:
        logger.error("Previous imaging model init failure detected: %s", _MODEL_INIT_ERROR)
        raise RuntimeError(
            f"Imaging model initialization failed earlier: {_MODEL_INIT_ERROR}"
        ) from _MODEL_INIT_ERROR

    _log_device_context()
    logger.info("Initializing imaging models on device: %s", _DEVICE)
    _classifier_processor = _from_pretrained_with_auth(
        AutoImageProcessor, CLASSIFIER_MODEL_ID
    )
    _classifier_model = _from_pretrained_with_auth(
        AutoModelForImageClassification, CLASSIFIER_MODEL_ID
    ).to(_DEVICE)
    _classifier_model.eval()

    _explainer_processor = _from_pretrained_with_auth(AutoProcessor, EXPLAINER_MODEL_ID)
    _explainer_model = _load_explainer_model(EXPLAINER_MODEL_ID)
    _explainer_model.eval()

    _MODELS_READY = True
    logger.info("Imaging models initialized successfully.")


def classify_pneumonia(image_path):
    # return pneumonia flag and probability
    initialize_imaging_models()
    logger.info("Running pneumonia classification for image: %s", image_path)
    image = Image.open(image_path).convert("RGB")

    inputs = _classifier_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _classifier_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = (
            _classifier_model.config.id2label.get(predicted_idx, str(predicted_idx))
            .lower()
            .strip()
        )
        pneumonia_probability = float(probs[predicted_idx].item())
        pneumonia_positive = "pneumonia" in predicted_label or predicted_label in (
            "1",
            "positive",
            "yes",
            "true",
        )

    return {
        "pneumonia_positive": pneumonia_positive,
        "probability": pneumonia_probability,
        "predicted_label": predicted_label,
    }


def _log_classification_result(result):
    status = "positive" if result["pneumonia_positive"] else "negative"
    logger.info(
        "Classifier result: status=%s, probability=%.4f, label=%s",
        status,
        result["probability"],
        result["predicted_label"],
    )


def generate_pneumonia_explanation(image_path, pneumonia_positive, probability):
    # generate explanation conditioned on positive/negative result
    initialize_imaging_models()
    logger.info(
        "Generating explanation for image: %s (pneumonia=%s, prob=%.4f)",
        image_path,
        "positive" if pneumonia_positive else "negative",
        probability,
    )
    image = Image.open(image_path).convert("RGB")

    status = "positive" if pneumonia_positive else "negative"
    question_text = (
        f"This chest X-ray suggests {status} for pneumonia. "
        "What is the main radiographic finding in the image that supports this diagnosis?"
    )
    prompt = _build_multimodal_prompt(question_text)

    # Keep image and text aligned as a single sample for LLaVA-style processors.
    inputs = _explainer_processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            generated_ids = _explainer_model.generate(**inputs, max_new_tokens=220)
    except ValueError as exc:
        # Some checkpoints are strict about image-token alignment. Retry with a minimal prompt.
        if "Image features and image tokens do not match" not in str(exc):
            raise
        logger.warning("Image/token mismatch during generation; retrying with fallback prompt.")
        fallback_prompt = f"<image>\n{question_text}"
        retry_inputs = _explainer_processor(
            images=image, text=fallback_prompt, return_tensors="pt"
        )
        retry_inputs = {
            k: v.to(_DEVICE) if hasattr(v, "to") else v
            for k, v in retry_inputs.items()
        }
        with torch.no_grad():
            generated_ids = _explainer_model.generate(**retry_inputs, max_new_tokens=220)
        inputs = retry_inputs

    # Decode only newly generated tokens to avoid prompt-echo cleanup issues.
    input_token_count = inputs["input_ids"].shape[-1]
    generated_only = generated_ids[0][input_token_count:]
    explanation = _explainer_processor.decode(
        generated_only, skip_special_tokens=True
    ).strip()
    logger.info("Explanation generated successfully.")
    return explanation


def analyze_imaging(image_path):
    """
    Trigger function:
      1) classify pneumonia
      2) call explanation model with classification signal
      3) return probability + explanation
    """
    logger.info("Starting imaging analysis pipeline for: %s", image_path)
    classification = classify_pneumonia(image_path)
    _log_classification_result(classification)
    explanation = generate_pneumonia_explanation(
        image_path=image_path,
        pneumonia_positive=classification["pneumonia_positive"],
        probability=classification["probability"],
    )
    logger.info("Imaging analysis pipeline complete for: %s", image_path)
    return {
        "triggered": True,
        "image_path": image_path,
        "pneumonia_positive": classification["pneumonia_positive"],
        "probability": classification["probability"],
        "explanation": explanation,
    }


def run(image_path=None):
    if not image_path:
        logger.warning("run() called without image_path.")
        return "Imaging: no image_path provided."

    logger.info("run() invoked for imaging path: %s", image_path)
    result = analyze_imaging(image_path=image_path)
    status = "positive" if result["pneumonia_positive"] else "negative"
    return (
        f"Imaging prediction: {status} "
        f"(probability={result['probability']:.4f}). "
        f"Explanation: {result['explanation']}"
    )


# Model load on script initialization.
try:
    initialize_imaging_models()
except Exception as exc:
    _MODEL_INIT_ERROR = exc
    logger.exception("Imaging model initialization failed during module import: %s", exc)
