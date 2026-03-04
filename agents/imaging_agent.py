import os
import logging
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import transformers
from torchvision import transforms, models
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoProcessor


DENSENET_MODEL_PATH = "model/rsna_densenet121_best_f1.pt"
CLASSIFIER_MODEL_ID = "lxyuan/vit-xray-pneumonia-classification"
EXPLAINER_MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

CUT_POINTS = [
    0.0,
    0.010970236826688051,
    0.032142025977373125,
    0.07188209816813472,
    0.14953728318214418,
    0.26543186604976654,
    0.4187747299671178,
    0.5828946948051452,
    0.7357503771781921,
    0.883841586112976,
    1.0,
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

_MODELS_READY = False
_MODEL_INIT_ERROR = None
_classifier_model = None
_alt_classifier_processor = None
_alt_classifier_model = None
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


# -----------------------------
# DenseNet helpers
# -----------------------------
_eval_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def build_densenet121(num_classes: int = 2) -> nn.Module:
    m = models.densenet121(weights=None)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_classes)
    return m


def assign_decile(probability: float, cut_points) -> int:
    for i in range(len(cut_points) - 1):
        left = cut_points[i]
        right = cut_points[i + 1]

        if i == len(cut_points) - 2:
            if left <= probability <= right:
                return i + 1
        else:
            if left <= probability < right:
                return i + 1

    return 10


def map_decile_to_band(decile: int) -> str:
    if decile in [1, 2, 3, 4]:
        return "NORMAL"
    elif decile in [5, 6, 7, 8]:
        return "UNCERTAIN"
    elif decile in [9, 10]:
        return "PNEUMONIA"
    return "UNCERTAIN"


def initialize_imaging_models():
    """Load DenseNet (primary classifier), HF ViT (alt classifier), and explainer VLM."""
    global _MODELS_READY
    global _MODEL_INIT_ERROR
    global _classifier_model
    global _alt_classifier_processor, _alt_classifier_model
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

    # Load local DenseNet checkpoint (primary classifier)
    ckpt_path = _ROOT_DIR / DENSENET_MODEL_PATH
    checkpoint = torch.load(ckpt_path, map_location=_DEVICE)

    _classifier_model = build_densenet121(num_classes=2).to(_DEVICE)
    _classifier_model.load_state_dict(checkpoint["state_dict"])
    _classifier_model.eval()

    # Load HF ViT alt classifier (used when DenseNet prob < 0.9)
    _alt_classifier_processor = _from_pretrained_with_auth(
        AutoImageProcessor, CLASSIFIER_MODEL_ID
    )
    _alt_classifier_model = _from_pretrained_with_auth(
        AutoModelForImageClassification, CLASSIFIER_MODEL_ID
    ).to(_DEVICE)
    _alt_classifier_model.eval()

    # Load explainer VLM
    _explainer_processor = _from_pretrained_with_auth(AutoProcessor, EXPLAINER_MODEL_ID)
    _explainer_model = _load_explainer_model(EXPLAINER_MODEL_ID)
    _explainer_model.eval()

    _MODELS_READY = True
    logger.info("Imaging models initialized successfully.")


def classify_alt_pneumonia(image_path):
    # return pneumonia flag and probability (HF ViT classifier, used when DenseNet prob < 0.9)
    initialize_imaging_models()
    logger.info("Running alt pneumonia classification for image: %s", image_path)
    image = Image.open(image_path).convert("RGB")

    inputs = _alt_classifier_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _alt_classifier_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = (
            _alt_classifier_model.config.id2label.get(predicted_idx, str(predicted_idx))
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


def classify_pneumonia(image_path):
    initialize_imaging_models()
    logger.info("Running DenseNet pneumonia classification for image: %s", image_path)

    image = Image.open(image_path).convert("RGB")
    x = _eval_tfm(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = _classifier_model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        pneumonia_probability = float(probs[1].item())   # class 1 = pneumonia

    logger.info("DenseNet probability: %.4f", pneumonia_probability)
    if (pneumonia_probability > 0.4 and pneumonia_probability < 0.9):
        alt_classification = classify_alt_pneumonia(image_path)
        pneumonia_probability = alt_classification["probability"]
        pneumonia_positive = alt_classification["pneumonia_positive"]
        predicted_label = alt_classification["predicted_label"]
    decile = assign_decile(pneumonia_probability, CUT_POINTS)
    band_label = map_decile_to_band(decile)

    return {
        "pneumonia_positive": band_label == "PNEUMONIA",
        "probability": pneumonia_probability,
        "predicted_label": band_label,
        "decile": decile,
        "band_label": band_label,
    }


def _log_classification_result(result):
    logger.info(
        "Classifier result: band=%s, probability=%.4f, decile=%s",
        result["band_label"],
        result["probability"],
        result["decile"],
    )


def generate_pneumonia_explanation(image_path, pneumonia_positive, probability):
    initialize_imaging_models()
    logger.info(
        "Preparing explanation call for image: %s (pneumonia=%s, prob=%.4f)",
        image_path,
        "positive" if pneumonia_positive else "negative",
        probability,
    )

    image = Image.open(image_path).convert("RGB")
    status = "positive" if pneumonia_positive else "negative"
    
    if status == "positive":
        question_text = (
            f"This chest X-ray suggests {status} for pneumonia. "
            "What are all the radiographic findings in this image that supports this diagnosis? "
        )
    else:
        question_text = (
            f"This chest X-ray suggests {status} for pneumonia. "
        "What are all the indications in this image that rejects the pneumonia diagnosis? "
        )

    print(question_text)
    prompt = _build_multimodal_prompt(question_text)

    inputs = _explainer_processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated_ids = None
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
      1) classify with DenseNet
      2) compute probability -> decile -> band
      3) keep VLM plumbing but skip generation for now
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
        "decile": classification["decile"],
        "band_label": classification["band_label"],
        "explanation": explanation,
    }


def run(image_path=None):
    if not image_path:
        logger.warning("run() called without image_path.")
        return "Imaging: no image_path provided."

    logger.info("run() invoked for imaging path: %s", image_path)
    result = analyze_imaging(image_path=image_path)

    return (
        f"Imaging prediction: {result['band_label']} "
        f"(probability={result['probability']:.4f}, decile={result['decile']}). "
        f"Explanation: {result['explanation']}"
    )


try:
    initialize_imaging_models()
except Exception as exc:
    _MODEL_INIT_ERROR = exc
    logger.exception("Imaging model initialization failed during module import: %s", exc)