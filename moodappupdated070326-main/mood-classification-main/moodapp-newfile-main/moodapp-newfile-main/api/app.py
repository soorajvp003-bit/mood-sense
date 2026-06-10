import base64
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from threading import Lock

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").disabled = True
logging.getLogger("absl").disabled = True

import cv2
import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from absl import logging as absl_logging
    import tensorflow as tf
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    tf.get_logger().setLevel("ERROR")
    tf.get_logger().disabled = True
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    TENSORFLOW_IMPORT_ERROR = ""
except Exception as tf_error:
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(tf_error)

API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
# New improved models from mood-classification-main
NEW_TEXT_DIR = os.path.join(PROJECT_ROOT, "mood-classification-main", "text")
NEW_VISUAL_DIR = os.path.join(PROJECT_ROOT, "mood-classification-main", "visual")
# Legacy dirs kept as fallback references
TEXT_DIR = os.path.join(PROJECT_ROOT, "text based")
VISUAL_DIR = os.path.join(PROJECT_ROOT, "visual_based")
UI_DIR = os.path.join(API_DIR, "ui")

# Ensure project packages are importable regardless of launch directory.
if API_DIR not in sys.path:
    sys.path.append(API_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load .env before reading feature flags in this module.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from data import session_storage  # noqa: E402
from rbac import init_db as rbac_init_db, rbac_bp  # noqa: E402
from rbac import services as rbac_services  # noqa: E402

app = Flask(__name__)
CORS(app)

rbac_init_db()
app.register_blueprint(rbac_bp, url_prefix="/rbac")

# Text model: use original (8-class: angry/contempt/disgust/fear/happy/neutral/sad/surprise)
# The new mood-classification-main text model uses a different 6-class scheme (joy/love/etc.)
# and cannot be fused with the visual model or QUICK_CHECKIN_MAP below.
TEXT_MODEL_PATH = os.path.join(TEXT_DIR, "text_emotion_model.pkl")
VECTORIZER_PATH = os.path.join(TEXT_DIR, "tfidf_vectorizer.pkl")
# Canonical label order for visual models.
DEFAULT_EMOTIONS = [
    "angry",    # 0
    "disgust",  # 1
    "fear",     # 2
    "happy",    # 3
    "neutral",  # 4
    "sad",      # 5
    "surprise", # 6
    "contempt", # 7
]

QUICK_CHECKIN_MAP = {
    "good": ("happy", 0.75, "Glad to hear that. Keep your momentum today."),
    "okay": ("neutral", 0.65, "Thanks for checking in. A small break can help you reset."),
    "not_great": ("sad", 0.75, "Thanks for sharing. You are not alone, take this day gently."),
}
AI_MODE = os.getenv("AI_MODE", "false").strip().lower() == "true"
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip().lower()

text_model = joblib.load(TEXT_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
VISUAL_MODEL_LOCK = Lock()
VISUAL_MODEL_LOAD_ATTEMPTED = False
VISUAL_MODEL_ERROR = ""
visual_model = None
visual_model_path = ""
visual_input_channels = 1
visual_output_classes = len(DEFAULT_EMOTIONS)
EMOTIONS = DEFAULT_EMOTIONS[:]


def _load_visual_model():
    if tf is None:
        raise RuntimeError(
            "TensorFlow is unavailable for visual prediction"
            + (f": {TENSORFLOW_IMPORT_ERROR}" if TENSORFLOW_IMPORT_ERROR else ".")
        )

    env_model = os.getenv("VISUAL_MODEL_PATH", "").strip()
    candidates = []
    if env_model:
        candidates.append(env_model)
    candidates.extend(
        [
            os.path.join(VISUAL_DIR, "emotion_model.keras"),
            os.path.join(NEW_VISUAL_DIR, "emotion_model.keras"),
        ]
    )

    errors = []
    for model_path in candidates:
        if not os.path.exists(model_path):
            errors.append(f"{model_path} (missing)")
            continue
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, model_path
        except Exception as model_error:
            errors.append(f"{model_path} ({model_error})")
            continue

    raise RuntimeError(
        "Unable to load any visual model. Attempts:\n- " + "\n- ".join(errors)
    )


def _ensure_visual_model_loaded():
    global VISUAL_MODEL_LOAD_ATTEMPTED
    global VISUAL_MODEL_ERROR
    global visual_model
    global visual_model_path
    global visual_input_channels
    global visual_output_classes
    global EMOTIONS

    if visual_model is not None:
        return True

    with VISUAL_MODEL_LOCK:
        if visual_model is not None:
            return True
        if VISUAL_MODEL_LOAD_ATTEMPTED:
            return False

        VISUAL_MODEL_LOAD_ATTEMPTED = True
        try:
            model, model_path = _load_visual_model()
            output_classes = int(model.output_shape[-1])
            emotions = _load_emotion_labels(model_path, output_classes)
            if len(emotions) != output_classes:
                emotions = DEFAULT_EMOTIONS[:output_classes]

            visual_model = model
            visual_model_path = model_path
            visual_input_channels = model.input_shape[-1]
            visual_output_classes = output_classes
            EMOTIONS = emotions
            VISUAL_MODEL_ERROR = ""
            return True
        except Exception as visual_error:
            VISUAL_MODEL_ERROR = str(visual_error)
            logging.warning("Visual model unavailable: %s", visual_error)
            return False


def _load_emotion_labels(model_path, expected_classes):
    labels_path = os.path.join(os.path.dirname(model_path), "emotion_labels.json")
    if not os.path.exists(labels_path):
        return DEFAULT_EMOTIONS[:expected_classes]

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if (
            isinstance(labels, list)
            and len(labels) == expected_classes
            and all(isinstance(item, str) for item in labels)
        ):
            return [item.strip().lower() for item in labels]
        logging.warning(
            "Ignoring invalid emotion_labels.json at %s (expected %s labels).",
            labels_path,
            expected_classes,
        )
    except Exception as labels_error:
        logging.warning("Failed reading emotion labels from %s: %s", labels_path, labels_error)

    return DEFAULT_EMOTIONS[:expected_classes]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

VISUAL_TRACK_SESSIONS = {}
VISUAL_TRACK_LOCK = Lock()
VISUAL_TRACK_TTL_SECONDS = int(os.getenv("VISUAL_TRACK_TTL_SECONDS", "180"))
VISUAL_TEMPLATE_ALPHA = float(os.getenv("VISUAL_TEMPLATE_ALPHA", "0.18"))
VISUAL_MATCH_THRESHOLD = float(os.getenv("VISUAL_MATCH_THRESHOLD", "0.82"))
VISUAL_LOCK_RETAIN_THRESHOLD = float(os.getenv("VISUAL_LOCK_RETAIN_THRESHOLD", "0.6"))
VISUAL_LOCK_HOLD_SECONDS = float(os.getenv("VISUAL_LOCK_HOLD_SECONDS", "6"))
VISUAL_LOCK_STICKY = os.getenv("VISUAL_LOCK_STICKY", "true").strip().lower() == "true"
FACE_PROFILE_MATCH_THRESHOLD = float(os.getenv("FACE_PROFILE_MATCH_THRESHOLD", "0.93"))
FACE_PROFILE_MAX_IMAGES = max(1, min(int(os.getenv("FACE_PROFILE_MAX_IMAGES", "6")), 10))
FACE_PROFILE_RECOMMENDED_IMAGES = max(1, min(int(os.getenv("FACE_PROFILE_RECOMMENDED_IMAGES", "3")), 10))
FACE_DETECT_MIN_SIZE = max(40, int(os.getenv("FACE_DETECT_MIN_SIZE", "80")))
FACE_DETECT_MIN_NEIGHBORS = max(3, int(os.getenv("FACE_DETECT_MIN_NEIGHBORS", "6")))


def _clip01(value):
    return max(0.0, min(float(value), 1.0))


def _update_locked_emotion(session_key, emotion, confidence):
    if not session_key:
        return
    with VISUAL_TRACK_LOCK:
        state = VISUAL_TRACK_SESSIONS.get(session_key)
        if not state:
            return
        state["last_emotion"] = str(emotion or "").strip().lower()
        state["last_confidence"] = float(confidence or 0.0)
        state["last_at"] = time.time()


def _maybe_hold_locked_emotion(session_key, required_profile=None, reason="no_face"):
    if not session_key:
        return None

    hold_seconds = float(VISUAL_LOCK_HOLD_SECONDS or 0.0)
    if not VISUAL_LOCK_STICKY and hold_seconds <= 0:
        return None
    with VISUAL_TRACK_LOCK:
        state = VISUAL_TRACK_SESSIONS.get(session_key)
        if not state:
            return None
        last_emotion = state.get("last_emotion")
        last_confidence = float(state.get("last_confidence", 0.0) or 0.0)
        last_at = float(state.get("last_at", 0.0) or 0.0)

        if not last_emotion:
            return None
        if not VISUAL_LOCK_STICKY and hold_seconds > 0 and (time.time() - last_at) > hold_seconds:
            return None

        state["updated_at"] = time.time()

    status = "locked_sticky" if VISUAL_LOCK_STICKY else "locked_hold"
    return {
        "face_detected": True,
        "emotion": last_emotion,
        "confidence": round(last_confidence, 4),
        "tracking_locked": True,
        "tracking_status": status,
        "hold_reason": reason,
        "faces_detected": 0,
        "ignored_faces": 0,
        "unauthorized_faces": 0,
        "account_face_enforced": bool(required_profile),
        "profile_match_score": 0.0,
        "session_id": session_key,
    }


def _cleanup_visual_sessions():
    cutoff = time.time() - VISUAL_TRACK_TTL_SECONDS
    with VISUAL_TRACK_LOCK:
        stale_keys = [
            key
            for key, state in VISUAL_TRACK_SESSIONS.items()
            if state.get("updated_at", 0.0) < cutoff
        ]
        for key in stale_keys:
            VISUAL_TRACK_SESSIONS.pop(key, None)


def _clear_visual_session(session_key):
    if not session_key:
        return
    with VISUAL_TRACK_LOCK:
        VISUAL_TRACK_SESSIONS.pop(session_key, None)


def _get_visual_session_key(payload, req):
    explicit = str(payload.get("session_id") or payload.get("visual_session_id") or "").strip()
    if explicit:
        return explicit

    email = _get_user_email(req, payload)
    if email:
        return f"user:{email}"

    remote_addr = str(req.remote_addr or "").strip()
    if remote_addr:
        return f"ip:{remote_addr}"

    return ""


def _face_center(bbox):
    x, y, w, h = bbox
    return (x + (w / 2.0), y + (h / 2.0))


def _extract_face_signature(gray, bbox):
    x, y, w, h = bbox
    face = gray[y : y + h, x : x + w]
    if face.size == 0:
        return None

    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (48, 48)).astype("float32") / 255.0

    vector = face.flatten()
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    hist = cv2.calcHist([(face * 255).astype("uint8")], [0], None, [16], [0, 256]).flatten().astype("float32")
    hist_sum = float(hist.sum()) or 1.0
    hist = hist / hist_sum

    return {
        "face": face,
        "vector": vector,
        "hist": hist,
    }


def _normalize_face_profile(profile):
    if not isinstance(profile, dict):
        return None
    if not bool(profile.get("enabled", True)):
        return None

    subject_id = str(profile.get("subject_id") or profile.get("email") or "").strip().lower()
    vector = np.asarray(profile.get("template_vector") or profile.get("vector") or [], dtype="float32").flatten()
    hist = np.asarray(profile.get("template_hist") or profile.get("hist") or [], dtype="float32").flatten()
    if vector.size == 0 or hist.size == 0:
        return None

    vector_norm = np.linalg.norm(vector)
    if vector_norm > 0:
        vector = vector / vector_norm

    hist_sum = float(hist.sum()) or 1.0
    hist = hist / hist_sum

    threshold = float(profile.get("match_threshold", FACE_PROFILE_MATCH_THRESHOLD) or FACE_PROFILE_MATCH_THRESHOLD)
    threshold = _clip01(threshold)
    if threshold <= 0.0:
        threshold = FACE_PROFILE_MATCH_THRESHOLD

    return {
        "subject_id": subject_id,
        "label": str(profile.get("label") or "").strip(),
        "vector": vector.astype("float32"),
        "hist": hist.astype("float32"),
        "sample_count": max(0, int(profile.get("sample_count") or 0)),
        "match_threshold": threshold,
        "enabled": True,
        "created_at": str(profile.get("created_at") or "").strip(),
        "updated_at": str(profile.get("updated_at") or "").strip(),
    }


def _score_face_profile(candidate_vector, candidate_hist, profile):
    profile_vector = profile.get("vector")
    profile_hist = profile.get("hist")
    cosine_score = float(np.dot(candidate_vector, profile_vector)) if profile_vector is not None else 0.0
    cosine_score = _clip01((cosine_score + 1.0) / 2.0)

    hist_score = 0.0
    if profile_hist is not None:
        hist_score = float(cv2.compareHist(candidate_hist, profile_hist, cv2.HISTCMP_CORREL))
        hist_score = _clip01((hist_score + 1.0) / 2.0)

    return (cosine_score * 0.8) + (hist_score * 0.2)


def _collect_enrollment_images(payload):
    images = []
    raw_images = payload.get("images")
    if isinstance(raw_images, list):
        images.extend(raw_images)
    elif raw_images:
        images.append(raw_images)

    if payload.get("image"):
        images.insert(0, payload.get("image"))

    clean_images = []
    for image in images:
        value = str(image or "").strip()
        if value:
            clean_images.append(value)
        if len(clean_images) >= FACE_PROFILE_MAX_IMAGES:
            break

    return clean_images


def _build_face_profile_template(images):
    accepted_signatures = []
    rejected_images = []

    for index, image in enumerate(images[:FACE_PROFILE_MAX_IMAGES], start=1):
        frame = _decode_base64_image(image)
        if frame is None:
            rejected_images.append({"index": index, "reason": "invalid_image"})
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
            minSize=(FACE_DETECT_MIN_SIZE, FACE_DETECT_MIN_SIZE),
        )
        if len(faces) == 0:
            rejected_images.append({"index": index, "reason": "no_face"})
            continue
        if len(faces) > 1:
            rejected_images.append({"index": index, "reason": "multiple_faces"})
            continue

        bbox = faces[0]
        signature = _extract_face_signature(gray, bbox)
        if not signature:
            rejected_images.append({"index": index, "reason": "face_read_failed"})
            continue

        accepted_signatures.append(signature)

    if not accepted_signatures:
        return None, rejected_images

    vector = np.mean(np.stack([signature["vector"] for signature in accepted_signatures], axis=0), axis=0)
    vector_norm = np.linalg.norm(vector)
    if vector_norm > 0:
        vector = vector / vector_norm

    hist = np.mean(np.stack([signature["hist"] for signature in accepted_signatures], axis=0), axis=0)
    hist_sum = float(hist.sum()) or 1.0
    hist = hist / hist_sum

    profile = {
        "template_vector": vector.astype("float32").tolist(),
        "template_hist": hist.astype("float32").tolist(),
        "sample_count": len(accepted_signatures),
    }
    return profile, rejected_images


def _initial_track_score(bbox, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    frame_area = max(frame_h * frame_w, 1)
    x, y, w, h = bbox
    area_score = min((w * h) / frame_area * 6.0, 1.0)

    cx, cy = _face_center(bbox)
    center_x = frame_w / 2.0
    center_y = frame_h / 2.0
    diagonal = max((center_x ** 2 + center_y ** 2) ** 0.5, 1.0)
    center_distance = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
    centered_score = max(0.0, 1.0 - (center_distance / diagonal))

    return (centered_score * 0.7) + (area_score * 0.3)


def _center_similarity(bbox, previous_bbox, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    diagonal = max((frame_h ** 2 + frame_w ** 2) ** 0.5, 1.0)
    cx, cy = _face_center(bbox)
    px, py = _face_center(previous_bbox)
    distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
    return max(0.0, 1.0 - (distance / diagonal))


def _match_locked_face(candidate_bbox, candidate_vector, candidate_hist, state, frame_shape):
    locked_vector = state.get("vector")
    locked_hist = state.get("hist")
    previous_bbox = state.get("bbox")

    cosine_score = float(np.dot(candidate_vector, locked_vector)) if locked_vector is not None else 0.0
    cosine_score = _clip01((cosine_score + 1.0) / 2.0)

    hist_score = 0.0
    if locked_hist is not None:
        hist_score = float(cv2.compareHist(candidate_hist, locked_hist, cv2.HISTCMP_CORREL))
        hist_score = _clip01((hist_score + 1.0) / 2.0)

    if previous_bbox is not None:
        spatial_score = _center_similarity(candidate_bbox, previous_bbox, frame_shape)
    else:
        spatial_score = _initial_track_score(candidate_bbox, frame_shape)

    return (cosine_score * 0.65) + (hist_score * 0.2) + (spatial_score * 0.15)


def _select_face_for_session(gray, faces, session_key, required_profile=None):
    _cleanup_visual_sessions()
    candidates = []
    profile_threshold = 0.0
    if required_profile:
        profile_threshold = float(required_profile.get("match_threshold", FACE_PROFILE_MATCH_THRESHOLD))

    for bbox in faces:
        signature = _extract_face_signature(gray, bbox)
        if signature:
            profile_score = None
            authorized = True
            if required_profile:
                profile_score = _score_face_profile(signature["vector"], signature["hist"], required_profile)
                authorized = profile_score >= profile_threshold
            candidates.append(
                {
                    "bbox": bbox,
                    "signature": signature,
                    "profile_score": profile_score,
                    "authorized": authorized,
                }
            )

    if not candidates:
        return None, {
            "tracking_locked": False,
            "tracking_status": "no_face",
            "ignored_faces": 0,
            "faces_detected": 0,
            "lock_score": 0.0,
            "profile_match_score": 0.0,
            "unauthorized_faces": 0,
            "account_face_enforced": bool(required_profile),
        }

    authorized_candidates = [item for item in candidates if item["authorized"]]
    unauthorized_faces = max(0, len(candidates) - len(authorized_candidates))
    best_profile_score = max(
        [float(item["profile_score"] or 0.0) for item in candidates],
        default=0.0,
    )

    if required_profile and not authorized_candidates:
        with VISUAL_TRACK_LOCK:
            state = VISUAL_TRACK_SESSIONS.get(session_key)
            if state:
                state["updated_at"] = time.time()
                state["misses"] = int(state.get("misses", 0)) + 1
        return None, {
            "tracking_locked": False,
            "tracking_status": "face_not_authorized",
            "ignored_faces": max(0, len(candidates)),
            "faces_detected": len(candidates),
            "lock_score": round(best_profile_score, 4),
            "profile_match_score": round(best_profile_score, 4),
            "unauthorized_faces": unauthorized_faces,
            "account_face_enforced": True,
        }

    with VISUAL_TRACK_LOCK:
        state = VISUAL_TRACK_SESSIONS.get(session_key)
        usable_candidates = authorized_candidates if required_profile else candidates

        if not state:
            candidate = max(
                usable_candidates,
                key=lambda item: (
                    float(item["profile_score"] or 0.0) * 0.7
                    + _initial_track_score(item["bbox"], gray.shape) * 0.3
                )
                if required_profile
                else _initial_track_score(item["bbox"], gray.shape),
            )
            bbox = candidate["bbox"]
            signature = candidate["signature"]
            VISUAL_TRACK_SESSIONS[session_key] = {
                "bbox": bbox,
                "vector": signature["vector"],
                "hist": signature["hist"],
                "updated_at": time.time(),
                "misses": 0,
            }
            return (
                (bbox, signature),
                {
                    "tracking_locked": True,
                    "tracking_status": "profile_locked" if required_profile else "locked",
                    "ignored_faces": max(0, len(candidates) - 1),
                    "faces_detected": len(candidates),
                    "lock_score": 1.0,
                    "profile_match_score": round(float(candidate["profile_score"] or 1.0), 4),
                    "unauthorized_faces": unauthorized_faces,
                    "account_face_enforced": bool(required_profile),
                },
            )

        scored = []
        for candidate in usable_candidates:
            track_score = _match_locked_face(
                candidate["bbox"],
                candidate["signature"]["vector"],
                candidate["signature"]["hist"],
                state,
                gray.shape,
            )
            total_score = (
                (track_score * 0.6) + (float(candidate["profile_score"] or 0.0) * 0.4)
                if required_profile
                else track_score
            )
            scored.append(
                (
                    total_score,
                    track_score,
                    float(candidate["profile_score"] or 0.0),
                    candidate,
                )
            )

        best_score, best_track_score, best_profile_score, best_candidate = max(scored, key=lambda item: item[0])
        best_bbox = best_candidate["bbox"]
        best_signature = best_candidate["signature"]

        retain_threshold = _clip01(VISUAL_LOCK_RETAIN_THRESHOLD)
        if best_score < VISUAL_MATCH_THRESHOLD:
            if retain_threshold > 0.0 and best_score >= retain_threshold:
                template_alpha = _clip01(VISUAL_TEMPLATE_ALPHA)
                blended_vector = ((1.0 - template_alpha) * state["vector"]) + (template_alpha * best_signature["vector"])
                blended_norm = np.linalg.norm(blended_vector)
                if blended_norm > 0:
                    blended_vector = blended_vector / blended_norm

                blended_hist = ((1.0 - template_alpha) * state["hist"]) + (template_alpha * best_signature["hist"])
                hist_sum = float(blended_hist.sum()) or 1.0
                blended_hist = blended_hist / hist_sum

                state["bbox"] = best_bbox
                state["vector"] = blended_vector
                state["hist"] = blended_hist
                state["updated_at"] = time.time()
                state["misses"] = 0

                return (
                    (best_bbox, best_signature),
                    {
                        "tracking_locked": True,
                        "tracking_status": "locked_relaxed",
                        "ignored_faces": max(0, len(candidates) - 1),
                        "faces_detected": len(candidates),
                        "lock_score": round(best_score, 4),
                        "profile_match_score": round(best_profile_score, 4),
                        "track_score": round(best_track_score, 4),
                        "unauthorized_faces": unauthorized_faces,
                        "account_face_enforced": bool(required_profile),
                    },
                )

            state["updated_at"] = time.time()
            state["misses"] = int(state.get("misses", 0)) + 1
            return (
                None,
                {
                    "tracking_locked": False,
                    "tracking_status": "locked_face_not_found",
                    "ignored_faces": max(0, len(candidates)),
                    "faces_detected": len(candidates),
                    "lock_score": round(best_score, 4),
                    "profile_match_score": round(best_profile_score, 4),
                    "unauthorized_faces": unauthorized_faces,
                    "account_face_enforced": bool(required_profile),
                },
            )

        template_alpha = _clip01(VISUAL_TEMPLATE_ALPHA)
        blended_vector = ((1.0 - template_alpha) * state["vector"]) + (template_alpha * best_signature["vector"])
        blended_norm = np.linalg.norm(blended_vector)
        if blended_norm > 0:
            blended_vector = blended_vector / blended_norm

        blended_hist = ((1.0 - template_alpha) * state["hist"]) + (template_alpha * best_signature["hist"])
        hist_sum = float(blended_hist.sum()) or 1.0
        blended_hist = blended_hist / hist_sum

        state["bbox"] = best_bbox
        state["vector"] = blended_vector
        state["hist"] = blended_hist
        state["updated_at"] = time.time()
        state["misses"] = 0

        return (
            (best_bbox, best_signature),
                {
                    "tracking_locked": True,
                    "tracking_status": "profile_locked" if required_profile else "locked",
                    "ignored_faces": max(0, len(candidates) - 1),
                    "faces_detected": len(candidates),
                    "lock_score": round(best_score, 4),
                    "profile_match_score": round(best_profile_score, 4),
                    "track_score": round(best_track_score, 4),
                    "unauthorized_faces": unauthorized_faces,
                    "account_face_enforced": bool(required_profile),
                },
            )


def _get_text_confidence(text_vector, predicted_emotion):
    if hasattr(text_model, "predict_proba"):
        probs = text_model.predict_proba(text_vector)[0]
        class_index = list(text_model.classes_).index(predicted_emotion)
        return float(probs[class_index])
    return 0.5


def _decode_base64_image(image_data):
    if not image_data:
        return None

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        binary = base64.b64decode(image_data)
        arr = np.frombuffer(binary, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def _predict_emotion_from_frame(frame, session_key="", reset_session=False, required_profile=None):
    if not _ensure_visual_model_loaded():
        return {
            "error": "Visual model unavailable",
            "detail": VISUAL_MODEL_ERROR or "visual model failed to load",
            "face_detected": False,
            "emotion": "neutral",
            "confidence": 0.0,
            "tracking_locked": False,
            "tracking_status": "visual_model_unavailable",
            "faces_detected": 0,
            "ignored_faces": 0,
            "unauthorized_faces": 0,
            "account_face_enforced": bool(required_profile),
            "profile_match_score": 0.0,
            "session_id": session_key,
        }

    if reset_session and session_key:
        _clear_visual_session(session_key)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
        minSize=(FACE_DETECT_MIN_SIZE, FACE_DETECT_MIN_SIZE),
    )

    if len(faces) == 0:
        hold = _maybe_hold_locked_emotion(session_key, required_profile, reason="no_face")
        if hold:
            return hold
        return {
            "face_detected": False,
            "emotion": "neutral",
            "confidence": 0.0,
            "tracking_locked": False,
            "tracking_status": "no_face",
            "faces_detected": 0,
            "ignored_faces": 0,
            "unauthorized_faces": 0,
            "account_face_enforced": bool(required_profile),
            "profile_match_score": 0.0,
            "session_id": session_key,
        }

    tracking_meta = {
        "tracking_locked": False,
        "tracking_status": "stateless",
        "faces_detected": int(len(faces)),
        "ignored_faces": max(0, int(len(faces)) - 1),
        "lock_score": 0.0,
        "profile_match_score": 0.0,
        "unauthorized_faces": 0,
        "account_face_enforced": bool(required_profile),
    }

    if session_key:
        selected, tracking_meta = _select_face_for_session(
            gray,
            faces,
            session_key,
            required_profile=required_profile,
        )
        if selected is None:
            hold = _maybe_hold_locked_emotion(
                session_key,
                required_profile,
                reason=tracking_meta.get("tracking_status", "locked_face_not_found"),
            )
            if hold:
                hold["faces_detected"] = tracking_meta.get("faces_detected", 0)
                hold["ignored_faces"] = tracking_meta.get("ignored_faces", 0)
                hold["unauthorized_faces"] = tracking_meta.get("unauthorized_faces", 0)
                hold["profile_match_score"] = tracking_meta.get("profile_match_score", 0.0)
                return hold
            return {
                "face_detected": False,
                "emotion": "neutral",
                "confidence": 0.0,
                "session_id": session_key,
                **tracking_meta,
            }
        (x, y, w, h), signature = selected
        face = signature["face"]
    else:
        x, y, w, h = max(faces, key=lambda f: _initial_track_score(f, gray.shape))
        face = gray[y : y + h, x : x + w]
        face = cv2.equalizeHist(face)
        face = cv2.resize(face, (48, 48)).astype("float32") / 255.0

    if visual_input_channels == 3:
        face = np.stack([face, face, face], axis=-1)
    else:
        face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)

    probs = visual_model.predict(face, verbose=0)[0]
    emotion = session_storage.normalize_mood(EMOTIONS[int(np.argmax(probs))])
    confidence = float(np.max(probs))
    _update_locked_emotion(session_key, emotion, confidence)

    return {
        "face_detected": True,
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "session_id": session_key,
        **tracking_meta,
    }


def _soft_weekly_summary(overall_rows):
    if not overall_rows:
        return "No weekly pattern yet. Keep checking in and this space will grow with you."

    supportive = 0
    heavy = 0
    for row in overall_rows:
        mood = str(row.get("overall_mood", "neutral")).lower()
        if mood in ["happy", "neutral", "surprise"]:
            supportive += 1
        elif mood in ["sad", "fear", "angry", "disgust", "contempt", "anxious"]:
            heavy += 1

    if heavy == 0:
        return "This week looked mostly steady and balanced. Keep the same routine that supports you."

    if supportive >= heavy:
        return (
            f"This week had both calm and heavy moments. "
            f"You stayed steady in many check-ins, with around {heavy} heavier moment(s)."
        )

    return (
        f"This week felt heavier in several moments. "
        f"Consider small daily resets and reach out if you need support."
    )


def _build_today_support_prompt(overall, trend, suggestions):
    return f"""
You are a gentle student wellbeing assistant.

Current overall mood: {overall.get('overall_mood', 'neutral')}
Severity: {overall.get('severity', 'low')}
Recent trend: {trend}
Helpful suggestions: {", ".join(suggestions) if suggestions else "Take a short calming break"}

Write:
1) one short supportive message (max 2 sentences)
2) one practical nudge sentence

Use warm, non-technical language.
Do not mention AI, models, probabilities, or diagnosis.
"""


def _build_weekly_reflection_prompt(rows):
    mood_counts = {}
    for row in rows:
        mood = str(row.get("overall_mood", "neutral")).lower()
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    return f"""
You are a student wellbeing coach.

Last 7-day check-in counts by mood:
{mood_counts}

Write a soft weekly reflection in 2-3 short sentences.
Use human-friendly language and avoid percentages.
No diagnosis, no technical wording.
"""


def _ai_mode_available():
    try:
        llm_mode = rbac_services.get_setting("llm_mode", "llm")
        if llm_mode == "mock":
            return False
    except Exception:
        pass
    return AI_MODE and session_storage.is_llm_ready()


def _ai_threshold():
    try:
        return float(rbac_services.get_setting("ai_threshold", "0.7"))
    except Exception:
        return 0.7


def _severity_with_threshold(mood, confidence):
    threshold = _ai_threshold()
    base = session_storage.determine_severity(mood, confidence)
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    if mood in getattr(session_storage, "HEAVY_MOODS", set()) and conf >= threshold:
        return "high"
    if mood in getattr(session_storage, "HEAVY_MOODS", set()) and conf >= (threshold * 0.8):
        return "medium"
    return base


def _admin_allowed(req):
    if not ADMIN_KEY and not ADMIN_EMAIL:
        return True

    if ADMIN_KEY:
        key = req.headers.get("X-Admin-Key") or req.args.get("key") or ""
        if key == ADMIN_KEY:
            return True

    if ADMIN_EMAIL:
        email = (req.headers.get("X-Admin-Email") or "").strip().lower()
        if email == ADMIN_EMAIL:
            return True

    return False


def _log_response(response_type, input_text, response_text, source, meta=None, subject_id=""):
    try:
        meta_payload = json.dumps(meta, ensure_ascii=False) if meta else ""
        session_storage.log_response(
            response_type,
            input_text,
            response_text,
            source,
            meta=meta_payload,
            subject_id=subject_id,
        )
    except Exception:
        return


def _get_anon_id(email, name="", provider="firebase"):
    if not email:
        return ""
    try:
        rbac_services.upsert_user_from_auth(email)
    except Exception:
        pass
    try:
        return rbac_services.hash_subject(email)
    except Exception:
        return ""


def _get_subject_id(value):
    if not value:
        return ""
    try:
        return rbac_services.hash_subject(value)
    except Exception:
        return ""


def _maybe_flag_text(anon_id, text, severity):
    if not anon_id or not text:
        return
    risky_terms = ["suicide", "kill myself", "self-harm", "hurt myself", "end my life"]
    lower = text.lower()
    if any(term in lower for term in risky_terms) or severity == "high":
        rbac_services.flag_content(
            anon_id,
            flag_type="high_risk",
            severity=severity or "high",
            snippet="",
        )


def _get_user_email(req, payload=None):
    email = ""
    if payload:
        email = str(payload.get("user_email") or payload.get("email") or "").strip()
    if not email:
        email = str(req.headers.get("X-User-Email") or "").strip()
    return email.lower()


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/llm/status")
def llm_status():
    return jsonify(session_storage.get_llm_status()), 200


@app.get("/")
def ui_index():
    return send_from_directory(UI_DIR, "index.html")


@app.get("/admin/config")
def admin_config():
    return jsonify(
        {
            "admin_email": ADMIN_EMAIL,
            "admin_email_enabled": bool(ADMIN_EMAIL),
            "admin_key_enabled": bool(ADMIN_KEY),
        }
    ), 200


@app.get("/admin/data")
def admin_data():
    if not _admin_allowed(request):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        limit = int(request.args.get("limit", 200))
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 5000))

    text_rows = session_storage.read_csv_rows(session_storage.TEXT_FILE, limit=limit)
    visual_rows = session_storage.read_csv_rows(session_storage.VISUAL_FILE, limit=limit)
    overall_rows = session_storage.read_csv_rows(session_storage.OVERALL_FILE, limit=limit)
    response_rows = session_storage.read_responses(limit=limit)
    user_rows = session_storage.read_users(limit=limit)
    activity_rows = session_storage.read_activity(limit=limit)
    face_profile_rows = session_storage.read_face_profiles(limit=limit)

    return jsonify(
        {
            "limit": limit,
            "storage": {
                "session_db": getattr(session_storage, "DB_FILE", ""),
                "weekly_snapshot": getattr(session_storage, "WEEKLY_FILE", ""),
                "face_profiles": getattr(session_storage, "FACE_PROFILE_FILE", ""),
            },
            "counts": {
                "text_sessions": len(text_rows),
                "visual_sessions": len(visual_rows),
                "overall_sessions": len(overall_rows),
                "responses": len(response_rows),
                "users": len(user_rows),
                "activity": len(activity_rows),
                "face_profiles": len(face_profile_rows),
            },
            "text_sessions": text_rows,
            "visual_sessions": visual_rows,
            "overall_sessions": overall_rows,
            "responses": response_rows,
            "users": user_rows,
            "activity": activity_rows,
            "face_profiles": face_profile_rows,
        }
    ), 200


@app.post("/auth/track")
def track_auth():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    name = str(payload.get("name", "")).strip()
    provider = str(payload.get("provider", "")).strip()
    created_at = str(payload.get("created_at", "")).strip()

    if not email:
        return jsonify({"error": "missing email"}), 400

    try:
        rbac_services.upsert_user_from_auth(email)
    except Exception:
        pass

    subject_id = rbac_services.hash_subject(email)
    ok = session_storage.upsert_user(subject_id, created_at=created_at)
    return jsonify({"ok": bool(ok)}), 200


@app.post("/checkin/quick")
def quick_checkin():
    payload = request.get_json(silent=True) or {}
    feeling = str(payload.get("feeling", "")).strip().lower()
    user_email = _get_user_email(request, payload)

    if feeling not in QUICK_CHECKIN_MAP:
        return jsonify({"error": "Invalid feeling option"}), 400

    mood, confidence, message = QUICK_CHECKIN_MAP[feeling]
    session_storage.save_text_session(mood, confidence)

    llm_message = ""
    llm_source = "none"
    if _ai_mode_available():
        prompt = f"""You are MoodSense, a warm student wellbeing companion.

A student just did a quick check-in and said they are feeling: {feeling}
Their mood is: {mood}

Write one warm, supportive sentence (max 25 words) that acknowledges how they feel.
Do NOT mention AI, models, or algorithms. Be human and gentle."""
        try:
            llm_message = session_storage.get_llm_response(prompt)
            llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
        except Exception:
            llm_message = ""
            llm_source = "error"

    display_mood = "good" if mood == "happy" else ("okay" if mood == "neutral" else "not great")
    response_message = llm_message if llm_message and llm_source not in ("none", "error") else message
    subject_id = _get_anon_id(user_email) if user_email else ""
    _log_response(
        "checkin_quick",
        feeling,
        response_message,
        llm_source,
        meta={"mood": mood, "confidence": confidence, "display_mood": display_mood},
        subject_id=subject_id,
    )
    if subject_id:
        severity = _severity_with_threshold(mood, confidence)
        rbac_services.record_mood_entry(subject_id, mood, confidence, severity, source="quick_checkin")
        rbac_services.record_activity(subject_id, "checkin_quick", mood=mood, confidence=confidence, detail=feeling)

    return jsonify(
        {
            "saved": True,
            "message": response_message,
            "display_mood": display_mood,
            "llm_source": llm_source,
        }
    ), 200


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    user_email = _get_user_email(request, payload)

    if len(message) < 1:
        return jsonify({"error": "Message is empty"}), 400

    reply = session_storage.get_chat_response(message)
    llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
    llm_error = getattr(session_storage, "LAST_LLM_ERROR", "")
    subject_id = _get_anon_id(user_email) if user_email else ""

    _log_response(
        "chat",
        message,
        reply,
        llm_source,
        meta={"llm_error": llm_error},
        subject_id=subject_id,
    )
    if subject_id:
        rbac_services.record_activity(
            subject_id,
            "chat",
            mood="",
            confidence="",
            detail=f"len:{len(message)}",
        )
        _maybe_flag_text(subject_id, message, "medium")

    return jsonify(
        {
            "reply": reply,
            "llm_source": llm_source,
            "llm_error": llm_error,
        }
    ), 200


@app.post("/predict")
@app.post("/predict-text")
@app.post("/predict/text")
def predict_text():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    user_email = _get_user_email(request, payload)

    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    text_vector = vectorizer.transform([text])
    emotion = session_storage.normalize_mood(text_model.predict(text_vector)[0])
    confidence = round(_get_text_confidence(text_vector, emotion), 4)

    if payload.get("save_session", True):
        session_storage.save_text_session(emotion, confidence)

    # LLM-enhanced analysis
    llm_response = ""
    llm_source = "none"
    if _ai_mode_available():
        prompt = f"""You are a warm, empathetic student wellbeing companion called MoodSense.

A student just shared this reflection about their day:
"{text}"

Our analysis detected their mood as: {emotion} (confidence: {confidence})

Please respond with:
1. A brief, warm acknowledgment of what they shared (1-2 sentences)
2. A gentle insight about their emotional state (1 sentence)
3. One small, practical self-care suggestion (1 sentence)

Rules:
- Be warm, human, and conversational
- Do NOT mention AI, models, algorithms, or confidence scores
- Do NOT diagnose or give medical advice
- Keep your total response under 80 words
- Use a supportive, non-judgmental tone"""
        try:
            llm_response = session_storage.get_llm_response(prompt)
            llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
        except Exception:
            llm_response = ""
            llm_source = "error"

    companion_synced = False
    if llm_response:
        try:
            session_storage.sync_reflection_to_companion(text, llm_response)
            companion_synced = True
        except Exception:
            companion_synced = False

    fallback_message = (
        f"Thanks for sharing. It sounds like a {emotion} moment. "
        f"Try one small reset and be gentle with yourself."
    )
    log_source = llm_source if llm_response else "fallback"
    subject_id = _get_anon_id(user_email) if user_email else ""
    _log_response(
        "text_reflection",
        text,
        llm_response or fallback_message,
        log_source,
        meta={"emotion": emotion, "confidence": confidence},
        subject_id=subject_id,
    )
    if subject_id:
        severity = _severity_with_threshold(emotion, confidence)
        rbac_services.record_mood_entry(subject_id, emotion, confidence, severity, source="text_reflection")
        rbac_services.record_activity(
            subject_id,
            "text_reflection",
            mood=emotion,
            confidence=confidence,
            detail=f"len:{len(text)}",
        )
        _maybe_flag_text(subject_id, text, severity)

    return jsonify(
        {
            "emotion": emotion,
            "confidence": confidence,
            "saved": bool(payload.get("save_session", True)),
            "llm_response": llm_response,
            "llm_source": llm_source,
            "companion_reply": llm_response,
            "companion_source": llm_source,
            "companion_synced": companion_synced,
        }
    ), 200


@app.get("/face/profile")
def face_profile_status():
    user_email = _get_user_email(request, {})
    if not user_email:
        return jsonify({"error": "missing user_email"}), 400

    subject_id = _get_subject_id(user_email)
    profile = _normalize_face_profile(session_storage.get_face_profile(subject_id))
    if not profile:
        return jsonify(
            {
                "enrolled": False,
                "recommended_images": FACE_PROFILE_RECOMMENDED_IMAGES,
                "max_images": FACE_PROFILE_MAX_IMAGES,
            }
        ), 200

    return jsonify(
        {
            "enrolled": True,
            "label": profile["label"],
            "sample_count": profile["sample_count"],
            "match_threshold": round(profile["match_threshold"], 4),
            "enabled": True,
            "created_at": profile["created_at"],
            "updated_at": profile["updated_at"],
            "recommended_images": FACE_PROFILE_RECOMMENDED_IMAGES,
            "max_images": FACE_PROFILE_MAX_IMAGES,
        }
    ), 200


@app.post("/face/enroll")
def face_profile_enroll():
    payload = request.get_json(silent=True) or {}
    user_email = _get_user_email(request, payload)
    user_name = str(payload.get("name", "")).strip()
    provider = str(payload.get("provider", "firebase")).strip() or "firebase"
    label = str(payload.get("label", "")).strip()

    if not user_email:
        return jsonify({"error": "missing user_email"}), 400

    subject_id = _get_anon_id(user_email, name=user_name, provider=provider)
    if not subject_id:
        return jsonify({"error": "unable to resolve user"}), 400

    images = _collect_enrollment_images(payload)
    if not images:
        return jsonify({"error": "Provide at least one reference image."}), 400

    template, rejected_images = _build_face_profile_template(images)
    if not template:
        return jsonify(
            {
                "error": "No valid single-face reference image found.",
                "rejected_images": rejected_images,
                "recommended_images": FACE_PROFILE_RECOMMENDED_IMAGES,
                "max_images": FACE_PROFILE_MAX_IMAGES,
            }
        ), 400

    match_threshold = payload.get("match_threshold", FACE_PROFILE_MATCH_THRESHOLD)
    display_label = label or user_name or "profile"
    saved = session_storage.save_face_profile(
        subject_id,
        template["template_vector"],
        template["template_hist"],
        template["sample_count"],
        label=display_label,
        match_threshold=match_threshold,
        enabled=True,
    )
    if not saved:
        return jsonify({"error": "Unable to save face profile."}), 500

    session_storage.log_activity(
        subject_id,
        "face_profile_enrolled",
        detail=f"{template['sample_count']} accepted / {len(images)} submitted",
    )
    rbac_services.record_activity(
        subject_id,
        "face_profile_enrolled",
        mood="",
        confidence=template["sample_count"],
        detail=f"{template['sample_count']} accepted / {len(images)} submitted",
    )

    return jsonify(
        {
            "saved": True,
            "enrolled": True,
            "label": display_label,
            "sample_count": template["sample_count"],
            "accepted_images": template["sample_count"],
            "submitted_images": len(images),
            "rejected_images": rejected_images,
            "recommended_images": FACE_PROFILE_RECOMMENDED_IMAGES,
            "max_images": FACE_PROFILE_MAX_IMAGES,
            "message": "Face lock saved (stored in memory only). Visual check-ins will only follow the enrolled face for this account.",
        }
    ), 200


@app.delete("/face/profile")
def face_profile_delete():
    payload = request.get_json(silent=True) or {}
    user_email = _get_user_email(request, payload)
    if not user_email:
        return jsonify({"error": "missing user_email"}), 400

    subject_id = _get_subject_id(user_email)
    deleted = session_storage.delete_face_profile(subject_id)
    session_key = f"user:{user_email}"
    _clear_visual_session(session_key)

    if deleted:
        session_storage.log_activity(subject_id, "face_profile_deleted", detail="profile removed")
        rbac_services.record_activity(
            subject_id,
            "face_profile_deleted",
            mood="",
            confidence=0.0,
            detail="profile removed",
        )

    return jsonify(
        {
            "deleted": bool(deleted),
            "session_cleared": True,
            "message": "Face lock removed." if deleted else "No saved face lock for this account.",
        }
    ), 200


@app.post("/visual/predict-frame")
def visual_predict_frame():
    payload = request.get_json(silent=True) or {}
    frame = _decode_base64_image(payload.get("image", ""))
    session_key = _get_visual_session_key(payload, request)
    user_email = _get_user_email(request, payload)
    reset_session = bool(payload.get("reset_session", False))
    required_profile = None
    if user_email:
        subject_id = _get_subject_id(user_email)
        required_profile = _normalize_face_profile(session_storage.get_face_profile(subject_id))

    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    result = _predict_emotion_from_frame(
        frame,
        session_key=session_key,
        reset_session=reset_session,
        required_profile=required_profile,
    )
    status = 503 if result.get("tracking_status") == "visual_model_unavailable" else 200
    return jsonify(result), status


@app.post("/visual/save-session")
def visual_save_session():
    payload = request.get_json(silent=True) or {}
    mood = session_storage.normalize_mood(payload.get("mood", "neutral"))
    user_email = _get_user_email(request, payload)
    session_key = _get_visual_session_key(payload, request)

    try:
        confidence = float(payload.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    confidence = max(0.0, min(confidence, 1.0))

    if mood not in {session_storage.normalize_mood(item) for item in EMOTIONS}:
        mood = "neutral"

    session_id = session_storage.save_visual_session(mood, confidence)
    if session_key:
        _clear_visual_session(session_key)
    if user_email:
        anon_id = _get_anon_id(user_email)
        severity = _severity_with_threshold(mood, confidence)
        rbac_services.record_mood_entry(anon_id, mood, confidence, severity, source="visual_session")
        rbac_services.record_activity(
            anon_id,
            "visual_session",
            mood=mood,
            confidence=confidence,
            detail=str(session_id),
        )

    return jsonify(
        {
            "saved": True,
            "session_id": session_id,
            "tracking_session_cleared": bool(session_key),
            "message": "Energy video check saved.",
        }
    ), 200


@app.post("/fuse")
def fuse_latest():
    payload = request.get_json(silent=True) or {}
    user_email = _get_user_email(request, payload)
    save_session = payload.get("save_session", True)
    if isinstance(save_session, str):
        save_session = save_session.strip().lower() not in {"0", "false", "no", "off"}
    else:
        save_session = bool(save_session)

    text_result = payload.get("text_result")
    if text_result:
        session_storage.save_text_session(
            text_result.get("mood", "neutral"),
            float(text_result.get("confidence", 0.5)),
        )

    visual_result = payload.get("visual_result")
    if visual_result:
        session_storage.save_visual_session(
            visual_result.get("mood", "neutral"),
            float(visual_result.get("confidence", 0.5)),
        )

    session_saved = False
    if save_session:
        overall = session_storage.save_overall_session()
        session_saved = bool(overall)
    else:
        overall = session_storage.get_latest_session(session_storage.OVERALL_FILE)
        if not overall:
            overall = session_storage.save_overall_session()
            session_saved = bool(overall)

    if not overall:
        return jsonify({"error": "Please complete at least one text and one visual check-in first."}), 400

    trend = session_storage.analyze_trend()
    suggestions = session_storage.generate_suggestions(overall, trend)
    if _ai_mode_available():
        prompt = _build_today_support_prompt(overall, trend, suggestions)
        llm_message = session_storage.get_llm_response(prompt)
    else:
        prompt = session_storage.build_llm_prompt(overall, trend, suggestions)
        llm_message = session_storage.get_llm_response(prompt)

    llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
    llm_error = getattr(session_storage, "LAST_LLM_ERROR", "")
    mode = "ai" if llm_source in ["openrouter", "gemini", "ollama"] else "fallback"
    subject_id = _get_anon_id(user_email) if user_email else ""

    _log_response(
        "fuse_support",
        "",
        llm_message,
        llm_source,
        meta={
            "overall_mood": overall.get("overall_mood"),
            "overall_confidence": overall.get("overall_confidence"),
            "severity": overall.get("severity"),
            "trend": trend,
            "nudge": suggestions[0] if suggestions else "",
        },
        subject_id=subject_id,
    )
    if subject_id and overall:
        if session_saved:
            rbac_services.record_mood_entry(
                subject_id,
                overall.get("overall_mood", "neutral"),
                overall.get("overall_confidence", 0.0),
                overall.get("severity", "low"),
                source="fuse",
            )
        rbac_services.record_activity(
            subject_id,
            "fuse_support",
            mood=overall.get("overall_mood", ""),
            confidence=overall.get("overall_confidence", ""),
            detail=overall.get("severity", ""),
        )

    return jsonify(
        {
            "message": llm_message,
            "nudge": suggestions[0] if suggestions else "Take one slow breath and check in with yourself.",
            "mode": mode,
            "llm_source": llm_source,
            "llm_error": llm_error,
            "session_saved": session_saved,
        }
    ), 200


@app.get("/insights/weekly")
def weekly_insights():
    rows = []
    cutoff = datetime.now() - timedelta(days=7)
    user_email = _get_user_email(request, {})

    for row in session_storage.read_csv_rows(session_storage.OVERALL_FILE, limit=5000):
        ts = str(row.get("timestamp", "")).strip()
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if dt >= cutoff:
            rows.append(row)

    if _ai_mode_available():
        prompt = _build_weekly_reflection_prompt(rows)
        summary = session_storage.get_llm_response(prompt)
        llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
        llm_error = getattr(session_storage, "LAST_LLM_ERROR", "")
        mode = "ai" if llm_source in ["openrouter", "gemini", "ollama"] else "fallback"
    else:
        summary = _soft_weekly_summary(rows)
        llm_source = "fallback"
        llm_error = "ai mode disabled or provider unavailable"
        mode = "fallback"

    subject_id = _get_anon_id(user_email) if user_email else ""
    _log_response(
        "weekly_summary",
        "",
        summary,
        llm_source,
        meta={"checkins": len(rows), "mode": mode, "llm_error": llm_error},
        subject_id=subject_id,
    )
    if subject_id:
        rbac_services.record_activity(
            subject_id,
            "weekly_summary",
            mood="",
            confidence="",
            detail=str(len(rows)),
        )

    return jsonify(
        {
            "summary": summary,
            "checkins": len(rows),
            "mode": mode,
            "llm_source": llm_source,
            "llm_error": llm_error,
        }
    ), 200


@app.get("/insights/charts")
def chart_insights():
    try:
        days = int(request.args.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    try:
        weeks = int(request.args.get("weeks", 8))
    except (TypeError, ValueError):
        weeks = 8

    days = max(1, min(days, 30))
    weeks = max(1, min(weeks, 26))

    chart_data = session_storage.get_chart_data(days=days, weeks=weeks)

    return jsonify(
        {
            "days": days,
            "weeks": weeks,
            "daily": chart_data.get("daily", []),
            "weekly": chart_data.get("weekly", []),
            "stats": chart_data.get("stats", {}),
            "saved_weekly_file": chart_data.get("saved_weekly_file", ""),
        }
    ), 200


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)
