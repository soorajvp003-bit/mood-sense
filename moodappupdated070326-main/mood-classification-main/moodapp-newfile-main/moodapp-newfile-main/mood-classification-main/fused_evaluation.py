import argparse
import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4
CANONICAL_EMOTIONS = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
VISUAL_DEFAULT_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt",
]
EMOTION_ALIASES = {
    "surprise": ["suprise"],
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Approximate fused text+visual confusion matrix using same-label paired samples."
    )
    parser.add_argument("--text-dataset", type=str, required=True)
    parser.add_argument("--visual-dataset", type=str, required=True)
    parser.add_argument("--text-val-size", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-class", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output PNG path. Default: <this_folder>/visual/fused_confusion_matrix.png",
    )
    return parser.parse_args()


def normalize_name(value):
    return str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def resolve_class_dir(dataset_path, emotion):
    direct_path = dataset_path / emotion
    if direct_path.exists():
        return direct_path

    accepted = {normalize_name(emotion)}
    for alias in EMOTION_ALIASES.get(emotion, []):
        accepted.add(normalize_name(alias))

    for child in dataset_path.iterdir():
        if child.is_dir() and normalize_name(child.name) in accepted:
            return child

    raise FileNotFoundError(
        f"Missing class folder for '{emotion}' under dataset root: {dataset_path}"
    )


def normalize_visual_dataset_root(path):
    try:
        for emotion in CANONICAL_EMOTIONS:
            resolve_class_dir(path, emotion)
        return path
    except FileNotFoundError:
        pass

    for split_name in ["test", "validation", "val", "train_balanced", "train"]:
        candidate = path / split_name
        if not candidate.exists():
            continue
        try:
            for emotion in CANONICAL_EMOTIONS:
                resolve_class_dir(candidate, emotion)
            return candidate
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Dataset root does not contain expected emotion folders: {path}"
    )


def load_visual_label_names(model_dir, model):
    labels_path = model_dir / "emotion_labels.json"
    expected = int(model.output_shape[-1])
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as handle:
                labels = json.load(handle)
            if (
                isinstance(labels, list)
                and len(labels) == expected
                and all(isinstance(item, str) for item in labels)
            ):
                return [str(item).strip().lower() for item in labels]
        except Exception:
            pass
    return VISUAL_DEFAULT_EMOTIONS[:expected]


def collect_visual_files(dataset_root):
    by_label = {}
    for emotion in CANONICAL_EMOTIONS:
        class_dir = resolve_class_dir(dataset_root, emotion)
        files = [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]
        files.sort()
        by_label[emotion] = files
    return by_label


def load_text_eval_split(dataset_path, val_size):
    df = pd.read_csv(dataset_path)
    x_text = df["text"].fillna("").astype(str)
    y = df["emotion"].fillna("").astype(str).str.strip().str.lower()
    keep_mask = y.isin(CANONICAL_EMOTIONS)
    x_text = x_text[keep_mask]
    y = y[keep_mask]

    if val_size and val_size > 0:
        _, x_eval, _, y_eval = train_test_split(
            x_text,
            y,
            test_size=val_size,
            random_state=SEED,
            stratify=y,
        )
        split_name = f"holdout ({len(y_eval)} samples)"
    else:
        x_eval = x_text
        y_eval = y
        split_name = f"full dataset ({len(y_eval)} samples)"

    by_label = {emotion: [] for emotion in CANONICAL_EMOTIONS}
    for text_value, label in zip(x_eval.tolist(), y_eval.tolist()):
        by_label[label].append(text_value)

    return by_label, split_name


def pair_same_label_samples(text_by_label, visual_by_label, max_pairs_per_class):
    rng = random.Random(SEED)
    pairs = []
    counts = {}

    for emotion in CANONICAL_EMOTIONS:
        text_items = list(text_by_label.get(emotion, []))
        visual_items = list(visual_by_label.get(emotion, []))
        rng.shuffle(text_items)
        rng.shuffle(visual_items)

        pair_count = min(len(text_items), len(visual_items))
        if max_pairs_per_class and max_pairs_per_class > 0:
            pair_count = min(pair_count, max_pairs_per_class)

        counts[emotion] = {
            "text_eval": len(text_items),
            "visual_eval": len(visual_items),
            "paired": pair_count,
        }

        for idx in range(pair_count):
            pairs.append((emotion, text_items[idx], visual_items[idx]))

    if not pairs:
        raise RuntimeError("No fused pairs could be created from the provided datasets.")

    return pairs, counts


def preprocess_visual_image(file_path, input_shape):
    height = int(input_shape[1])
    width = int(input_shape[2])
    channels = int(input_shape[3])

    gray = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    image = cv2.resize(gray, (width, height)).astype("float32") / 255.0
    if channels == 3:
        image = np.stack([image] * 3, axis=-1)
    else:
        image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)


def predict_text(model, vectorizer, text_value):
    text_vector = vectorizer.transform([text_value])
    probabilities = model.predict_proba(text_vector)[0]
    best_idx = int(np.argmax(probabilities))
    label = str(model.classes_[best_idx]).strip().lower()
    confidence = float(probabilities[best_idx])
    return label, confidence


def predict_visual(model, label_names, image_path):
    image = preprocess_visual_image(image_path, model.input_shape)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    probabilities = model.predict(image, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))
    label = str(label_names[best_idx]).strip().lower()
    confidence = float(probabilities[best_idx])
    return label, confidence


def fuse_predictions(text_label, text_conf, visual_label, visual_conf):
    text_score = float(text_conf) * TEXT_WEIGHT
    visual_score = float(visual_conf) * VISUAL_WEIGHT
    fused_label = text_label if text_score >= visual_score else visual_label
    fused_conf = round(text_score + visual_score, 4)
    return fused_label, fused_conf


def centered_text(image, text, center_x, center_y, font, scale, color, thickness):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = int(center_x - (text_size[0] / 2))
    text_y = int(center_y + (text_size[1] / 2))
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def save_confusion_matrix_image(cm, labels, output_path):
    rows, cols = cm.shape
    cell_size = 120
    left_pad = 220
    top_pad = 185
    right_pad = 40
    bottom_pad = 60
    width = left_pad + (cols * cell_size) + right_pad
    height = top_pad + (rows * cell_size) + bottom_pad

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm.astype(np.float32),
        row_totals,
        out=np.zeros_like(cm, dtype=np.float32),
        where=row_totals != 0,
    )

    centered_text(
        canvas,
        "Fused Confusion Matrix",
        width // 2,
        34,
        font,
        0.95,
        (25, 25, 25),
        2,
    )
    centered_text(
        canvas,
        "Approximate paired evaluation: same-label text + visual samples",
        width // 2,
        70,
        font,
        0.52,
        (80, 80, 80),
        1,
    )
    centered_text(
        canvas,
        "Predicted label",
        left_pad + ((cols * cell_size) // 2),
        120,
        font,
        0.7,
        (60, 60, 60),
        2,
    )
    cv2.putText(
        canvas,
        "True label",
        (32, top_pad + ((rows * cell_size) // 2)),
        font,
        0.7,
        (60, 60, 60),
        2,
        lineType=cv2.LINE_AA,
    )

    for col_idx, label in enumerate(labels):
        x_center = left_pad + (col_idx * cell_size) + (cell_size // 2)
        centered_text(canvas, label, x_center, 152, font, 0.5, (40, 40, 40), 1)

    for row_idx, label in enumerate(labels):
        y_center = top_pad + (row_idx * cell_size) + (cell_size // 2)
        centered_text(canvas, label, left_pad - 86, y_center, font, 0.55, (40, 40, 40), 1)

        for col_idx in range(cols):
            x0 = left_pad + (col_idx * cell_size)
            y0 = top_pad + (row_idx * cell_size)
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            intensity = int(round(float(normalized[row_idx, col_idx]) * 255.0))
            heat = cv2.applyColorMap(
                np.array([[intensity]], dtype=np.uint8),
                cv2.COLORMAP_VIRIDIS,
            )[0, 0]
            color = tuple(int(channel) for channel in heat.tolist())
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness=-1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 220, 220), thickness=1)

            brightness = int(np.mean(heat))
            text_color = (245, 245, 245) if brightness < 120 else (20, 20, 20)
            count = int(cm[row_idx, col_idx])
            pct = float(normalized[row_idx, col_idx]) * 100.0

            centered_text(canvas, str(count), x0 + (cell_size // 2), y0 + 44, font, 0.7, text_color, 2)
            centered_text(canvas, f"{pct:.1f}%", x0 + (cell_size // 2), y0 + 82, font, 0.5, text_color, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write confusion matrix image to {output_path}")


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent

    text_model_path = project_dir / "text based" / "text_emotion_model.pkl"
    vectorizer_path = project_dir / "text based" / "tfidf_vectorizer.pkl"
    visual_model_path = project_dir / "visual_based" / "emotion_model.keras"

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (base_dir / "visual" / "fused_confusion_matrix.png")
    )

    text_dataset = Path(args.text_dataset).expanduser().resolve()
    visual_dataset = normalize_visual_dataset_root(
        Path(args.visual_dataset).expanduser().resolve()
    )

    text_model = joblib.load(text_model_path)
    vectorizer = joblib.load(vectorizer_path)
    visual_model = tf.keras.models.load_model(visual_model_path, compile=False)
    visual_label_names = load_visual_label_names(visual_model_path.parent, visual_model)

    text_by_label, split_name = load_text_eval_split(text_dataset, args.text_val_size)
    visual_by_label = collect_visual_files(visual_dataset)
    pairs, pair_counts = pair_same_label_samples(
        text_by_label,
        visual_by_label,
        args.max_pairs_per_class,
    )

    y_true = []
    y_pred = []
    fused_confidences = []

    for true_label, text_value, image_path in pairs:
        text_label, text_conf = predict_text(text_model, vectorizer, text_value)
        visual_label, visual_conf = predict_visual(visual_model, visual_label_names, image_path)
        fused_label, fused_conf = fuse_predictions(
            text_label,
            text_conf,
            visual_label,
            visual_conf,
        )

        y_true.append(true_label)
        y_pred.append(fused_label)
        fused_confidences.append(fused_conf)

    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_EMOTIONS)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    average_confidence = float(np.mean(fused_confidences))
    save_confusion_matrix_image(cm, CANONICAL_EMOTIONS, output_path)

    print(f"Text dataset: {text_dataset}")
    print(f"Visual dataset: {visual_dataset}")
    print(f"Evaluation split: {split_name}")
    print("Pair counts by class:")
    for emotion in CANONICAL_EMOTIONS:
        info = pair_counts[emotion]
        print(
            f"  {emotion:9s} text={info['text_eval']:4d} "
            f"visual={info['visual_eval']:4d} paired={info['paired']:4d}"
        )

    print(f"\nTotal fused pairs: {len(y_true)}")
    print(f"Average fused confidence: {average_confidence:.4f}")
    print(f"Fusion rule: text={TEXT_WEIGHT:.1f}, visual={VISUAL_WEIGHT:.1f}")
    print("\nConfusion matrix:")
    print(cm)
    print(f"\nApproximate fused accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CANONICAL_EMOTIONS,
            target_names=CANONICAL_EMOTIONS,
            digits=4,
        )
    )
    print(f"\nFused confusion matrix image saved: {output_path}")


if __name__ == "__main__":
    main()
