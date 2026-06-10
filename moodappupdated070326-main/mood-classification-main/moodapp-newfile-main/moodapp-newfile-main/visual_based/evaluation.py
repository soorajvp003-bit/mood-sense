import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

IMG_SIZE = 48
DEFAULT_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
    "contempt",
]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EMOTION_ALIASES = {
    "surprise": ["suprise"],
}


def normalize_name(value):
    return str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def resolve_class_dir(dataset_path, emotion):
    direct_path = dataset_path / emotion
    if direct_path.exists():
        return direct_path

    accepted_names = {normalize_name(emotion)}
    for alias in EMOTION_ALIASES.get(emotion, []):
        accepted_names.add(normalize_name(alias))

    for child in dataset_path.iterdir():
        if child.is_dir() and normalize_name(child.name) in accepted_names:
            return child

    raise FileNotFoundError(
        f"Missing class folder for '{emotion}' under dataset root: {dataset_path}"
    )


def is_emotion_root(path, emotions):
    if not path.exists():
        return False
    try:
        for emotion in emotions:
            resolve_class_dir(path, emotion)
    except FileNotFoundError:
        return False
    return True


def normalize_dataset_root(path, emotions):
    if is_emotion_root(path, emotions):
        return path

    for split_name in ["test", "validation", "val", "train_balanced", "train"]:
        candidate = path / split_name
        if is_emotion_root(candidate, emotions):
            return candidate

    raise FileNotFoundError(
        f"Dataset root does not contain expected emotion folders: {path}"
    )


def resolve_dataset(base_dir, dataset_arg, emotions):
    if dataset_arg:
        dataset = Path(dataset_arg).expanduser().resolve()
        if dataset.exists():
            return normalize_dataset_root(dataset, emotions)
        raise FileNotFoundError(f"Dataset path not found: {dataset}")

    candidates = [
        Path(base_dir) / "fer2013" / "test",
        Path(base_dir) / "fer2013" / "validation",
        Path(base_dir) / "fer2013" / "train_balanced",
        Path(base_dir) / "fer2013" / "train",
        Path(base_dir).parent / "mood-classification-main" / "visual" / "fer2013" / "test",
        Path(base_dir).parent / "mood-classification-main" / "visual" / "fer2013" / "validation",
        Path(base_dir).parent / "mood-classification-main" / "visual" / "fer2013" / "train_balanced",
        Path(base_dir).parent / "mood-classification-main" / "visual" / "fer2013" / "train",
    ]
    for candidate in candidates:
        if candidate.exists():
            return normalize_dataset_root(candidate.resolve(), emotions)
    raise FileNotFoundError("Could not find dataset directory.")


def load_labels(base_dir, model):
    labels_path = Path(base_dir) / "emotion_labels.json"
    expected = int(model.output_shape[-1])
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if (
                isinstance(labels, list)
                and len(labels) == expected
                and all(isinstance(item, str) for item in labels)
            ):
                return [item.strip().lower() for item in labels]
        except Exception:
            pass
    return DEFAULT_EMOTIONS[:expected]


def preprocess_image(file_path, input_channels):
    gray = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    if input_channels == 3:
        return np.stack([gray] * 3, axis=-1)
    return np.expand_dims(gray, axis=-1)


def load_data(dataset_path, emotions, max_per_class, input_channels):
    x_data = []
    y_data = []
    counts = {}

    for idx, emotion in enumerate(emotions):
        class_dir = resolve_class_dir(dataset_path, emotion)

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        files.sort()
        if max_per_class and max_per_class > 0:
            files = files[:max_per_class]

        for file_path in files:
            image = preprocess_image(file_path, input_channels)
            if image is None:
                continue
            x_data.append(image)
            y_data.append(idx)

        counts[emotion] = len(files)

    x_data = np.asarray(x_data, dtype="float32")
    y_data = np.asarray(y_data, dtype="int32")
    return x_data, y_data, counts


def centered_text(image, text, center_x, center_y, font, scale, color, thickness):
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
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
    cell_size = 118
    left_pad = 220
    top_pad = 160
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
        "Confusion Matrix",
        width // 2,
        36,
        font,
        0.95,
        (25, 25, 25),
        2,
    )
    centered_text(
        canvas,
        "Predicted label",
        left_pad + ((cols * cell_size) // 2),
        90,
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
        centered_text(
            canvas,
            str(label),
            x_center,
            122,
            font,
            0.5,
            (40, 40, 40),
            1,
        )

    for row_idx, label in enumerate(labels):
        y_center = top_pad + (row_idx * cell_size) + (cell_size // 2)
        centered_text(
            canvas,
            str(label),
            left_pad - 86,
            y_center,
            font,
            0.55,
            (40, 40, 40),
            1,
        )

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

            centered_text(
                canvas,
                str(count),
                x0 + (cell_size // 2),
                y0 + 42,
                font,
                0.7,
                text_color,
                2,
            )
            centered_text(
                canvas,
                f"{pct:.1f}%",
                x0 + (cell_size // 2),
                y0 + 78,
                font,
                0.5,
                text_color,
                1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write confusion matrix image to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate visual emotion model.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output PNG path for the confusion matrix image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "emotion_model.keras"
    model = tf.keras.models.load_model(model_path, compile=False)

    emotions = load_labels(base_dir, model)
    input_channels = int(model.input_shape[-1])
    dataset_path = resolve_dataset(base_dir, args.dataset, emotions)
    print(f"Dataset path: {dataset_path}")
    print(f"Labels: {emotions}")
    print(f"Model input channels: {input_channels}")

    x_data, y_data, counts = load_data(
        dataset_path,
        emotions,
        args.max_per_class,
        input_channels,
    )
    print(f"Loaded samples: {len(x_data)}")
    print(f"Per-class counts: {counts}")

    if args.val_size and args.val_size > 0:
        _, x_eval, _, y_eval = train_test_split(
            x_data,
            y_data,
            test_size=args.val_size,
            random_state=42,
            stratify=y_data,
        )
        print(f"Evaluation split: holdout ({len(y_eval)} samples)")
    else:
        x_eval = x_data
        y_eval = y_data
        print(f"Evaluation split: full dataset ({len(y_eval)} samples)")

    probs = model.predict(x_eval, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc = float(np.mean(y_pred == y_eval))
    cm = confusion_matrix(y_eval, y_pred, labels=np.arange(len(emotions)))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (base_dir / "confusion_matrix.png")
    )
    save_confusion_matrix_image(cm, emotions, output_path)

    print(f"\nValidation accuracy: {acc:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_eval, y_pred, target_names=emotions, digits=4))
    print(f"\nConfusion matrix image saved: {output_path}")


if __name__ == "__main__":
    main()
