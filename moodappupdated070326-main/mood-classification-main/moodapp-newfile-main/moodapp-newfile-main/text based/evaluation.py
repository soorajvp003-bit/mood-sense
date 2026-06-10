import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the text emotion model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to goemotions_clean_8.csv",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Dataset column containing input text.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="emotion",
        help="Dataset column containing labels.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split size. Use 0 to score the full dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output PNG path for the confusion matrix image.",
    )
    return parser.parse_args()


def resolve_dataset(base_dir, dataset_arg):
    if dataset_arg:
        dataset_path = Path(dataset_arg).expanduser().resolve()
        if dataset_path.exists():
            return dataset_path
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    dataset_path = base_dir / "goemotions_clean_8.csv"
    if dataset_path.exists():
        return dataset_path

    raise FileNotFoundError(
        "Could not find goemotions_clean_8.csv. Pass --dataset with the CSV path."
    )


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
        "Text Model Confusion Matrix",
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
                y0 + 44,
                font,
                0.7,
                text_color,
                2,
            )
            centered_text(
                canvas,
                f"{pct:.1f}%",
                x0 + (cell_size // 2),
                y0 + 82,
                font,
                0.5,
                text_color,
                1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write confusion matrix image to {output_path}")


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    dataset_path = resolve_dataset(base_dir, args.dataset)
    model_path = base_dir / "text_emotion_model.pkl"
    vectorizer_path = base_dir / "tfidf_vectorizer.pkl"
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (base_dir / "text_confusion_matrix.png")
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(dataset_path)

    if args.text_column not in df.columns:
        raise KeyError(f"Missing text column: {args.text_column}")
    if args.label_column not in df.columns:
        raise KeyError(f"Missing label column: {args.label_column}")

    x_text = df[args.text_column].fillna("").astype(str)
    y = df[args.label_column].fillna("").astype(str).str.strip().str.lower()

    if args.val_size and args.val_size > 0:
        _, x_eval, _, y_eval = train_test_split(
            x_text,
            y,
            test_size=args.val_size,
            random_state=42,
            stratify=y,
        )
        split_name = f"holdout ({len(y_eval)} samples)"
    else:
        x_eval = x_text
        y_eval = y
        split_name = f"full dataset ({len(y_eval)} samples)"

    x_eval_vec = vectorizer.transform(x_eval)
    y_pred = model.predict(x_eval_vec)
    classes = [str(label).strip().lower() for label in getattr(model, "classes_", sorted(y.unique()))]
    cm = confusion_matrix(y_eval, y_pred, labels=classes)
    accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_eval)))

    save_confusion_matrix_image(cm, classes, output_path)

    print(f"Dataset path: {dataset_path}")
    print(f"Text column: {args.text_column}")
    print(f"Label column: {args.label_column}")
    print(f"Labels: {classes}")
    print(f"Evaluation split: {split_name}")
    print(f"\nValidation accuracy: {accuracy:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_eval, y_pred, labels=classes, target_names=classes, digits=4))
    print(f"\nConfusion matrix image saved: {output_path}")


if __name__ == "__main__":
    main()
