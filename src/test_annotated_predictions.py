from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from predictor import (
    create_face_detector,
    detect_all_persons,
    predict_person_attributes,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_DIR / "data" / "pexels-gustavo-fring-8770998.jpg"

DEMO_MODEL_PATH = PROJECT_DIR / "models" / "demographic_model_best.keras"
AGE_MODEL_PATH = PROJECT_DIR / "models" / "age_model_best.keras"
EMOTION_MODEL_PATH = PROJECT_DIR / "models" / "emotion_model_best.keras"

PERSON_MODEL_PATH = PROJECT_DIR / "models" / "yolo11n.pt"
DRESS_MODEL_PATH = PROJECT_DIR / "models" / "yolo11n-seg.pt"
FACE_MODEL_PATH = PROJECT_DIR / "models" / "blaze_face_short_range.tflite"

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "annotated_predictions.jpg"


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

print("Image size:", image.shape)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading demographic model...")
demo_model = tf.keras.models.load_model(DEMO_MODEL_PATH)

print("Loading age model...")
age_model = tf.keras.models.load_model(AGE_MODEL_PATH)

print("Loading emotion model...")
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)

print("Loading YOLO person model...")
person_model = YOLO(str(PERSON_MODEL_PATH))

print("Loading YOLO segmentation model...")
dress_model = YOLO(str(DRESS_MODEL_PATH))

print("Loading MediaPipe face detector...")
face_detector = create_face_detector(FACE_MODEL_PATH)


# ============================================================
# DETECT PERSONS + FACES
# ============================================================

print("\nDetecting persons and faces...")

persons = detect_all_persons(
    image,
    person_model,
    conf=0.25,
    face_detector=face_detector,
)

print("\nPersons detected:", len(persons))


# ============================================================
# RUN PREDICTIONS
# ============================================================

results = []

for index, person_data in enumerate(persons, start=1):

    print("\n" + "=" * 60)
    print(f"PERSON {index}")
    print("=" * 60)

    prediction = predict_person_attributes(
        person_data,
        demo_model,
        age_model,
        emotion_model,
        dress_model,
    )

    print("Demographic:", prediction["demographic"])
    print(
        "Demographic confidence:",
        f'{prediction["demographic_confidence"]:.2f}%'
    )

    print("Age:", prediction["age"])

    print("Emotion:", prediction["emotion"])
    print(
        "Emotion confidence:",
        f'{prediction["emotion_confidence"]:.2f}%'
    )

    print("Dress colour:", prediction["dress_color"])
    print(
        "Dress confidence:",
        f'{prediction["dress_confidence"]:.2f}%'
    )

    results.append(prediction)


# ============================================================
# DRAW RESULTS ON IMAGE
# ============================================================

annotated = image.copy()

for index, (person_data, prediction) in enumerate(
    zip(persons, results),
    start=1
):

    # --------------------------------------------------------
    # PERSON BOX
    # --------------------------------------------------------

    x1, y1, x2, y2 = person_data["person_bbox"]

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        8,
    )

    # --------------------------------------------------------
    # FACE BOX
    # --------------------------------------------------------

    if person_data["face_bbox"] is not None:

        fx1, fy1, fx2, fy2 = person_data["face_bbox"]

        fx1, fy1, fx2, fy2 = map(
            int,
            (fx1, fy1, fx2, fy2)
        )

        cv2.rectangle(
            annotated,
            (fx1, fy1),
            (fx2, fy2),
            (255, 0, 0),
            8,
        )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    demographic = prediction["demographic"]
    demographic_conf = prediction["demographic_confidence"]

    age = prediction["age"]

    emotion = prediction["emotion"]
    emotion_conf = prediction["emotion_confidence"]

    dress = prediction["dress_color"]
    dress_conf = prediction["dress_confidence"]

    lines = [
        f"Person {index}",
        f"Demographic: {demographic} ({demographic_conf:.1f}%)",
        f"Age: {age}",
        f"Emotion: {emotion} ({emotion_conf:.1f}%)",
        f"Dress: {dress} ({dress_conf:.1f}%)",
    ]

    # Text starts slightly above person box
    text_x = x1
    text_y = max(50, y1 - 180)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 4

    # Calculate background size
    text_sizes = []

    for line in lines:

        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )

        text_sizes.append(
            (text_width, text_height, baseline)
        )

    max_width = max(size[0] for size in text_sizes)

    total_height = sum(
        size[1] + size[2] + 15
        for size in text_sizes
    )

    # Background rectangle
    cv2.rectangle(
        annotated,
        (
            text_x,
            text_y - 40,
        ),
        (
            text_x + max_width + 30,
            text_y + total_height,
        ),
        (0, 0, 0),
        -1,
    )

    # Draw each line
    current_y = text_y

    for line, size in zip(lines, text_sizes):

        text_height = size[1]
        baseline = size[2]

        cv2.putText(
            annotated,
            line,
            (
                text_x + 15,
                current_y,
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        current_y += text_height + baseline + 15


# ============================================================
# SAVE IMAGE
# ============================================================

print("\nSaving annotated image...")

success = cv2.imwrite(
    str(OUTPUT_PATH),
    annotated,
)

if not success:
    raise RuntimeError(
        f"Failed to save image: {OUTPUT_PATH}"
    )

print("\n" + "=" * 70)
print("ANNOTATED PREDICTION TEST COMPLETED")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_PATH)