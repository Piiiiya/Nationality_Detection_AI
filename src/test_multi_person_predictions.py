import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model

sys.path.insert(0, str(Path(__file__).parent))

import predictor


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).parent.parent

IMAGE_PATH = (
    PROJECT_DIR
    / "data"
    / "pexels-gustavo-fring-8770998.jpg"
)

DEMO_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "demographic_model_best.keras"
)

AGE_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "age_model_best.keras"
)

EMOTION_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "emotion_model_best.keras"
)

PERSON_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo11n.pt"
)

DRESS_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "yolo11n-seg.pt"
)

FACE_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "blaze_face_short_range.tflite"
)


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Image size:",
    image.size
)


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("Loading demographic model...")

demo_model = load_model(
    DEMO_MODEL_PATH
)

print("Loading age model...")

age_model = load_model(
    AGE_MODEL_PATH
)

print("Loading emotion model...")

emotion_model = load_model(
    EMOTION_MODEL_PATH
)

print("Loading YOLO person model...")

person_model = YOLO(
    str(PERSON_MODEL_PATH)
)

print("Loading YOLO segmentation model...")

dress_model = YOLO(
    str(DRESS_MODEL_PATH)
)

print("Loading MediaPipe face detector...")

face_detector = predictor.create_face_detector(
    FACE_MODEL_PATH
)


# ============================================================
# DETECT ALL PERSONS
# ============================================================

print()
print("Detecting persons and faces...")

persons = predictor.detect_all_persons(
    image,
    person_model,
    conf=0.25,
    face_detector=face_detector
)


# ============================================================
# BASIC RESULT
# ============================================================

print()
print("=" * 70)
print("MULTI-PERSON PREDICTION")
print("=" * 70)

print(
    "Persons detected:",
    len(persons)
)


# ============================================================
# PREDICT EACH PERSON
# ============================================================

for index, person in enumerate(
    persons,
    start=1
):

    print()
    print("-" * 70)
    print(f"PERSON {index}")
    print("-" * 70)

    print(
        "Person bounding box:",
        person["person_bbox"]
    )

    print(
        "Face detected:",
        person["face_crop"] is not None
    )

    print(
        "Face bounding box:",
        person["face_bbox"]
    )

    if person["face_crop"] is None:

        print(
            "Skipping age and emotion "
            "because face was not detected."
        )

        continue

    print()
    print("Running predictions...")

    prediction = (
        predictor.predict_person_attributes(
            person,
            demo_model,
            age_model,
            emotion_model,
            dress_model
        )
    )

    print()
    print(
        "Demographic:",
        prediction["demographic"]
    )

    print(
        "Demographic confidence:",
        f'{prediction["demographic_confidence"]:.2f}%'
    )

    print(
        "Age:",
        prediction["age"]
    )

    print(
        "Emotion:",
        prediction["emotion"]
    )

    print(
        "Emotion confidence:",
        f'{prediction["emotion_confidence"]:.2f}%'
    )

    print(
        "Dress colour:",
        prediction["dress_color"]
    )

    print(
        "Dress confidence:",
        f'{prediction["dress_confidence"]:.2f}%'
    )


# ============================================================
# CLOSE FACE DETECTOR
# ============================================================

face_detector.close()

print()
print("=" * 70)
print("Prediction test completed successfully.")
print("=" * 70)