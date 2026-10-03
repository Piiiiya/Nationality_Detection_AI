import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# Allow importing predictor.py from the same src folder
sys.path.insert(0, str(Path(__file__).parent))

import predictor


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent

IMAGE_PATH = PROJECT_DIR / "data" / "pexels-gustavo-fring-8770998.jpg"
PERSON_MODEL_PATH = PROJECT_DIR / "models" / "yolo11n.pt"
FACE_MODEL_PATH = PROJECT_DIR / "models" / "blaze_face_short_range.tflite"


# --------------------------------------------------
# Load image
# --------------------------------------------------

print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image size:", image.size)


# --------------------------------------------------
# Load YOLO person detector
# --------------------------------------------------

print("Loading YOLO person detector...")

person_model = YOLO(str(PERSON_MODEL_PATH))


# --------------------------------------------------
# Load MediaPipe face detector
# --------------------------------------------------

print("Loading MediaPipe face detector...")

face_detector = predictor.create_face_detector(
    str(FACE_MODEL_PATH)
)


# --------------------------------------------------
# Detect all persons
# --------------------------------------------------

print("Detecting persons and faces...")

persons = predictor.detect_all_persons(
    image,
    person_model,
    conf=0.25,
    face_detector=face_detector
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("MULTI-PERSON DETECTION RESULT")
print("=" * 60)

print("Persons detected:", len(persons))

for i, person in enumerate(persons, start=1):

    print()
    print(f"Person {i}")

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


# --------------------------------------------------
# Close detector
# --------------------------------------------------

face_detector.close()

print()
print("Test completed successfully.")