import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .dress_color import detect_dress_color_segmentation

# ============================================================
# LABELS
# ============================================================

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

DEMOGRAPHIC_LABELS = {
    0: "East Asian",
    1: "Indian",
    2: "Black",
    3: "White",
    4: "Middle Eastern",
    5: "Latino/Hispanic",
    6: "Southeast Asian"
}


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_rgb_image(image):
    """
    Convert PIL or NumPy image into RGB NumPy array.
    """

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    raise TypeError(
        "Unsupported image type. Use a PIL Image or NumPy array."
    )


# ============================================================
# CREATE MEDIAPIPE FACE DETECTOR
# ============================================================

def create_face_detector(model_path):
    """
    Create MediaPipe Tasks FaceDetector.
    """

    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

    detector = vision.FaceDetector.create_from_options(
        options
    )

    return detector


# ============================================================
# SINGLE LARGEST PERSON
# ============================================================

def detect_person_crop(
    image,
    person_model,
    conf=0.25
):
    """
    Detect the largest person in an image.

    Returns:
        person_crop
        person_bbox
    """

    rgb_image = prepare_rgb_image(image)

    results = person_model.predict(
        rgb_image,
        conf=conf,
        verbose=False
    )

    best_box = None
    best_area = 0

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i, cls_id in enumerate(
            boxes.cls.cpu().numpy()
        ):

            # COCO class 0 = person
            if int(cls_id) != 0:
                continue

            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )

            area = (
                max(0, x2 - x1)
                * max(0, y2 - y1)
            )

            if area > best_area:
                best_area = area
                best_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = best_box

    height, width = rgb_image.shape[:2]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    person_crop = rgb_image[
        y1:y2,
        x1:x2
    ]

    if person_crop.size == 0:
        return None, None

    return person_crop, (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# FACE DETECTION INSIDE PERSON
# ============================================================

def detect_face_from_person(
    person_crop,
    face_detector
):
    """
    Detect the largest face inside one person crop.

    Returns:
        face_crop
        face_bbox

    face_bbox is relative to person_crop.
    """

    if person_crop is None:
        return None, None

    if person_crop.size == 0:
        return None, None

    if face_detector is None:
        return None, None

    rgb_person = prepare_rgb_image(
        person_crop
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_person
    )

    # Detect faces
    detection_result = face_detector.detect(
        mp_image
    )

    if not detection_result.detections:
        return None, None

    height, width = rgb_person.shape[:2]

    best_detection = None
    best_area = 0

    for detection in detection_result.detections:

        bbox = detection.bounding_box

        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )

        area = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )

        if area > best_area:
            best_area = area
            best_detection = (
                x1,
                y1,
                x2,
                y2
            )

    if best_detection is None:
        return None, None

    x1, y1, x2, y2 = best_detection

    face_crop = rgb_person[
        y1:y2,
        x1:x2
    ]

    if face_crop.size == 0:
        return None, None

    return face_crop, (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# DETECT ALL PERSONS + FACES
# ============================================================

def detect_all_persons(
    image,
    person_model,
    conf=0.25,
    face_detector=None
):
    """
    Detect all people.

    For every person:
        - person crop
        - person bounding box
        - face crop
        - face bounding box
    """

    rgb_image = prepare_rgb_image(
        image
    )

    results = person_model.predict(
        rgb_image,
        conf=conf,
        verbose=False
    )

    persons = []

    height, width = rgb_image.shape[:2]

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i, cls_id in enumerate(
            boxes.cls.cpu().numpy()
        ):

            # COCO class 0 = person
            if int(cls_id) != 0:
                continue

            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1 = max(
                0,
                min(x1, width)
            )

            x2 = max(
                0,
                min(x2, width)
            )

            y1 = max(
                0,
                min(y1, height)
            )

            y2 = max(
                0,
                min(y2, height)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = rgb_image[
                y1:y2,
                x1:x2
            ]

            if person_crop.size == 0:
                continue

            # Detect face inside this person
            face_crop, local_face_bbox = (
                detect_face_from_person(
                    person_crop,
                    face_detector
                )
            )

            face_bbox = None

            if local_face_bbox is not None:

                fx1, fy1, fx2, fy2 = (
                    local_face_bbox
                )

                # Convert local face coordinates
                # to original-image coordinates
                face_bbox = (
                    x1 + fx1,
                    y1 + fy1,
                    x1 + fx2,
                    y1 + fy2
                )

            persons.append(
                {
                    "person_crop": person_crop,

                    "person_bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "face_crop": face_crop,

                    "face_bbox": face_bbox
                }
            )

    # Sort from left to right
    persons.sort(
        key=lambda item:
        item["person_bbox"][0]
    )

    return persons


# ============================================================
# DEMOGRAPHIC PREDICTION
# ============================================================

def predict_demographic(
    image,
    demo_model
):
    """
    Predict FairFace demographic category.

    IMPORTANT:
    This is a demographic/race category,
    not actual nationality.
    """

    rgb_image = prepare_rgb_image(
        image
    )

    rgb_128 = cv2.resize(
        rgb_image,
        (128, 128)
    )

    rgb_128 = (
        rgb_128.astype(
            np.float32
        ) / 255.0
    )

    rgb_input = np.expand_dims(
        rgb_128,
        axis=0
    )

    demo_probs = demo_model.predict(
        rgb_input,
        verbose=0
    )[0]

    demo_id = int(
        np.argmax(demo_probs)
    )

    demo_label = DEMOGRAPHIC_LABELS.get(
        demo_id,
        "Unknown"
    )

    demo_confidence = float(
        demo_probs[demo_id] * 100
    )

    return (
        demo_label,
        demo_confidence
    )


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(
    face_image,
    age_model
):
    """
    Predict age from face crop.
    """

    rgb_face = prepare_rgb_image(
        face_image
    )

    face_128 = cv2.resize(
        rgb_face,
        (128, 128)
    )

    face_128 = (
        face_128.astype(
            np.float32
        ) / 255.0
    )

    face_input = np.expand_dims(
        face_128,
        axis=0
    )

    age_prediction = age_model.predict(
        face_input,
        verbose=0
    )[0][0]

    predicted_age = max(
        0,
        int(
            round(
                float(age_prediction)
            )
        )
    )

    return predicted_age


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(
    face_image,
    emotion_model
):
    """
    Predict emotion from face crop.
    """

    rgb_face = prepare_rgb_image(
        face_image
    )

    gray_face = cv2.cvtColor(
        rgb_face,
        cv2.COLOR_RGB2GRAY
    )

    gray_48 = cv2.resize(
        gray_face,
        (48, 48)
    )

    gray_48 = (
        gray_48.astype(
            np.float32
        ) / 255.0
    )

    gray_input = np.expand_dims(
        gray_48,
        axis=(0, -1)
    )

    emotion_probs = emotion_model.predict(
        gray_input,
        verbose=0
    )[0]

    emotion_id = int(
        np.argmax(emotion_probs)
    )

    emotion_label = EMOTION_LABELS[
        emotion_id
    ]

    emotion_confidence = float(
        emotion_probs[emotion_id] * 100
    )

    return (
        emotion_label,
        emotion_confidence
    )


# ============================================================
# ONE PERSON PREDICTION
# ============================================================

def predict_person_attributes(
    person_data,
    demo_model,
    age_model,
    emotion_model,
    dress_model
):
    """
    Predict attributes for one detected person.

    Demographic:
        person crop

    Age:
        face crop

    Emotion:
        face crop

    Dress colour:
        person/body crop
    """

    person_crop = person_data[
        "person_crop"
    ]

    face_crop = person_data[
        "face_crop"
    ]

    # --------------------------------------------------------
    # DEMOGRAPHIC
    # --------------------------------------------------------

    demographic, demographic_confidence = (
        predict_demographic(
            person_crop,
            demo_model
        )
    )

    # --------------------------------------------------------
    # AGE + EMOTION
    # --------------------------------------------------------

    if face_crop is not None:

        predicted_age = predict_age(
            face_crop,
            age_model
        )

        (
            emotion_label,
            emotion_confidence
        ) = predict_emotion(
            face_crop,
            emotion_model
        )

    else:

        predicted_age = None

        emotion_label = (
            "Face not detected"
        )

        emotion_confidence = 0.0

    # --------------------------------------------------------
    # DRESS COLOUR
    # --------------------------------------------------------

    dress_color, dress_confidence = (
        detect_dress_color_segmentation(
            person_crop,
            dress_model
        )
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "demographic": demographic,

        "demographic_confidence":
            demographic_confidence,

        "age": predicted_age,

        "emotion": emotion_label,

        "emotion_confidence":
            emotion_confidence,

        "dress_color": dress_color,

        "dress_confidence":
            float(dress_confidence),

        "face_detected":
            face_crop is not None
    }


# ============================================================
# OLD SINGLE-PERSON FUNCTION
# ============================================================

def predict_all_attributes(
    image,
    demo_model,
    age_model,
    emotion_model,
    dress_model
):
    """
    Compatibility function for the old application.

    This function works on one image/person.
    For the new multi-person system,
    use detect_all_persons() followed by
    predict_person_attributes().
    """

    rgb_image = prepare_rgb_image(
        image
    )

    demographic, demographic_confidence = (
        predict_demographic(
            rgb_image,
            demo_model
        )
    )

    dress_color, dress_confidence = (
        detect_dress_color_segmentation(
            rgb_image,
            dress_model
        )
    )

    return {
        "demographic": demographic,

        "demographic_confidence":
            demographic_confidence,

        "age": None,

        "emotion":
            "Face detection required",

        "emotion_confidence": 0.0,

        "dress_color": dress_color,

        "dress_confidence":
            float(dress_confidence)
    }