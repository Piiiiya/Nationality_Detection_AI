import cv2
import numpy as np
from PIL import Image


def detect_dress_color_segmentation(image, model, conf=0.25):
    """
    Detect the person using YOLO segmentation and estimate
    clothing colour from the lower portion of the person mask.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    if image is None or image.size == 0:
        return "Not visible", 0.0

    if len(image.shape) != 3 or image.shape[2] != 3:
        return "Not visible", 0.0

    results = model.predict(
        source=image,
        conf=conf,
        verbose=False
    )

    result = results[0]

    if result.masks is None:
        return "Not visible", 0.0

    person_indices = []

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])

        # COCO class 0 = person
        if class_id == 0:
            person_indices.append(i)

    if not person_indices:
        return "Not visible", 0.0

    # Select the most confident person
    best_index = max(
        person_indices,
        key=lambda i: float(result.boxes[i].conf[0])
    )

    # Get segmentation mask
    mask = result.masks.data[best_index].cpu().numpy()

    h, w = image.shape[:2]

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    person_mask = mask > 0.5

    # Person bounding box
    box = result.boxes[best_index]

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].cpu().numpy()
    )

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    person_height = y2 - y1

    if person_height < 80:
        return "Not visible", 0.0

    # Lower portion of person's body
    clothing_start = int(
        y1 + person_height * 0.40
    )

    clothing_region = np.zeros_like(person_mask)

    clothing_region[
        clothing_start:y2,
        x1:x2
    ] = True

    # Only actual person pixels
    clothing_mask = person_mask & clothing_region

    pixel_count = np.sum(clothing_mask)

    if pixel_count < 500:
        return "Not visible", 0.0

    # Extract clothing/person pixels
    clothing_pixels = image[clothing_mask]

    hsv_pixels = cv2.cvtColor(
        clothing_pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)

    H = hsv_pixels[:, 0]
    S = hsv_pixels[:, 1]
    V = hsv_pixels[:, 2]

    total = len(V)

    # -------------------------
    # BLACK / DARK
    # -------------------------
    black_mask = (
        (V < 105) &
        (S > 20)
    )

    black_percentage = (
        np.sum(black_mask) / total
    ) * 100

    if black_percentage >= 35:
        return "Black", round(float(black_percentage), 2)

    # -------------------------
    # WHITE
    # -------------------------
    white_mask = (
        (S < 45) &
        (V >= 180)
    )

    white_percentage = (
        np.sum(white_mask) / total
    ) * 100

    if white_percentage >= 35:
        return "White", round(float(white_percentage), 2)

    # -------------------------
    # GREY
    # -------------------------
    gray_mask = (
        (S < 50) &
        (V >= 70) &
        (V < 180)
    )

    gray_percentage = (
        np.sum(gray_mask) / total
    ) * 100

    if gray_percentage >= 35:
        return "Grey", round(float(gray_percentage), 2)

    # -------------------------
    # COLOURED CLOTHING
    # -------------------------
    color_masks = {
        "Red": ((H < 10) | (H >= 170)) & (S > 60),
        "Orange": (H >= 10) & (H < 22) & (S > 60),
        "Yellow": (H >= 22) & (H < 35) & (S > 60),
        "Green": (H >= 35) & (H < 85) & (S > 60),
        "Blue": (H >= 85) & (H < 130) & (S > 60),
        "Purple": (H >= 130) & (H < 160) & (S > 60),
        "Pink": (H >= 160) & (H < 170) & (S > 60),
    }

    percentages = {}

    for color, color_mask in color_masks.items():
        percentages[color] = (
            np.sum(color_mask) / total
        ) * 100

    dominant_color = max(
        percentages,
        key=percentages.get
    )

    dominant_percentage = percentages[dominant_color]

    if dominant_percentage < 25:
        return "Not visible", round(
            float(dominant_percentage), 2
        )

    return dominant_color, round(
        float(dominant_percentage), 2
    )