
import os

# ============================================================
# CPU CONFIGURATION
# Set these BEFORE importing TensorFlow, PyTorch or MediaPipe
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ============================================================
# IMPORTS WITH CLOUD DIAGNOSTIC CHECKPOINTS
# ============================================================

from pathlib import Path
import sys

print("DIAGNOSTIC: Starting app imports", flush=True)

print("DIAGNOSTIC: Importing OpenCV...", flush=True)
import cv2
print("DIAGNOSTIC: OpenCV imported successfully", flush=True)

print("DIAGNOSTIC: Importing NumPy...", flush=True)
import numpy as np
print("DIAGNOSTIC: NumPy imported successfully", flush=True)

print("DIAGNOSTIC: Importing Streamlit...", flush=True)
import streamlit as st
print("DIAGNOSTIC: Streamlit imported successfully", flush=True)

print("DIAGNOSTIC: Importing TensorFlow...", flush=True)
import tensorflow as tf
print("DIAGNOSTIC: TensorFlow imported successfully", flush=True)

print("DIAGNOSTIC: Importing PyTorch...", flush=True)
import torch
print("DIAGNOSTIC: PyTorch imported successfully", flush=True)

print("DIAGNOSTIC: Importing Pillow...", flush=True)
from PIL import Image
print("DIAGNOSTIC: Pillow imported successfully", flush=True)

print("DIAGNOSTIC: Importing Ultralytics...", flush=True)
from ultralytics import YOLO
print("DIAGNOSTIC: Ultralytics imported successfully", flush=True)

print("DIAGNOSTIC: Importing project predictor...", flush=True)
from src.predictor import (
    create_face_detector,
    detect_all_persons,
    predict_person_attributes,
)
print("DIAGNOSTIC: Project predictor imported successfully", flush=True)

print("DIAGNOSTIC: All imports completed", flush=True)


# ============================================================
# LIMIT PYTORCH CPU THREADS
# ============================================================

print("DIAGNOSTIC: Configuring PyTorch threads...", flush=True)

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

print("DIAGNOSTIC: PyTorch thread configuration completed", flush=True)


# ============================================================
# LIMIT TENSORFLOW CPU THREADS
# ============================================================

print("DIAGNOSTIC: Configuring TensorFlow threads...", flush=True)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("DIAGNOSTIC: TensorFlow thread configuration completed", flush=True)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEMO_MODEL_PATH = (
    PROJECT_DIR / "models" / "demographic_model_best.keras"
)

AGE_MODEL_PATH = (
    PROJECT_DIR / "models" / "age_model_best.keras"
)

EMOTION_MODEL_PATH = (
    PROJECT_DIR / "models" / "emotion_model_balanced_best.keras"
)

PERSON_MODEL_PATH = (
    PROJECT_DIR / "models" / "yolo11n.pt"
)

DRESS_MODEL_PATH = (
    PROJECT_DIR / "models" / "yolo11n-seg.pt"
)

FACE_MODEL_PATH = (
    PROJECT_DIR / "models" / "blaze_face_short_range.tflite"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nationality Detection AI",
    page_icon="👤",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("👤 Nationality Detection AI")

st.markdown(
    """
Upload an image containing one or more people.

The system detects each person and estimates:

- Demographic category
- Age
- Emotion
- Dress colour
"""
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
**Important limitation:**

The demographic model uses FairFace demographic categories.
These categories are NOT actual nationality predictions.

A person's nationality cannot reliably be determined from
facial appearance alone.

The dress-colour result is a computer-vision estimate based
on person segmentation and colour analysis.
"""
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    print("DIAGNOSTIC: Starting model loading", flush=True)

    # ----------------------------------------
    # DEMOGRAPHIC MODEL
    # ----------------------------------------

    print("DIAGNOSTIC: Loading demographic model...", flush=True)

    demo_model = tf.keras.models.load_model(
        DEMO_MODEL_PATH,
        compile=False,
    )

    print("DIAGNOSTIC: Demographic model loaded", flush=True)

    # ----------------------------------------
    # AGE MODEL
    # ----------------------------------------

    print("DIAGNOSTIC: Loading age model...", flush=True)

    age_model = tf.keras.models.load_model(
        AGE_MODEL_PATH,
        compile=False,
    )

    print("DIAGNOSTIC: Age model loaded", flush=True)

    # ----------------------------------------
    # EMOTION MODEL
    # ----------------------------------------

    print("DIAGNOSTIC: Loading emotion model...", flush=True)

    emotion_model = tf.keras.models.load_model(
        EMOTION_MODEL_PATH,
        compile=False,
    )

    print("DIAGNOSTIC: Emotion model loaded", flush=True)

    # ----------------------------------------
    # YOLO PERSON MODEL
    # ----------------------------------------

    print("DIAGNOSTIC: Loading YOLO person model...", flush=True)

    person_model = YOLO(
        str(PERSON_MODEL_PATH)
    )

    print("DIAGNOSTIC: YOLO person model loaded", flush=True)

    # ----------------------------------------
    # YOLO DRESS SEGMENTATION MODEL
    # ----------------------------------------

    print("DIAGNOSTIC: Loading YOLO segmentation model...", flush=True)

    dress_model = YOLO(
        str(DRESS_MODEL_PATH)
    )

    print("DIAGNOSTIC: YOLO segmentation model loaded", flush=True)

    # ----------------------------------------
    # MEDIAPIPE FACE DETECTOR
    # ----------------------------------------

    print("DIAGNOSTIC: Creating MediaPipe face detector...", flush=True)

    face_detector = create_face_detector(
        FACE_MODEL_PATH
    )

    print("DIAGNOSTIC: MediaPipe face detector created", flush=True)

    print("DIAGNOSTIC: All models loaded successfully", flush=True)

    return (
        demo_model,
        age_model,
        emotion_model,
        person_model,
        dress_model,
        face_detector,
    )


# ============================================================
# LOAD MODELS WITH ERROR HANDLING
# ============================================================

try:

    with st.spinner("Loading AI models..."):

        (
            demo_model,
            age_model,
            emotion_model,
            person_model,
            dress_model,
            face_detector,
        ) = load_models()

    st.success("AI models loaded successfully.")

except Exception as error:

    st.error(
        "Unable to load the AI models."
    )

    st.exception(error)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ------------------------------------
        # READ IMAGE
        # ------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        st.subheader("Input Image")

        st.image(
            image,
            width="stretch",
        )

        st.divider()

        # ------------------------------------
        # DETECT PERSONS
        # ------------------------------------

        with st.spinner(
            "Detecting people and faces..."
        ):

            persons = detect_all_persons(
                image_array,
                person_model,
                conf=0.25,
                face_detector=face_detector,
            )

        # ------------------------------------
        # NO PERSON DETECTED
        # ------------------------------------

        if not persons:

            st.error(
                "No person was detected in the uploaded image."
            )

        else:

            st.success(
                f"{len(persons)} person(s) detected."
            )

            # --------------------------------
            # RUN PREDICTIONS
            # --------------------------------

            predictions = []

            with st.spinner(
                "Running predictions..."
            ):

                for person_data in persons:

                    prediction = predict_person_attributes(
                        person_data,
                        demo_model,
                        age_model,
                        emotion_model,
                        dress_model,
                    )

                    predictions.append(
                        prediction
                    )

            # --------------------------------
            # DISPLAY RESULTS
            # --------------------------------

            st.subheader("Prediction Results")

            for index, (
                person_data,
                prediction,
            ) in enumerate(
                zip(persons, predictions),
                start=1,
            ):

                st.markdown(
                    f"### 👤 Person {index}"
                )

                col1, col2 = st.columns(2)

                # ----------------------------
                # DEMOGRAPHIC AND AGE
                # ----------------------------

                with col1:

                    st.markdown(
                        "#### Demographic"
                    )

                    st.metric(
                        "Category",
                        prediction["demographic"],
                    )

                    st.write(
                        "Confidence:",
                        f'{prediction["demographic_confidence"]:.2f}%',
                    )

                    st.markdown(
                        "#### Age"
                    )

                    if prediction["age"] is not None:

                        st.metric(
                            "Estimated Age",
                            f'{prediction["age"]} years',
                        )

                    else:

                        st.write(
                            "Face not detected"
                        )

                # ----------------------------
                # EMOTION AND DRESS COLOUR
                # ----------------------------

                with col2:

                    st.markdown(
                        "#### Emotion"
                    )

                    st.metric(
                        "Emotion",
                        prediction["emotion"],
                    )

                    st.write(
                        "Confidence:",
                        f'{prediction["emotion_confidence"]:.2f}%',
                    )

                    st.markdown(
                        "#### Dress Colour"
                    )

                    st.metric(
                        "Colour",
                        prediction["dress_color"],
                    )

                    st.write(
                        "Confidence:",
                        f'{prediction["dress_confidence"]:.2f}%',
                    )

                # ----------------------------
                # FACE DETECTION STATUS
                # ----------------------------

                if prediction["face_detected"]:

                    st.success(
                        "Face detected successfully."
                    )

                else:

                    st.warning(
                        "Face was not detected. "
                        "Age and emotion could not be estimated."
                    )

                st.divider()

    except Exception as error:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Nationality Detection AI — "
    "Assignment / Educational Project"
)
