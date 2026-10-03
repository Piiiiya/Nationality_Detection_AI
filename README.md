
# 👤 Nationality Detection AI

An AI-powered computer vision application that detects people in images and estimates demographic categories, age, facial emotion, and dress colour using Deep Learning and Machine Learning.

The application is built using Python, TensorFlow, PyTorch, YOLO, MediaPipe, and Streamlit.

> **Note:** The demographic model predicts FairFace demographic categories, not actual nationality. Nationality cannot reliably be determined from facial appearance alone.

---

## 🚀 Live Demo

**[Click here to open Nationality Detection AI](https://nationalitydetectionai-lpwcweggprs8uwhxx7qcrs.streamlit.app/)**

Try the application by uploading an image containing one or more people.

---

## 📌 Project Overview

This project combines multiple computer vision and deep learning models into a single interactive application.

Users can upload an image containing people and receive individual predictions for each detected person.

The system performs:

1. Person detection
2. Face detection
3. Demographic classification
4. Age estimation
5. Emotion recognition
6. Dress-colour estimation

The application displays the prediction results through a user-friendly Streamlit interface.

---

## ✨ Key Features

- **Person Detection:** Detects multiple people in a single image using YOLO.
- **Face Detection:** Uses MediaPipe Face Detector to locate faces.
- **Demographic Classification:** Predicts FairFace demographic categories.
- **Age Estimation:** Estimates age using a trained CNN model.
- **Emotion Recognition:** Classifies facial expressions into emotion categories.
- **Dress-Colour Estimation:** Uses person segmentation and colour analysis to estimate clothing colour.
- **Multiple-Person Processing:** Processes detected people individually.
- **Interactive GUI:** Provides an image-upload interface with prediction results.
- **CPU-Compatible Deployment:** Configured to run without a GPU.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model training and inference |
| PyTorch | YOLO model execution |
| YOLO | Person detection and segmentation |
| MediaPipe | Face detection |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pillow | Image handling |
| Scikit-learn | Machine learning utilities and evaluation |
| Streamlit | Interactive web application |
| Git & GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

---

## 🧠 AI Models

The application integrates the following trained models:

| Model | Task |
|---|---|
| `demographic_model_best.keras` | Demographic classification |
| `age_model_best.keras` | Age estimation |
| `emotion_model_balanced_best.keras` | Emotion recognition |
| `yolo11n.pt` | Person detection |
| `yolo11n-seg.pt` | Person segmentation |
| `blaze_face_short_range.tflite` | Face detection |

The trained models are stored in the `models/` directory.

---

## 📂 Project Structure

```text
Nationality_Detection_AI/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── age_model_best.keras
│   ├── demographic_model_best.keras
│   ├── emotion_model_balanced_best.keras
│   ├── yolo11n.pt
│   ├── yolo11n-seg.pt
│   └── blaze_face_short_range.tflite
│
├── src/
│   └── predictor.py
│
└── screenshots/
    └── application screenshots
```

**Note:** Dataset files and virtual environments are excluded from GitHub to keep the repository manageable. The structure above highlights the main application files.

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Piiiiya/Nationality_Detection_AI.git
```

### 2. Navigate to the project directory

```bash
cd Nationality_Detection_AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 🖥️ How to Use

1. Open the application.
2. Upload an image in JPG, JPEG, PNG, or WebP format.
3. Wait for the AI models to process the image.
4. View the number of detected people.
5. Explore the individual predictions for each person:
   - Demographic category
   - Estimated age
   - Emotion and confidence
   - Dress colour and confidence
6. Upload another image to test the application.

---

## 📊 Output

For every detected person, the application displays:

| Output | Description |
|---|---|
| Demographic | Predicted FairFace category |
| Age | Estimated age in years |
| Emotion | Predicted facial emotion |
| Dress Colour | Estimated clothing colour |
| Confidence | Model confidence for the prediction |
| Face Status | Indicates whether a face was detected |

Prediction confidence represents the model's output confidence and should not be interpreted as a guarantee of correctness.

---

## ⚠️ Limitations and Responsible AI

This project is intended for educational and experimental purposes.

- **Nationality:** The system does not determine a person's actual nationality. Demographic categories are not nationality labels.
- **Demographic Classification:** Predictions may be inaccurate and can reflect biases in the training data.
- **Age Estimation:** Predicted ages are estimates and may differ from a person's actual age.
- **Emotion Recognition:** Facial expressions do not reliably reveal a person's internal emotional state.
- **Dress Colour:** Colour predictions can be affected by lighting, shadows, image quality, and clothing patterns.
- **Confidence Scores:** Confidence values do not necessarily represent calibrated probabilities.

The predictions should not be used for identity verification, hiring decisions, law enforcement, or other high-stakes decisions.

---

## 🔮 Future Improvements

- Improve model accuracy and generalization.
- Add video-based processing.
- Improve detection under challenging lighting conditions.
- Add detailed evaluation metrics and visualizations.
- Optimize inference speed.
- Enhance the user interface.
- Add downloadable prediction reports.

---

## 👩‍💻 Developer

**Piya Shaikh**

MCA Graduate | AI & Machine Learning Enthusiast

**GitHub:** [Piiiiya](https://github.com/Piiiiya)

---

## 📜 License

This project is developed for educational and internship purposes.

Please review the licenses and usage terms of the datasets, pretrained models, and third-party libraries before redistributing them.

---

⭐ If you find this project interesting, feel free to explore the repository and try the live demo!
