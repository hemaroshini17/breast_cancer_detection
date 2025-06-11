import streamlit as st
import os
import gdown
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# File settings
MODEL_DIR = "model"
MODEL_FILE = "final_finetuned_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Public link from Google Drive (ID only)
FILE_ID = "1oX25iHg87JEe8rS06U1t2-8ko1pyoptW"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Create folder and download if missing
if not os.path.exists(MODEL_PATH):
    st.warning("🔄 Downloading model (may take a moment)...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("❌ Error loading the model. It may be incompatible or corrupted.")
    st.stop()

# Class labels
class_labels = ['Benign', 'Malignant', 'Normal']

# Preprocess function
def preprocess_image(image, img_size=128):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("🩺 Breast Cancer Detection")
st.write("Upload a breast scan to detect **Benign**, **Malignant**, or **Normal** cases.")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image")

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    st.write(f"### 🔍 Prediction: **{class_labels[class_idx]}**")
    st.progress(float(confidence))
    st.write(f"Confidence: **{confidence:.2f}**")

    st.write("### 🔬 Confidence Breakdown:")
    for i in range(3):
        st.write(f"- {class_labels[i]}: **{prediction[i]:.2f}**")
