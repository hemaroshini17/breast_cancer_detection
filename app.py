import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load final model
model = load_model('model/final_finetuned_model.keras')

# Correct class label order based on folder structure used by flow_from_directory
class_labels = ['Benign', 'Malignant', 'Normal']

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
