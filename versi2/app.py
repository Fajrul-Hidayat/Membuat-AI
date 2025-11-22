%%writefile app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2

from ocr_utils import load_and_preprocess_image
from decoder import greedy_decode

MODEL_PATH = "inference_model.keras"
VECT_PATH = "vectorizer.pkl"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_vectorizer():
    with open(VECT_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
vect = load_vectorizer()

st.title("📝 Handwriting OCR — CRNN + CTC")
st.write("Upload gambar paragraf handwriting kamu.")

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    raw_bytes = uploaded.read()
    arr = np.frombuffer(raw_bytes, np.uint8)
    raw = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    st.image(raw, caption="Uploaded image", use_column_width=True)

    img = load_and_preprocess_image(raw_bytes)
    img = np.expand_dims(img, axis=0)

    st.write("🔄 Running OCR...")
    pred = model.predict(img)

    result = greedy_decode(pred, vect.idx_to_char)

    st.subheader("📜 OCR Result")
    st.code(result)
