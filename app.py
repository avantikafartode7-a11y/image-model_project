import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# Page config
st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("YOLO Object Detection App")
st.write("Upload an image and detect objects using YOLO")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")   # make sure this file is in the same folder

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=Tru
