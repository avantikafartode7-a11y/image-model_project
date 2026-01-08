import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("YOLO Object Detection App")
st.write("Upload an image and detect objects using YOLO")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            results = model(temp_path)
            result_img = results[0].plot()

            st.image(
                result_img,
                caption="Detected Objects",
                use_container_width=True
            )

        os.remove(temp_path)
