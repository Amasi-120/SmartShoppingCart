import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("Smart Shopping Cart")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image)

    results = model(image)

    st.write(results)
