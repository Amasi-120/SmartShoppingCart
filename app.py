import streamlit as st
from ultralytics import YOLO
from PIL import Image
import re


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Shopping Cart",
    page_icon="🛒",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main-title {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    color: #1B5E20;
}

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #555;
    margin-bottom: 30px;
}


.card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
    border: 1px solid #ddd;
}


.product-card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border-left: 6px solid #2E7D32;
    margin-bottom: 10px;
    font-size: 18px;
}


.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}


</style>
""", unsafe_allow_html=True)



# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()



# -----------------------------
# Helper Function
# -----------------------------
def clean_class_name(name):

    name = re.sub(r"^\d+_", "", str(name))
    name = name.replace("_", " ")

    return name.title()



# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🛒 Smart Shopping Cart</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-Powered Retail Product Detection System</div>',
    unsafe_allow_html=True
)



# -----------------------------
# Introduction
# -----------------------------
st.markdown("""
<div class="card">

<b>About the System</b><br><br>

Smart Shopping Cart uses a YOLO deep learning model
to automatically detect retail products from images.

Upload a product image and the AI model will identify
the detected items with confidence scores.

</div>
""", unsafe_allow_html=True)



# -----------------------------
# Guidelines
# -----------------------------
with st.expander("📌 Image Guidelines"):

    st.write("""
    For better detection accuracy:

    ✅ Keep products clearly visible  
    ✅ Avoid blurry images  
    ✅ Use good lighting  
    ✅ Place products separately when possible
    """)



# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file:


    image = Image.open(uploaded_file)


    with st.spinner("Analyzing image using AI..."):

        results = model(image)

        result = results[0]



    detected_image = result.plot()



    # -----------------------------
    # Display Images
    # -----------------------------
    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )



    with col2:

        st.subheader("Detection Result")

        st.image(
            detected_image,
            use_container_width=True
        )



    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Detected Products")



    if len(result.boxes) == 0:

        st.warning(
            "No products detected."
        )


    else:

        st.success(
            f"{len(result.boxes)} product(s) detected successfully!"
        )


        for box in result.boxes:


            class_id = int(box.cls[0])

            confidence = float(
                box.conf[0]
            )


            product = clean_class_name(
                model.names[class_id]
            )


            st.markdown(
                f"""
                <div class="product-card">

                🏷️ <b>Product:</b> {product}<br>

                🎯 <b>Confidence:</b> {confidence:.2%}

                </div>
                """,
                unsafe_allow_html=True
            )



# -----------------------------
# Footer
# -----------------------------
st.markdown(
"""
<div class="footer">

Developed using YOLO + Streamlit  
Artificial Intelligence Project

</div>
""",
unsafe_allow_html=True
)
