import streamlit as st
from ultralytics import YOLO
from PIL import Image
import re


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Smart Shopping Cart AI",
    page_icon="🛒",
    layout="wide"
)


# =========================
# Premium Styling
# =========================

st.markdown("""
<style>

body {
    background-color: #f7f9fc;
}


.hero {
    background: linear-gradient(135deg,#1b5e20,#43a047);
    padding: 35px;
    border-radius: 25px;
    text-align:center;
    color:white;
    margin-bottom:30px;
}


.hero h1 {
    font-size:48px;
    margin-bottom:10px;
}


.hero p {
    font-size:20px;
}


.card {

    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom:20px;

}


.metric-card {

    background:white;
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);

}


.metric-number {

    font-size:35px;
    font-weight:bold;
    color:#1b5e20;

}


.metric-title {

    color:#666;
    font-size:16px;

}



.product-card {

    background:#ffffff;
    padding:18px;
    border-radius:15px;
    margin:10px 0;
    box-shadow:0px 3px 12px rgba(0,0,0,0.08);
    border-left:5px solid #43a047;

}


.footer {

text-align:center;
color:#777;
padding:30px;

}

</style>
""", unsafe_allow_html=True)



# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():

    return YOLO("best.pt")


model = load_model()



# =========================
# Helper
# =========================

def clean_name(name):

    name = re.sub(
        r"^\d+_",
        "",
        str(name)
    )

    return name.replace("_"," ").title()



# =========================
# Header
# =========================

st.markdown("""
<div class="hero">

<h1>🛒 Smart Shopping Cart</h1>

<p>
AI-Powered Retail Product Detection System
</p>

</div>
""", unsafe_allow_html=True)



# =========================
# About
# =========================

st.markdown("""
<div class="card">

<h3>🚀 About The Project</h3>

Smart Shopping Cart is an AI-based system that uses
computer vision and YOLO deep learning technology
to automatically recognize retail products from images.

The system provides fast product detection with
confidence scores through an interactive web interface.

</div>
""", unsafe_allow_html=True)



# =========================
# Upload
# =========================

uploaded_file = st.file_uploader(
    "📸 Upload Product Image",
    type=["jpg","jpeg","png"]
)



if uploaded_file:


    image = Image.open(uploaded_file)



    with st.spinner("🤖 AI is analyzing your image..."):

        results = model(image)

        result = results[0]



    detected_image = result.plot()



    col1,col2 = st.columns(2)



    with col1:

        st.markdown(
            "### Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )



    with col2:

        st.markdown(
            "### AI Detection"
        )

        st.image(
            detected_image,
            use_container_width=True
        )



    # Statistics

    count = len(result.boxes)


    confidences=[]


    for box in result.boxes:

        confidences.append(
            float(box.conf[0])
        )


    best_confidence = max(confidences) if confidences else 0



    st.write("")


    c1,c2,c3 = st.columns(3)



    with c1:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-number">
        {count}
        </div>

        <div class="metric-title">
        Products Detected
        </div>

        </div>
        """,
        unsafe_allow_html=True)



    with c2:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-number">
        YOLO
        </div>

        <div class="metric-title">
        Detection Model
        </div>

        </div>
        """,
        unsafe_allow_html=True)



    with c3:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-number">
        {best_confidence:.0%}
        </div>

        <div class="metric-title">
        Best Confidence
        </div>

        </div>
        """,
        unsafe_allow_html=True)



    # Products

    st.markdown(
        "## 🛍️ Detected Products"
    )


    if count == 0:

        st.warning(
            "No products detected."
        )


    else:


        for box in result.boxes:


            class_id=int(box.cls[0])

            confidence=float(box.conf[0])


            product=clean_name(
                model.names[class_id]
            )


            st.markdown(f"""

            <div class="product-card">

            🏷️ <b>{product}</b>
            <br>
            🎯 Confidence: {confidence:.2%}

            </div>

            """,
            unsafe_allow_html=True)




# Footer

st.markdown("""
<div class="footer">

Built with ❤️ using YOLO | PyTorch | Streamlit

</div>
""",
unsafe_allow_html=True)
