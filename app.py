import streamlit as st
from ultralytics import YOLO
from PIL import Image
import re

# إعداد الصفحة (يجب أن يكون أول أمر Streamlit)
st.set_page_config(
    page_title="Smart Shopping Cart",
    page_icon="🛒",
    layout="centered"
)

# تصميم الواجهة
st.markdown("""
<style>

.main-title {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    color: #2E7D32;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #666;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f5f5;
}

</style>
""", unsafe_allow_html=True)


# تحميل النموذج
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# تنظيف أسماء الكلاسات
def clean_class_name(name):
    name = re.sub(r'^\d+_', '', str(name))
    name = name.replace("_", " ")
    return name.title()


# العنوان
st.markdown(
    '<div class="main-title">🛒 Smart Shopping Cart</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Product Detection System</div>',
    unsafe_allow_html=True
)


st.write("")

st.info("""
📌 للحصول على أفضل نتيجة:
- ضع المنتج بشكل واضح داخل الصورة.
- تجنب الصور المظلمة أو المشوشة.
- حاول تصوير المنتجات من الأمام.
""")


# رفع الصورة
uploaded_file = st.file_uploader(
    "📸 Upload a product image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    with st.spinner("🤖 Detecting products..."):

        results = model(image)


    result = results[0]


    # صورة الكشف
    plotted_image = result.plot()

    st.subheader("🔍 Detection Result")

    st.image(
        plotted_image,
        use_container_width=True
    )


    st.subheader("🛒 Detected Products")


    if len(result.boxes) == 0:

        st.warning("No products detected.")

    else:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            product_name = clean_class_name(
                model.names[class_id]
            )


            st.markdown(
                f"""
                <div class="result-box">
                🏷️ <b>{product_name}</b><br>
                🎯 Confidence: {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )
