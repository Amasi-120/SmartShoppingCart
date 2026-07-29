import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل النموذج
model = YOLO("best.pt")

st.set_page_config(page_title="Smart Shopping Cart")

st.title("🛒 Smart Shopping Cart")
st.write("Upload an image to detect products.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting products..."):

        results = model(image)

    result = results[0]

    plotted = result.plot()

    st.image(
        plotted,
        caption="Detection Result",
        use_container_width=True
    )

    st.subheader("Detected Products")

    if len(result.boxes) == 0:
        st.warning("No products detected.")
    else:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            st.write(
                f"**{model.names[class_id]}** - {confidence:.2%}"
            )
