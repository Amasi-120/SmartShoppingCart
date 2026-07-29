from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# تحميل نموذج YOLO
model = YOLO("best.pt")


@app.get("/")
def home():
    return {"message": "Smart Shopping Cart API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(image)

    detections = []

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence > 0.1:
                class_id = int(box.cls[0])

                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "product": model.names[class_id]
                })

    return {
        "detections": detections
    }