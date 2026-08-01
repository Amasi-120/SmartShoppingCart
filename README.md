# 🛒 Smart Shopping Cart - AI-Based Product Detection System

## 📌 Overview

Smart Shopping Cart is an AI-powered system designed to automatically detect retail products from images using computer vision techniques. The system uses a trained YOLO deep learning model to identify products and provide detection results through an interactive Streamlit web application.

The project aims to improve the shopping experience by reducing manual product scanning and enabling faster product recognition.

---

## ✨ Features

* 📸 Upload product images through a user-friendly web interface.
* 🤖 Detect products using a YOLO object detection model.
* 🔍 Identify multiple products in a single image.
* 📊 Display detected product categories with confidence scores.
* 🌐 Deployable as a Streamlit web application.

---

## 🧠 AI Model

The project uses:

* **YOLO (You Only Look Once)** for real-time object detection.
* A custom-trained model (`best.pt`) trained on retail product images.

The model can detect different product categories such as:

* Drink
* Chocolate
* Milk
* Candy
* Dried Food
* Personal Hygiene
* Tissue
* Instant Noodles
* And other retail categories

---


## 📷 How to Use

1. Open the Smart Shopping Cart application.
2. Upload an image containing retail products.
3. The AI model processes the image.
4. The detected products and confidence scores are displayed.

---

## 🛠️ Technologies Used

* Python
* YOLO
* Ultralytics
* OpenCV
* PyTorch
* Pillow
* Streamlit
* Google Colab
* Kaggle Dataset

---

## 📊 Dataset

The model was trained using a retail product detection dataset containing annotated product images in YOLO format.

The dataset includes multiple product categories to support automatic product recognition.

---


## 🔮 Future Improvements

* Add automatic price calculation.
* Implement shopping cart management.
* Improve product classification accuracy.
* Add product recommendations.
* Support real-time camera scanning.
* Integrate barcode recognition.
