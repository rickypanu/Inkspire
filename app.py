import streamlit as st
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
with open("emnist_model.pkl", "rb") as file:
    model = pickle.load(file)

# EMNIST ByClass labels mapping (0-9, A-Z, a-z)
emnist_byclass_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Streamlit UI
st.title("📝 InkSpire")
st.write("Upload a handwritten image to extract text.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


def preprocess_image(image):
    # Convert to grayscale
    image = np.array(image.convert("L"))
    # Invert colors: white text on black background
    image = cv2.bitwise_not(image)
    # Binarize (thresholding)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return thresh


def segment_characters(image):
    # Find contours (character boundaries)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char = image[y:y + h, x:x + w]
        char = cv2.resize(char, (28, 28))  # Resize for EMNIST
        char_images.append(char)

    # Sort contours from left to right
    char_images = sorted(char_images, key=lambda img: img.shape[1])
    return char_images


def predict_characters(char_images):
    predictions = []
    for char in char_images:
        char = char / 255.0  # Normalize
        char = np.expand_dims(char, axis=(0, -1))  # Add batch and channel dims
        prediction = model.predict(char)
        predicted_class = np.argmax(prediction)
        if 0 <= predicted_class < len(emnist_byclass_labels):
            predictions.append(emnist_byclass_labels[predicted_class])
    return "".join(predictions)


if uploaded_file is not None:
    # Convert uploaded image to PIL format
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and segment characters
    processed_image = preprocess_image(image)
    char_images = segment_characters(processed_image)

    if char_images:
        # Predict characters
        predicted_text = predict_characters(char_images)
        st.subheader(f"📝 Predicted Text: {predicted_text}")
    else:
        st.error("No characters detected.")
