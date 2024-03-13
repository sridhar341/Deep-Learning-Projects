import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

#Function to preprocess uploaded image
def preprocess_image(image_file):
    image = Image.open(image_file)
    image = np.array(image)
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    return image

# Load saved models
models = {
    "Simple CNN": "best_model_simple_cnn.h5",
    "VGG16": "best_model_vgg16.h5",
    "MobileNetV2": "best_model_mobilenetv2.h5",
    "ResNet50": "best_model_resnet50.h5"
}

# Set page title and favicon
st.set_page_config(page_title="Malaria Detection App", page_icon="🦠")

# Set app title and subtitle
st.title("Malaria Detection App")
st.write("An AI-powered app to detect malaria in blood cell images.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    st.image(uploaded_file, caption='', use_column_width=True)
    
    # Model selection
    st.subheader("Select Model")
    model_name = st.selectbox("Choose the detection model", list(models.keys()))
    
    # Load selected model
    model_path = models[model_name]
    model = load_model(model_path)
    
    # Preprocess image
    processed_image = preprocess_image(uploaded_file)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(processed_image)
    prediction_label = "Infected" if prediction > 0.5 else "Uninfected"
    
    # Display prediction result
    st.subheader("Prediction")
    if prediction_label == "Infected":
        st.error("Malaria Detected 🦠")
    else:
        st.success("No Malaria Detected 🩺")
