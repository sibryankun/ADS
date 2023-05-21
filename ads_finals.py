import streamlit as st
import tensorflow as tf
from tensorflow import keras

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('base_dir.h5')
  return model
model=load_model()
st.write("""
# Bag Classifier"""
)

import cv2
from PIL import Image,ImageOps
import numpy as np

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Normalize the pixel values
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    # Reshape the image to match the model's input shape
    image = image.reshape(1, 28, 28)
    return image

uploaded_file = st.file_uploader("Drag and Drop an Image File", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.text("Please upload a Bag Image to read")
else:
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = tf.argmax(predictions, axis=1)[0].numpy()

    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", predictions[0][predicted_class])
    st.image(image, caption='Uploaded Image')