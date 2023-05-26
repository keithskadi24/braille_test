import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

@st.cache(allow_output_mutation=True)
def load_model_from_file():
    model = tf.keras.models.load_model('path_to_your_trained_model.h5')  # Replace 'path_to_your_trained_model.h5' with the actual path to your trained model file
    return model

model = load_model_from_file()

classes = {0: 'a', 1: 'b', 2: 'c', ...}  # Replace the class indices with the corresponding braille characters

st.write("""
# Braille Character Recognition
""")
st.write("#### Deployed by Your Name")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (28, 28)  # Adjust the size to match your trained model input shape
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_gray = img.mean(axis=2)  # Convert the image to grayscale
    img_resized = cv2.resize(img_gray, size)  # Resize the image
    img_normalized = img_resized / 255.0  # Normalize the image
    img_reshaped = img_normalized[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)
    return predicted_class

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            predicted_character = classes[prediction]
            st.success(f"Predicted Braille Character: {predicted_character}")
        else:
            st.text("The file is invalid. Upload a valid image file.")
    except Exception as e:
        st.text("An error occurred while processing the image.")
        st.text(str(e))
