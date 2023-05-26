import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the saved Keras model
@st.cache_resource
def load_model():
  model = keras.models.load_model('Final_Model.h5')
  return model
model=load_model()

def main():
    st.title("Braille Character Recognition")
    st.write("Upload an image for prediction.")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize(image, (28, 28))  # Resize to model input size
        img_array = np.array(image) / 255.0  # Normalize pixel values
        img_array = img_array[..., np.newaxis]  # Add channel dimension

        # Make prediction
        prediction = model.predict(np.array([img_array]))
        predicted_class = np.argmax(prediction)
        predicted_label = chr(prediction.argmax() + 65)

        # Display the uploaded image and prediction
        st.image(image, caption=f"Predicted Label: {predicted_label}", use_column_width=True)

if __name__ == "__main__":
    main()
