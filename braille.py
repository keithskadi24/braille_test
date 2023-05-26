import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the saved Keras model
#@st.cache_resource
def load_model():
  model = keras.models.load_model('Final_Model.h5')
  return model
model=load_model()

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to (28, 28) and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    # Normalize the image
    image = np.array(image) / 255.0
    # Add channel dimension
    image = image[np.newaxis, ..., np.newaxis]
    return image

# Function to make predictions
 def predict(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(preprocessed_image)
    # Get the predicted label
    predicted_label = chr(np.argmax(predictions) + 97)
    return predicted_label

# Streamlit app
def main():
    # Set the title and description of the app
    st.title("Braille Character Recognition")
    st.write("Upload an image of a braille character to predict its label.")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    # Perform prediction if an image is uploaded
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform prediction on the image
        predicted_label = predict(image)
        
        # Display the predicted label
        st.write("Predicted Label:", predicted_label)

# Run the app
if __name__ == "__main__":
    main()
