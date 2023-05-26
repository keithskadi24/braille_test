import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the saved Keras model
model = keras.models.load_model('Final_Model.h5')

# Function to make predictions
def make_prediction(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image to the input shape expected by the model
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class
    return predicted_class

# Streamlit app code
def main():
    st.title("Image Classifier")
    st.write("Upload an image and get the predicted class.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction on the uploaded image
        if st.button("Classify"):
            prediction = make_prediction(image)
            st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()