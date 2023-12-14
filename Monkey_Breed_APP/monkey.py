import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load a pre-trained model for monkey breed classification
model = tf.keras.models.load_model('monkeyBreed.h5')

# Set Streamlit app title and description
st.set_page_config(
    page_title="Monkey Breed Classifier",
    page_icon="🐒",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Monkey Breed Classifier")

# Create a file uploader widget
uploaded_image = st.file_uploader("Upload an image (JPG, PNG, or JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Check if the uploaded file is an image and classify the breed
    try:
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))  # Resize to match the model input size
        image = np.array(image)
        image = image / 255.0  # Normalize the image

        # Make a prediction using the model
        prediction = model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(prediction)

        # Define a list of monkey breeds (replace with your own list)
        monkey_breeds = ["Mantled Howler", "Patas Monkey", "Bald Uakari", "Japanese Macaque", "Pygmy Marmoset", "White Headed Capuchin", "Silvery Marmoset", "Common Squirrel Monkey", "Black Headed Night Monkey", "Nilgiri Langur"]

        # Display the predicted monkey breed
        st.success(f"Predicted Monkey Breed: {monkey_breeds[predicted_class]}")
    except:
        st.error("❌ This is not an image.")

