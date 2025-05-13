import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set Streamlit app title
st.title('🍼 Plastic Bottle Anomaly Detection')

@st.cache_resource
def load_model():
    # Load the pre-trained model (cached for faster reloads)
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# Class names
class_names = ['Normal', 'Anomaly']

# File uploader
uploaded_file = st.file_uploader("📷 Upload a bottle image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # Ensure RGB
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner('🔍 Analyzing...'):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Output
    st.success(f'### 🏷️ Prediction: **{predicted_class}**')
    st.write(f'**Confidence:** {confidence:.2f}%')

else:
    st.info("⬆️ Please upload an image to get a prediction.")
