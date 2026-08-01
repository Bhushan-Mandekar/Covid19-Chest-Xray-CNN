import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/final_covid_model.keras")

model = load_model()

# ---------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------
class_names = [
    "COVID-19",
    "Normal",
    "Viral Pneumonia"
]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🩺 COVID-19 Detection")

st.sidebar.info(
    """
This application predicts chest X-ray images using a trained Deep Learning model.

### Classes
- 🦠 COVID-19
- ✅ Normal
- 🫁 Viral Pneumonia

### Model
Transfer Learning + CNN

### Image Size
224 × 224
"""
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🩺 COVID-19 Detection from Chest X-rays using CNN")

st.write(
    "Upload a Chest X-ray image and the trained model will predict "
    "whether the patient has COVID-19, Viral Pneumonia, or is Normal."
)

st.markdown("---")

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

    with col2:

        img = image.resize((224, 224))
        img = np.array(img)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        start = time.time()

        with st.spinner("Analyzing Chest X-ray..."):

            prediction = model.predict(img, verbose=0)

        end = time.time()

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.subheader("Prediction Result")

        if predicted_class == 0:

            st.error("🚨 COVID-19 Detected")

        elif predicted_class == 1:

            st.success("✅ Normal Chest X-ray")
            st.balloons()

        else:

            st.warning("⚠️ Viral Pneumonia Detected")

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )

        st.progress(float(confidence))

        st.metric(
            label="Prediction Time",
            value=f"{end-start:.3f} sec"
        )

    st.markdown("---")

    st.subheader("Prediction Probabilities")

    covid_prob = prediction[0][0]
    normal_prob = prediction[0][1]
    viral_prob = prediction[0][2]

    st.write("🦠 COVID-19")

    st.progress(float(covid_prob))

    st.write(f"{covid_prob*100:.2f}%")

    st.write("✅ Normal")

    st.progress(float(normal_prob))

    st.write(f"{normal_prob*100:.2f}%")

    st.write("🫁 Viral Pneumonia")

    st.progress(float(viral_prob))

    st.write(f"{viral_prob*100:.2f}%")

    st.markdown("---")

    with st.expander("View Raw Prediction Values"):

        st.write(prediction)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
"""
### Project Information

**Mini Project:** COVID-19 Detection from Chest X-rays using CNN

**Developed Using**
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy

**Classes Predicted**
- COVID-19
- Normal
- Viral Pneumonia
"""
)