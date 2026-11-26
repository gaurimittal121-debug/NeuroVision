
import streamlit as st
import cv2
import numpy as np
import joblib
import os
import gdown
from skimage.feature import hog

# Page settings
st.set_page_config(
    page_title="NEUROVISION",
    page_icon="🧠",
    layout="wide"
)

# Load model
if not os.path.exists("svm_model.pkl"):
    gdown.download(
        id="1bHS0i4vFPIqQPTYi-jlFxjtO2wMafPQ_",
        output="svm_model.pkl"
    )

svm = joblib.load("svm_model.pkl")

labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Website design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b192e, #172f4a);
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton button {
    background-color: #168aad;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Website heading
st.title("🧠 NeuroVision")
st.caption("Smart Brain MRI Analysis | Machine Learning Project")

st.divider()

st.subheader("Welcome to NeuroVision")
st.write(
    "Upload a brain MRI image to explore its machine learning "
    "classification result and basic image-processing analysis."
)

st.info("Supported categories: Glioma | Meningioma | No Tumor | Pituitary")

# Image upload
st.subheader("📤 Upload MRI Image")

file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = cv2.imdecode(
        np.frombuffer(file.getvalue(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        st.error("Unable to read this image. Please upload another image.")

    else:
        st.image(img, caption="Uploaded MRI Image", width=350)

        # Resize image
        img = cv2.resize(img, (224, 224))

        # HOG feature extraction
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        # SVM prediction
        result = svm.predict([features])

        st.divider()
        st.subheader("🔍 Analysis Result")

        st.success(
            "Predicted Category: " + labels[int(result[0])]
        )

        # Bright-region calculation
        roi = img[20:204, 20:204]

        _, mask = cv2.threshold(
            roi, 180, 255, cv2.THRESH_BINARY
        )

        percentage = (
            np.count_nonzero(mask) / mask.size
        ) * 100

        st.metric(
            "Percentage: Estimated Bright Area (%)",
            str(round(percentage, 2)) + "%"
        )

        # Find largest bright region
        count, labels_img, stats, centres = (
            cv2.connectedComponentsWithStats(mask)
        )

        if count > 1:

            largest = 1 + np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )

            cx, cy = centres[largest]

            if cx < roi.shape[1] / 3:
                location = "Left"
            elif cx > 2 * roi.shape[1] / 3:
                location = "Right"
            else:
                location = "Centre"

            st.write(
                "Location: Detected Bright Area Location:",
                location
            )

        else:
            st.write("No bright region detected")

        st.caption(
            "Note: Bright pixel percentage and location are "
            "basic image-processing estimates, not actual tumor "
            "segmentation or medical measurements."
        )

st.divider()

st.subheader("About the Project")
st.write(
    "NeuroVision uses Histogram of Oriented Gradients (HOG) "
    "for feature extraction and Support Vector Machine (SVM) "
    "for MRI image classification."
)


