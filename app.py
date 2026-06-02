import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️",
    layout="centered"
)

# Custom CSS for Colors
st.markdown("""
    <style>
    .stApp {
        background-color: orange;
    }
    h1 {
        color: #4B2E2E; /* Dark Brown */
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        padding-bottom: 20px;
    }
    .description-text {
        color: #4B2E2E;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    [data-testid="stFileUploader"] {
        background-color: brown;
        padding: 15px;
        border-radius: 10px;
    }
    /* Uploader label, drag-drop text, and small helper text to black */
    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploader"] section div, 
    [data-testid="stFileUploader"] small {
        color: black !important;
    }
    div.stButton > button {
        background-color: #D2B48C !important; /* Light Brown */
        color: black !important;
        border: none;
        font-weight: bold;
    }
    /* Result Box Styling */
    .result-container {
        padding: 25px;
        border-radius: 12px;
        color: white !important;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.3);
    }
    .real-bg { background-color: #004d40; } /* Deep Dark Green */
    .fake-bg { background-color: #8b0000; } /* Deep Dark Red */
    </style>
    """, unsafe_allow_html=True)

st.title("🕵️ Deepfake Image Detection")
st.markdown('<p class="description-text">Upload an image and detect whether it is Real or Fake.</p>', unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(),
        nn.Dropout(0.5),

        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    model.load_state_dict(
        torch.load(
            "best_resnet52.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model


model = load_model()

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# Upload Image
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Submit Analysis"):
        img = transform(image)
        img = img.unsqueeze(0)

        with torch.no_grad():
            prediction = model(img)
            confidence = prediction.item()

        if confidence >= 0.6:
            st.markdown(f'<div class="result-container real-bg">✅ REAL IMAGE<br><br>Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-container fake-bg">❌ FAKE IMAGE<br><br>Confidence: {(1-confidence):.2%}</div>', unsafe_allow_html=True)

        st.progress(float(max(confidence, 1-confidence)))