import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# ======================
# Konfigurasi halaman
# ======================
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("😊 Emosi Deteksi CNN (Happy vs Sad)")
st.write("Upload gambar wajah untuk diklasifikasikan sebagai *Happy*, *Sad*, atau *Unknown*.")

# ======================
# Load Model
# ======================
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    model = EmotionCNN()
    model.load_state_dict(torch.load("emotion_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ======================
# Preprocessing
# ======================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)  # tambah batch dimensi

# ======================
# Upload Image
# ======================
uploaded_file = st.file_uploader("Upload gambar wajah (jpeg/jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Proses & prediksi
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()

    # Interpretasi
    happy_threshold = 0.6
    sad_threshold = 0.4

    if prob >= happy_threshold:
        label = "😊 Happy"
    elif prob <= sad_threshold:
        label = "😢 Sad"
    else:
        label = "😐 Unknown"

    st.subheader("Hasil Prediksi:")
    st.success(f"Model memprediksi: **{label}** (Probabilitas: {prob:.2f})")
