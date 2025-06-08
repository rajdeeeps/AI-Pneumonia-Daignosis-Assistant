import streamlit as st
from PIL import Image
import torch

st.set_page_config(page_title="Pneumonia Daignosis", layout='centered')
st.title('AI Pneumonia Detector')

uploaded_file = st.file_uploader("Upload a chest XRay Image", type=['jpg','jpeg','png'])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded Chest Xray', use_column_width=True)

  with st.spinner("Analyzing")
  prediction = prediction(image)

  st.succes(f'Prediction: {prediction}')