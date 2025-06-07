# 🩺 AI Diagnosis Assistant – Pneumonia Detection from Chest X-Rays

This project is an AI-powered medical diagnosis assistant that identifies **pneumonia** from **chest X-ray images** using deep learning. Built with **PyTorch** and trained on a Kaggle dataset, the model leverages the **ResNet18** architecture to accurately classify whether a patient has pneumonia or not.

---

## 📂 Dataset

- **Source:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:**
  - **Normal** – Patient with healthy lungs
  - **Pneumonia** – Patient infected with pneumonia

The dataset includes separate training, validation, and test folders with labeled X-ray images.

---

## 🧠 Model Architecture

- **Model Used:** Pretrained **ResNet18**
- **Modifications:**
  - Final fully connected layer adapted for binary classification (2 classes)
  - Fine-tuned on the chest X-ray dataset



## 🔧 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-diagnosis-assistant.git
   cd ai-diagnosis-assistant

2. Install the Dependencies:
   ```bash
   pip install -r requirements.txt

3. Run Training:
   ```bash
   python train.py

4. Run Evaluation:
   ```bash
   python evaluate.py
---
## 🔧 Results
 - **Classification Report**
 (https://drive.google.com/file/d/1eHNkQ9GM1goWwPaSkdOA2Fi03SbtFdeN/view?usp=drive_link)
---
## 🚀 Inference
1. Clone the repository:
   ```bash
   python predict.py --image path_to_image.jpg

2. Output:
   ```makefile
   Prediction: Pneumonia (Confidence: 72.3%)
---
## 🔬 Future Improvement
- Add support for **multi-class classification**
- Integrate with a **Streamlit or FastAPI** web app for live demo
- Quantization-aware training for faster inference on edge devices
--- 
## 👨‍⚕️ Dicslaimer 
#### This AI assistant is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis or advice.
---
## 📌 References
- Kaggle Dataset: [Chest X-Ray Images(Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Paper: Rajpurkar et al. _"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"_
