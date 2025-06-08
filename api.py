from fastapi import FastAPI, File, UploadFile
from predition import prediction

app = FastAPI()

@app.get("/"):
def home():
  return {"Message": "Chest Xray Detection API is running"}

@app.post("/predict"):
def predict(file: UploadFile = File(...)):
  image = await file.read()
  prediction = prediction(image)
  return {"prediction": prediction}