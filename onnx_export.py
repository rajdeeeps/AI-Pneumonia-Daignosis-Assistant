%%writefile onnx_export.py
import torch
import onnx
from models.resnet import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(classes=2)
model.load_state_dict(torch.load("DaignosisCNN.pth")))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(model, dummy_input, 'DaignosisCNN.onnx',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={input:{0: 'batch_size'},'output': {0: 'output'}},
                  opset_version=11)

from PIL import Image
import numpy as np
import onnxruntime
from torchvision import transforms

transform = transforms([
    transform.Resize((224, 224)),
    transform.ToTensor()
])

img = Image.open("sample.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).numpy()

#Running inference
ort_session = onnxruntime.InferenceSession("DaignosisCNN.onnx")
output = ort_session.run(None, {"input": img_tensor})
pred = np.argmax(output[0])
print(f"prediction is {pred}")