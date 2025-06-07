
import torch
from model import DaignosisCNN

#loading model
model = DaignosisCNN()
model.load_state_dict(torch.load('DaignosisCNN.pth'))
model.to(device)
model.eval()

print("Model loaded and ready for inference")