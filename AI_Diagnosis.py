#Loading Dependecies
import torch 
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder
import os 

#defining transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#Defining dataset
train_data = ImageFolder(root='D:/XRAY!/chest_xray/train', transform=transform)
test_data = ImageFolder(root='D:/XRAY!/chest_xray/test', transform=transform)

#Defining Dataloaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
test_loader = DataLoader(test_data, shuffle=False, batch_size=32)


#Designing Model Architecture
class DaignosisCNN(nn.Module):
    def __init__(self):
        super(DaignosisCNN, self).__init__()
        self.features = models.resnet18()
        self.linear = nn.Linear(self.features.fc.in_features, 2)
        
    def forward(self, x):
        return self.features(x)

#Assigning model, loss and optimization
model = DaignosisCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training Model
epochs = 5
for epoch in range(epochs):
    total_loss = 0 
    model.train()
    for images, labels in train_loader:
    
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Model Epoch {epoch+1} Loss: {total_loss:.4f}")

#Evaluating Model
model.eval()
with torch.no_grad():
    correct = 0 
    total = 0
    for images, lables in test_loader:
       
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (labels == preds).sum().item()

accuracy = 100 * correct / total
average_loss = total_loss / len(train_loader)
print(f"Model Test Accuracy {accuracy:.2f}%")

#saving model 
torch.save(model.state_dict(), 'daignosis_cnn.pth')

#Applying QAT to the model
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

#Pseudo-Training QAT Model 
epochs = 3
for epoch in range(epochs):
    total_loss = 0 
    model.train()
    for images, labels in train_loader:
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"QAT Epoch {epoch+1} Loss: {total_loss:.4f}")

model.eval()
with torch.no_grad():
    correct = 0 
    total = 0
    for images, lables in test_loader:
        
        ouputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (labels == preds).sum().item()

accuracy = 100 * correct / total
average_loss = total_loss / len(train_loader)
print(f"QAT Test Accuracy {accuracy:.2f}%")
        
torch.quantization.convert(model.eval(), inplace=False)

#Saving QAT Model
torch.save(model.state_dict(), 'QAT_Daignosis_CNN.pth')

#Designing Front-End
#defining dependencies
from PIL import Image
import streamlit as st
from model import DaignosisCNN

#Transfer learning of model
model = DaignosisCNN()
model.load_state_dict(torch.load('QAT_Daignosis_CNN.pth'))
model.eval()

#streamlit UI Design
st.title('AI Powered Medical Daignosis Assisstant')

upload = st.file_uploader('Upload Chest X-ray', type=['jpg', 'png'])

if upload:
    image = Image.open(upload).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
         
    st.write("Daignosis", "Pneumonia" if pred else "Normal")  