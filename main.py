

from model import DaignosisCNN
from dataset import get_dataloaders
from train import train
from eval import evaluate
from eval import plot_confusion_matrix
from utils import plot_loss_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config
from dataset import train_loader, test_loader
from model import optimizer, loss_fn
from torchvision import models

def run():
  model = DaignosisCNN()
  train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)
  train(model, train_loader, config.device, epochs=config.epochs)
  evaluate(model, test_loader, config.device)
  losses = train(model, train_loader, device, epochs)
  plot_loss_curve(losses)
  plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
  run()

#Saving the model
torch.save(model.state_dict(), 'DaignosisCNN.pth')
print("✅ Model saved as DaignosisCNN.pth")

#Applying Dynamic Quantization
from torch.quantization import quantize_dynamic
from models.resnet import build_model

model_fp32 = build_model(num_classes=2)
model_fp32.load_state_dict(torch.load('DaignosisCNN.pth'))

model_quantized = quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

torch.save(model_quantized.state_dict(), 'Quantized_DaignosisCNN.pth')

#applying QAT
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model_fp32, inplace=True)

#Pseudo-Training QAT Model
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    model_fp32.train()
    for images, labels in train_loader:

        optimizer.zero_grad()
        outputs = model_fp32(images)
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

        ouputs = model_fp32(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (labels == preds).sum().item()

accuracy = 100 * correct / total
average_loss = total_loss / len(train_loader)
print(f"QAT Test Accuracy {accuracy:.2f}%")

torch.quantization.convert(model_fp32.eval(), inplace=False)

#Saving QAT Model
torch.save(model.state_dict(), 'QAT_Daignosis_CNN.pth')

