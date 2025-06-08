import torch
from torch import nn, optim
from tqdm import tqdm

def train(model, train_loader, device, epochs=5):
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_fn = nn.CrossEntropyLoss()

  for epoch in range(epochs):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Model Epoch {epoch+1} Loss: {total_loss:.4f}")