import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class DaignosisCNN(nn.Module):
    def __init__(self):
        super(DaignosisCNN, self).__init__()
        self.features = models.resnet18()
        self.linear = nn.Linear(self.features.fc.in_features, 2)

    def forward(self, x):
        return self.features(x)