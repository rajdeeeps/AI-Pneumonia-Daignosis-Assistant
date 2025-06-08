import torch
batch_size=32
epochs=5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')