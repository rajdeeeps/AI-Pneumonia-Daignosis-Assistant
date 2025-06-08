from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

#defining transform
def get_dataloaders(batch_size):
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

  #Defining dataset
  train_data = ImageFolder(root='/content/drive/MyDrive/AI_Datasets/X_Ray/XRAY!/chest_xray/train', transform=transform)
  test_data = ImageFolder(root='/content/drive/MyDrive/AI_Datasets/X_Ray/XRAY!/chest_xray/test', transform=transform)


  #Defining Dataloaders
  train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
  test_loader = DataLoader(test_data, shuffle=False, batch_size=32)

  return train_loader, test_loader