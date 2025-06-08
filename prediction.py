from torchvision import transforms
from PIL import Image

def prediction(img):
  transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
  ])

  img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  input_tensor = transform(img).unsqueeze(0).to(device)

  with torch.no_grad():
      output = model(input_tensor)
      pred = torch.max(output, 1)

  print(f"Predicted Class: {pred.item(0)}")