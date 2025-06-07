%%writefile eval.py
import torch
from sklearn.metrics import classification_report

def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Model Test Accuracy: {accuracy:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

#Adding a confusion matrix plot

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8,6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='blues', xticklabels=class_names, yticklabels=class_names)
  plt.xlabel('predicted')
  plt.ylabel('actual')
  plt.title('confusion matrix')
  plt.show()