import matplotlib.pyplot as plt

def plot_loss_curve(losses):
  plt.figure(figsize=(8,6))
  plt.plot(losses, marker='o')
  plt.title('training loss over epochs')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.grid(True)
  plt.show()