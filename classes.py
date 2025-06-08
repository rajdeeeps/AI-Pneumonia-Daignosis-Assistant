from eval import evaluate
from eval import plot_confusion_matrix

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

y_true, y_pred = evaluate(model, test_loader, device, class_names)
plot_confusion_matrix(y_true, y_pred, class_names)