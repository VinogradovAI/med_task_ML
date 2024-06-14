import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name):
    """ Plot the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion matrix for ' + model_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()