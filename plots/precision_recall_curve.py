import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def plot_precision_recall_curve(y_true, y_score, model_name):
    """ Plot the precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_score = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % auc_score)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for ' + model_name)
    plt.legend(loc="lower right")
    plt.show()