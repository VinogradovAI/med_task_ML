import matplotlib.pyplot as plt
import numpy as np


def feature_importance(name, model, feature_names):
    """ Display the feature importance or coefficients depending on the model"""
    try:
        importances = model.feature_importances_
    except AttributeError:
        # For models without feature_importances_ (like Logistic Regression), use coefficients
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("The model does not have feature importances or coefficients.")
            return

    indices = np.argsort(importances)
    plt.figure(figsize=(12, 6))
    plt.title(f'Feature Importances for {name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
