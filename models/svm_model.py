from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class SVMModel:
    """ Support Vector Machine Model"""
    def __init__(self, random_state=42):
        """ Initialize the model"""
        self.model = SVC(random_state=random_state, probability=True)

    def train(self, X_train, y_train):
        """ Train the model with grid search cross validation"""
        param_grid = {
            'C': [0.1, 0.5, 1, 5, 10, 20, 100],
            'gamma': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
            'kernel': ['rbf', 'linear', 'poly'],
        }
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

    def predict(self, X_test):
        """ Predict the target variable"""
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """ Evaluate the model"""
        print(classification_report(y_test, y_pred))

    def predict_proba(self, X_test):
        """ Predict the probabilities of the positive class"""
        return self.model.predict_proba(X_test)
