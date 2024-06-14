from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

class GradientBoostingModel:
    """ Gradient Boosting Model"""
    def __init__(self, random_state=42):
        """ Initialize the model"""
        self.model = GradientBoostingClassifier(random_state=random_state)

    def train(self, X_train, y_train):
        """ Train the model with grid search cross validation"""

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 1],
            'max_depth': [3, 5, 7],
            # Add other parameters as needed
        }

        # Initialize the grid search
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   cv=3, scoring='accuracy',
                                   verbose=1,
                                   n_jobs=-1)

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Update the model with the best parameters
        self.model = grid_search.best_estimator_

    def predict(self, X_test, threshold=0.5):
        """ Predict the target variable"""
        # Predict the probabilities of the positive class
        y_scores = self.model.predict_proba(X_test)[:, 1]
        # Classify the samples based on the threshold
        return np.where(y_scores >= threshold, 1, 0)

    def evaluate(self, y_test, y_pred):
        """ Evaluate the model"""
        print(classification_report(y_test, y_pred))

    def predict_proba(self, X_test):
        """ Predict the probabilities of the positive class"""
        return self.model.predict_proba(X_test)