from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

class RandomForestModel:
    """ Random Forest Model"""
    def __init__(self, random_state=42):
        """ Initialize the model"""
        self.model = RandomForestClassifier(random_state=random_state)

    def train(self, X_train, y_train):
        """ Train the model with grid search cross validation"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
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

