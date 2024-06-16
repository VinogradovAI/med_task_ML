import warnings
warnings.filterwarnings('ignore')

import configs.config as cfg
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

from eda.dataloader import data_loader
from eda.dataloader import data_viewer
from eda.datacleaner import zero_nan_viewer
from eda.datacleaner import data_cleaner
from eda.datacleaner import zero_filler_adv

from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel
from models.gradient_boosting_model import GradientBoostingModel
from models.logistic_regression import LogisticRegressionModel

from plots.confusion_matrix import plot_confusion_matrix
from plots.roc_curve import plot_roc_curve
from plots.precision_recall_curve import plot_precision_recall_curve

from plots.interpretation_plots import feature_importance

from scipy.stats import chi2_contingency

# Load the data
df = data_loader(cfg.FILE_PATH)

# Check the general information about the data
data_viewer(df)

# Check the missing and zero values in accordance with the rules
zero_nan_viewer(df)

# Define the target variable (binary classification)
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df = df.rename(columns={"num": cfg.TARGET_VARIABLE})

# Clean the data (based on visualizations and recommendations from the EDA part)
columns_to_drop = ["id", "origin", "slope", "ca", "thal", "fbs"]
columns_to_fill = ["chol", "oldpeak"]
lines_to_drop = ["trestbps", "restecg"]

# Drop the rows with too many missing values
df = df.dropna(subset=lines_to_drop)
# Clean the columns with too many missing values
df = data_cleaner(df, drop_columns=columns_to_drop, fill_columns=columns_to_fill)

# Fill the missing values (zeros) with the mean value of the same target class
zero_filler_adv(df, fill_columns=["chol", "trestbps"], key=True)

# Typing the data
columns_numerical = ["age", "trestbps", "chol", "thalch", "oldpeak", cfg.TARGET_VARIABLE]
df[columns_numerical] = df[columns_numerical].astype("float64")

columns_categorical = ["sex", "cp", "restecg", "exang"]
df[columns_categorical] = df[columns_categorical].astype("category")

# Check data for outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=df, orient="h")
plt.show()

# Remove outliers (based on visualizations and business logic)
df = df[df["trestbps"] < 180]
df = df[df["chol"] < 400]
df = df[df["thalch"] > 70]
df = df[df["thalch"] < 200]

# Fill the missing values (zeros) with the mean value of the same target class
zero_filler_adv(df, fill_columns=["chol", "trestbps"], key=True, name="target")

# Check dependencies between features with focus on target
sns.pairplot(df, hue="target", diag_kind="hist", palette="husl")
plt.show()

# Check the correlation between features
plt.figure(figsize=(15, 10))
sns.heatmap(df[columns_numerical].corr(), annot=True, cmap="coolwarm")
plt.show()

# Check the correlation between categorical features
for column1 in columns_categorical:
    for column2 in columns_categorical:
        if column1 != column2:
            print(f"{column1} vs {column2}")
            contingency_table = pd.crosstab(df[column1], df[column2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            print(f"Chi-square statistic: {chi2}")
            print(f"p-value: {p}")

# Check the correlation between categorical target
for column in columns_categorical:
    print(f"TARGET vs {column}")
    contingency_table = pd.crosstab(df["target"], df[column])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square statistic: {chi2}")
    print(f"p-value: {p}")
    print("\n")

# Remove the features with low correlation with target (based on result)
columns_to_remove = "exang"
df = df.drop(columns_to_remove, axis=1)
columns_categorical.remove(columns_to_remove)

# Encode categorical features
le = LabelEncoder()
df[columns_categorical] = df[columns_categorical].apply(le.fit_transform)

# Train and test split of data
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1),
                                                    df["target"],
                                                    test_size=cfg.TEST_SIZE,
                                                    random_state=cfg.RANDOM_STATE)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the models
models = {
    "Logistic Regression": LogisticRegressionModel(random_state=cfg.RANDOM_STATE),
    "SVM": SVMModel(random_state=cfg.RANDOM_STATE),
    "Random Forest": RandomForestModel(random_state=cfg.RANDOM_STATE),
    "Gradient Boosting": GradientBoostingModel(random_state=cfg.RANDOM_STATE)
}
results = {}

# Train, predict and evaluate each model
for name, model in models.items():
    # Train the model
    print(f"Training {name}...")
    model.train(X_train, y_train)
    # Display the feature importances
    print(f"Displaying feature importances for {name}...")
    feature_importance(name, model.model, df.drop("target", axis=1).columns)
    # Predict the target variable
    y_pred = model.predict(X_test)
    # Evaluate the model
    print(f"Evaluating {name}...")
    model.evaluate(y_test, y_pred)
    # Store the results
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report['weighted avg']  # store the weighted average results
    results[name]["accuracy"] = accuracy_score(y_test, y_pred)  # store the accuracy
    # Plot the evaluation metrics
    plot_confusion_matrix(y_test, y_pred, name)
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], name)
    plot_precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1], name)

# Print the results
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {round(value, 4)}")
