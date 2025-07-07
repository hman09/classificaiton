# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load Variables
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

# Identify the Categories

target_names = cancer.target_names
# print("Target classes:", target_names)  # ['malignant' 'benign']
# print("Target distribution:\n", pd.Series(y).value_counts())

# Data Cleaning

## Check for missing values
# print(df.isnull().sum())
# ## Check for duplicates
# print(df.duplicated().sum())


# Feature Selection

## heat map
# plt.figure(figsize=(10, 10))
# sb.heatmap(df.corr(), cmap='coolwarm')
# plt.title("Feature Correlation")
# plt.show()

# Training

from sklearn.model_selection import train_test_split

## pre normalisation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Testing

y_pred = model.predict(X_test)



# Evaluation

## ----- Unmanipulated input data results -----
print("\n--- Model 1: Raw (Unmanipulated)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

## ----- with Standardisation -----

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Retrain model with pre-processing
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model 2: Standardised (All Features)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

## ----- with feature selection -----

correlations = df.corr()['target'].drop('target').sort_values(ascending=False)

top_features = correlations.abs().sort_values(ascending=False).head(10).index.tolist()

X_selected = df[top_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Retrain model with selected features
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model 3: Feature Selection (No Scaling)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

## ----- Combining Standardisation and Feature Selection -----

# Redo feature selection from original DataFrame
correlations = df.corr()['target'].drop('target').sort_values(ascending=False)
top_features = correlations.abs().sort_values(ascending=False).head(10).index.tolist()

# Get selected features
X_selected = df[top_features]

# Standardise the selected features
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)

# Split the standardised, selected data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n--- Model 4: Feature Selection + Standardised")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
