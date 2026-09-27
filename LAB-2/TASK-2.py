import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# =========================================================
# NAME: Abdullah
# ROLL NO: 23-NTU-CS-1005
# =========================================================


# =========================================================
# 1. LOAD DATASET
# =========================================================

# Load the Telco Customer Churn dataset
df = pd.read_csv("Data Set/Telco Customer Churn.csv")


# Remove extra spaces from column names
df.columns = df.columns.str.strip()


# Display first 5 rows
print("First 5 rows:")
print(df.head())


# Display column names
print("\nColumn Names:")
print(df.columns)


# Display dataset information
print("\nDataset Information:")
print(df.info())


# Display statistical information
print("\nStatistical Information:")
print(df.describe())


# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

# TotalCharges contains some empty values.
# Convert TotalCharges into numeric values.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove rows containing missing values
df.dropna(inplace=True)


# Convert Churn into numerical values
# Yes = 1
# No = 0

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

# Features
# These columns will be used to predict customer churn

X = df[
    [
        "tenure",
        "MonthlyCharges",
        "Contract",
        "InternetService"
    ]
]


# Target
# Churn tells whether customer leaves the company

y = df["Churn"]


print("\nFeatures:")
print(X.head())


print("\nTarget:")
print(y.head())


# =========================================================
# 4. TRAIN TEST SPLIT
# =========================================================

# Split data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 5. PREPROCESSING
# =========================================================

# Numerical columns

numerical_features = [
    "tenure",
    "MonthlyCharges"
]


# Categorical columns

categorical_features = [
    "Contract",
    "InternetService"
]


# StandardScaler scales numerical values
# OneHotEncoder converts categorical values into numbers

preprocessor = ColumnTransformer(
    transformers=[

        # Scale numerical features

        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        # Encode categorical features

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# =========================================================
# 6. LOGISTIC REGRESSION MODEL
# =========================================================

# Create pipeline
# First preprocessing will happen
# Then Logistic Regression will train

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classification", LogisticRegression())
    ]
)


# Train the model

model.fit(
    x_train,
    y_train
)


# =========================================================
# 7. PREDICTION
# =========================================================

# Predict churn for test data

pred = model.predict(
    x_test
)


# Predict probability of customer churn
# [:, 1] means probability of Churn = 1

probability = model.predict_proba(
    x_test
)[:, 1]


print("\nActual Churn:")
print(y_test.head())


print("\nPredicted Churn:")
print(pred[:5])


# =========================================================
# 8. MODEL EVALUATION
# =========================================================

# Calculate Accuracy

accuracy = accuracy_score(
    y_test,
    pred
)


# Calculate Precision

precision = precision_score(
    y_test,
    pred
)


# Calculate Recall

recall = recall_score(
    y_test,
    pred
)


# Calculate F1 Score

f1 = f1_score(
    y_test,
    pred
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")


print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)


# =========================================================
# 9. CONFUSION MATRIX
# =========================================================

# Create confusion matrix

cm = confusion_matrix(
    y_test,
    pred
)


print("\nConfusion Matrix:")

print(cm)


# Display confusion matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)


disp.plot()


plt.title("Confusion Matrix")


plt.show()


# =========================================================
# 10. ROC CURVE
# =========================================================

# Calculate False Positive Rate
# and True Positive Rate

fpr, tpr, thresholds = roc_curve(
    y_test,
    probability
)


# Calculate ROC-AUC Score

auc = roc_auc_score(
    y_test,
    probability
)


# Create ROC Curve

plt.figure(figsize=(8, 6))


# Plot ROC Curve

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression (AUC = {:.2f})".format(auc)
)


# Random prediction line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")


plt.legend()


plt.show()


print("\nROC-AUC Score:", auc)