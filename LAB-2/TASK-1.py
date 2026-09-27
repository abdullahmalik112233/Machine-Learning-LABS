import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# =========================================================
# NAME: Abdullah
# ROLL NO: 23-NTU-CS-1005
# =========================================================


# =========================================================
# 1. LOAD DATASET
# =========================================================

# Load the Medical Insurance dataset
df = pd.read_csv("Data Set/Medical Cost Personal Datasets.csv")


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
# 2. FEATURES AND TARGET
# =========================================================

# Features
# These columns will be used to predict insurance cost
X = df[["age", "bmi", "children", "smoker", "region"]]


# Target
# charges is the insurance cost that we want to predict
y = df["charges"]


print("\nFeatures:")
print(X.head())


print("\nTarget:")
print(y.head())


# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================

# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. PREPROCESSING
# =========================================================

# Numerical columns
numerical_features = [
    "age",
    "bmi",
    "children"
]


# Categorical columns
categorical_features = [
    "smoker",
    "region"
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
# 5. LINEAR REGRESSION MODEL
# =========================================================

# Create pipeline
# First preprocessing will happen
# Then Linear Regression will train
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)


# Train the model
model.fit(x_train, y_train)


# =========================================================
# 6. PREDICTION
# =========================================================

# Predict insurance charges for test data
pred = model.predict(x_test)


print("\nActual Charges:")
print(y_test.head())


print("\nPredicted Charges:")
print(pred[:5])


# =========================================================
# 7. MODEL EVALUATION
# =========================================================

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, pred)


# Calculate RMSE
rmse = np.sqrt(mse)


# Calculate R2 Score
r2 = r2_score(y_test, pred)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("RMSE:", rmse)
print("R2 Score:", r2)


# =========================================================
# 8. PREDICTED VS ACTUAL COSTS
# =========================================================

plt.figure(figsize=(8, 6))


# Plot actual and predicted values
plt.scatter(
    y_test,
    pred,
    color="blue",
    alpha=0.6
)


# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)


plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")

plt.title("Predicted vs Actual Medical Insurance Costs")

plt.show()