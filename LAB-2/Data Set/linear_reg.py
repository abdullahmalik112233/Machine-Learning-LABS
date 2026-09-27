import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Load dataset
df = pd.read_csv("Admission_Predict.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove Serial Number
df.drop("Serial No.", axis=1, inplace=True)

# Target variable
y = df["Chance of Admit"]

# Features
X = df.drop("Chance of Admit", axis=1)

print(X.head())


# =========================================================
# SIMPLE LINEAR REGRESSION
# Using only GRE Score
# =========================================================

simple_lr = LinearRegression()

simple_lr.fit(X[["GRE Score"]], y)

# Predictions using only GRE Score
simple_pred = simple_lr.predict(X[["GRE Score"]])

# Plot actual data
plt.scatter(
    X["GRE Score"],
    y,
    color="green",
    label="Actual Data",
    alpha=0.5
)

# Regression line
plt.plot(
    X["GRE Score"],
    simple_pred,
    color="red",
    linewidth=3,
    label="Regression Line"
)

plt.xlabel("GRE Score")
plt.ylabel("Admission Chance")
plt.title("GRE Score vs Admission Chance")
plt.legend()
plt.show()


# =========================================================
# TRAIN-TEST SPLIT FOR MULTIPLE LINEAR REGRESSION
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# MULTIVARIATE LINEAR REGRESSION
# =========================================================

lr = LinearRegression()

lr.fit(x_train, y_train)

# Predictions
pred = lr.predict(x_test)

# RMSE
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))

print("RMSE:", rmse)


# =========================================================
# COEFFICIENTS
# =========================================================

coefficients = lr.coef_
features = X.columns

plt.figure(figsize=(8, 5))

plt.barh(
    features,
    coefficients,
    color="teal"
)

plt.xlabel("Coefficient Value (Weight)")
plt.title("Feature Importance in Multivariate Linear Regression")

plt.axvline(
    x=0,
    color="black",
    linewidth=0.8
)

plt.show()