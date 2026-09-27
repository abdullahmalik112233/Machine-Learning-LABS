import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# NAME: Abdullah
# ROLL NO: 23-NTU-CS-1005
# =========================================================

# =========================================================
# 1. LOAD DATASET
# =========================================================
df = pd.read_csv("Data Set/Car Price Prediction.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Information:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# =========================================================
# 2. FEATURES AND TARGET
# =========================================================

# Available numerical features in this dataset
X = df[["enginesize", "horsepower", "curbweight"]]

# Target variable
y = df["price"]


# =========================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. LINEAR REGRESSION
# =========================================================

linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict prices
y_pred_linear = linear_model.predict(X_test)

# Calculate RMSE
linear_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_linear)
)

# Calculate R2
linear_r2 = r2_score(y_test, y_pred_linear)

print("\n=========================================================")
print("LINEAR REGRESSION RESULTS")
print("=========================================================")

print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)


# =========================================================
# 5. POLYNOMIAL REGRESSION - DEGREE 2
# =========================================================

poly2_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2)),
    ("regression", LinearRegression())
])

# Train model
poly2_model.fit(X_train, y_train)

# Predict
y_pred_poly2 = poly2_model.predict(X_test)

# Calculate RMSE
poly2_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_poly2)
)

# Calculate R2
poly2_r2 = r2_score(y_test, y_pred_poly2)


# =========================================================
# 6. POLYNOMIAL REGRESSION - DEGREE 3
# =========================================================

poly3_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=3)),
    ("regression", LinearRegression())
])

# Train model
poly3_model.fit(X_train, y_train)

# Predict
y_pred_poly3 = poly3_model.predict(X_test)

# Calculate RMSE
poly3_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_poly3)
)

# Calculate R2
poly3_r2 = r2_score(y_test, y_pred_poly3)


# =========================================================
# 7. POLYNOMIAL REGRESSION - DEGREE 4
# =========================================================

poly4_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=4)),
    ("regression", LinearRegression())
])

# Train model
poly4_model.fit(X_train, y_train)

# Predict
y_pred_poly4 = poly4_model.predict(X_test)

# Calculate RMSE
poly4_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_poly4)
)

# Calculate R2
poly4_r2 = r2_score(y_test, y_pred_poly4)


# =========================================================
# 8. COMPARE ALL MODELS
# =========================================================

print("\n=========================================================")
print("MODEL COMPARISON")
print("=========================================================")

print("Linear Regression")
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)

print("\nPolynomial Regression Degree 2")
print("RMSE:", poly2_rmse)
print("R2 Score:", poly2_r2)

print("\nPolynomial Regression Degree 3")
print("RMSE:", poly3_rmse)
print("R2 Score:", poly3_r2)

print("\nPolynomial Regression Degree 4")
print("RMSE:", poly4_rmse)
print("R2 Score:", poly4_r2)


# =========================================================
# 9. PLOT FITTED CURVES
# =========================================================

# We will show the effect of Engine Size on Price.
# Other features are kept at their average values.

engine_values = np.linspace(
    X["enginesize"].min(),
    X["enginesize"].max(),
    300
)

# Create data for prediction
curve_data = pd.DataFrame({
    "enginesize": engine_values,
    "horsepower": X["horsepower"].mean(),
    "curbweight": X["curbweight"].mean()
})

# Predict using polynomial models
curve_poly2 = poly2_model.predict(curve_data)
curve_poly3 = poly3_model.predict(curve_data)
curve_poly4 = poly4_model.predict(curve_data)


# Plot actual data
plt.scatter(
    X["enginesize"],
    y,
    label="Actual Data"
)

# Plot degree 2 curve
plt.plot(
    engine_values,
    curve_poly2,
    label="Polynomial Degree 2"
)

# Plot degree 3 curve
plt.plot(
    engine_values,
    curve_poly3,
    label="Polynomial Degree 3"
)

# Plot degree 4 curve
plt.plot(
    engine_values,
    curve_poly4,
    label="Polynomial Degree 4"
)

plt.xlabel("Engine Size")
plt.ylabel("Car Price")

plt.title("Polynomial Regression Fitted Curves")

plt.legend()
plt.show()