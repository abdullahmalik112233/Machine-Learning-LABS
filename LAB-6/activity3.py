import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Task 1 & 2: Load train.csv, display shape, first 10 rows, missing values & SalePrice stats
# ---------------------------------------------------------
df_house = pd.read_csv('train.csv') # Kaggle House Prices train.csv

print("Dataset Shape:", df_house.shape)
print("\nFirst 10 Rows:")
display(df_house.head(10))

print("\nColumns with Missing Values:")
missing_counts = df_house.isnull().sum()
print(missing_counts[missing_counts > 0].sort_values(ascending=False))

print("\nSummary Statistics of SalePrice:")
print(df_house['SalePrice'].describe())

# ---------------------------------------------------------
# Task 3 & 4: Separate X and y, drop 'Id', handle missing values & encode categoricals
# ---------------------------------------------------------
X_house = df_house.drop(columns=['Id', 'SalePrice']).copy()
y_house = df_house['SalePrice'].copy()

# Separate numeric and categorical columns
num_cols = X_house.select_dtypes(include=[np.number]).columns
cat_cols = X_house.select_dtypes(exclude=[np.number]).columns

# Fill numeric NAs with median, categorical NAs with 'Missing'
X_house[num_cols] = X_house[num_cols].fillna(X_house[num_cols].median())
X_house[cat_cols] = X_house[cat_cols].fillna('Missing')

# One-Hot Encode categorical variables
X_house_encoded = pd.get_dummies(X_house, columns=cat_cols, drop_first=True)

# ---------------------------------------------------------
# Task 5: Split dataset (80/20 ratio, random_state=42)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_house_encoded, y_house, test_size=0.20, random_state=42
)

# ---------------------------------------------------------
# Task 6: Train Basic RandomForestRegressor (Default Parameters)
# ---------------------------------------------------------
rf_reg_default = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_reg_default.fit(X_train, y_train)
y_pred_def = rf_reg_default.predict(X_test)

def_mae = mean_absolute_error(y_test, y_pred_def)
def_mse = mean_squared_error(y_test, y_pred_def)
def_rmse = np.sqrt(def_mse)
def_r2 = r2_score(y_test, y_pred_def)

# ---------------------------------------------------------
# Task 7, 8 & 9: GridSearchCV with 5-Fold CV & 'neg_mean_squared_error'
# ---------------------------------------------------------
param_grid_reg = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [0.7, 1.0]
}

grid_reg = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid_reg,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_reg.fit(X_train, y_train)

best_neg_mse = grid_reg.best_score_
best_cv_mse = -best_neg_mse
best_cv_rmse = np.sqrt(best_cv_mse)

print("\nBest Hyperparameters:", grid_reg.best_params_)
print(f"Best CV Negative MSE : {best_neg_mse:,.2f}")
print(f"Best CV Positive MSE : {best_cv_mse:,.2f}")
print(f"Best CV RMSE         : {best_cv_rmse:,.2f}")

# ---------------------------------------------------------
# Task 10 & 11: Evaluate Best Model on Test Set & Compare in a Table
# ---------------------------------------------------------
best_rf_reg = grid_reg.best_estimator_
y_pred_opt = best_rf_reg.predict(X_test)

opt_mae = mean_absolute_error(y_test, y_pred_opt)
opt_mse = mean_squared_error(y_test, y_pred_opt)
opt_rmse = np.sqrt(opt_mse)
opt_r2 = r2_score(y_test, y_pred_opt)

comparison_reg = pd.DataFrame({
    'Model': ['Default RandomForestRegressor', 'Optimized RandomForestRegressor (GridSearchCV)'],
    'MAE': [round(def_mae, 2), round(opt_mae, 2)],
    'MSE': [round(def_mse, 2), round(opt_mse, 2)],
    'RMSE': [round(def_rmse, 2), round(opt_rmse, 2)],
    'R² Score': [round(def_r2, 4), round(opt_r2, 4)]
})
print("\n--- Model Performance Comparison Table ---")
display(comparison_reg)

# ---------------------------------------------------------
# Task 12: Plot Actual vs Predicted Prices & Top 5 Feature Importances
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_opt, alpha=0.6, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.title('Actual vs. Predicted House Sale Prices')
plt.xlabel('Actual SalePrice ($)')
plt.ylabel('Predicted SalePrice ($)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Top 5 Most Influential Features
house_feat_imp = pd.Series(
    best_rf_reg.feature_importances_, index=X_house_encoded.columns
).sort_values(ascending=False).head(5)

print("\nTop 5 Most Influential Features for House Prices:")
print(house_feat_imp)

plt.figure(figsize=(8, 5))
sns.barplot(x=house_feat_imp.values, y=house_feat_imp.index, palette='magma')
plt.title('Top 5 Most Influential Features (House Prices)')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()