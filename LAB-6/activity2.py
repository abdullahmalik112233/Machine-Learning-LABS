import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ---------------------------------------------------------
# Task 1: Load dataset, shape, first 10 rows, missing values & class distribution
# ---------------------------------------------------------
df_heart = pd.read_csv('heart.csv') # Kaggle Heart Disease UCI CSV

print("Dataset Shape:", df_heart.shape)
print("\nFirst 10 Rows:")
display(df_heart.head(10))

print("\nMissing Values:")
print(df_heart.isnull().sum())

# Identify target column ('target' or 'num')
target_col = 'target' if 'target' in df_heart.columns else df_heart.columns[-1]
print("\nTarget Class Distribution:")
print(df_heart[target_col].value_counts())

# ---------------------------------------------------------
# Task 2 & 3: Clean dataset (missing values + encoding) & Separate X and y
# ---------------------------------------------------------
df_heart_clean = df_heart.replace('?', np.nan).copy()

# Fill numeric missing values with median and categorical with mode
for col in df_heart_clean.columns:
    if df_heart_clean[col].dtype == 'object':
        df_heart_clean[col] = df_heart_clean[col].fillna(df_heart_clean[col].mode()[0])
    else:
        df_heart_clean[col] = df_heart_clean[col].fillna(df_heart_clean[col].median())

X_heart = df_heart_clean.drop(columns=[target_col])
y_heart = (df_heart_clean[target_col] > 0).astype(int) # Ensure binary 0/1 target

# One-hot encode any categorical columns
X_heart = pd.get_dummies(X_heart, drop_first=True)

# ---------------------------------------------------------
# Task 4: 80/20 Train-Test Split with stratify=y and random_state=42
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_heart, y_heart, test_size=0.20, stratify=y_heart, random_state=42
)

# ---------------------------------------------------------
# Task 5: Basic RandomForestClassifier with default parameters
# ---------------------------------------------------------
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
default_acc = accuracy_score(y_test, y_pred_default)
print(f"\nDefault Random Forest Test Accuracy: {default_acc:.4f}")

# ---------------------------------------------------------
# Task 6, 7 & 8: GridSearchCV (5-Fold CV, scoring='accuracy')
# ---------------------------------------------------------
param_grid_clf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid_clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_clf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_clf.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid_clf.best_params_)
print(f"Best 5-Fold Cross-Validation Accuracy: {grid_clf.best_score_:.4f}")

# ---------------------------------------------------------
# Task 9: Evaluate Best Model on Test Set
# ---------------------------------------------------------
best_rf_clf = grid_clf.best_estimator_
y_pred_best = best_rf_clf.predict(X_test)

print("\n--- Test Set Evaluation Metrics ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_best):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_best):.4f}")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Tuned Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# ---------------------------------------------------------
# Task 10: Plot Feature Importances & Identify Top 3 Features
# ---------------------------------------------------------
feat_imp = pd.Series(best_rf_clf.feature_importances_, index=X_heart.columns).sort_values(ascending=False)

print("\nTop 3 Most Influential Features:")
print(feat_imp.head(3))

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Feature Importances - Heart Disease Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()