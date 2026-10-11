# ============================================================
# Lab 04 - Activity 1: KNN for Credit Card Fraud Detection
# Name: Abdullah
# Roll No: 23-NTU-CS-1005
# Dataset: creditcard.csv (Kaggle: mlg-ulb/creditcardfraud)
# Each "Cell" below = one cell in Google Colab. Upload creditcard.csv first.
# ============================================================

# ---------- Cell 1: Imports ----------
# imbalanced-learn is usually pre-installed in Colab. If not, run in a cell:  !pip install imbalanced-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE   # (for ADASYN use: from imblearn.over_sampling import ADASYN)

# ---------- Cell 2: Load the dataset and split into X and y ----------
df = pd.read_csv('creditcard.csv')
print("Dataset shape:", df.shape)
print("Class distribution:\n", df['Class'].value_counts())
print("Fraud percentage: {:.3f}%".format(df['Class'].mean() * 100))

X = df.drop('Class', axis=1)   # features
y = df['Class']                # target (0 = normal, 1 = fraud)

# ---------- Cell 3: Split into training and test sets ----------
# stratify=y keeps the same fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(cd E:\7-semsester\ML-LAB\LAB-4
python TASK-1.py
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------- Cell 4: Scale features using StandardScaler ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on training data
X_test_scaled = scaler.transform(X_test)

# ---------- Cell 5: Handle class imbalance using SMOTE ----------
# SMOTE is applied ONLY on the training data (never on the test data)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
y_train_sm = np.asarray(y_train_sm)
print("Class counts before SMOTE:", np.bincount(y_train))
print("Class counts after SMOTE: ", np.bincount(y_train_sm))

# ---------- Cell 6: Baseline KNN (k=5) WITHOUT SMOTE, for comparison ----------
knn_base = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_base.fit(X_train_scaled, y_train)
y_pred_base = knn_base.predict(X_test_scaled)
acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)
print("Baseline (no SMOTE) -> Accuracy:", acc_base, " F1:", f1_base)

# ---------- Cell 7: KNN with k=5 on SMOTE data ----------
knn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn5.fit(X_train_sm, y_train_sm)
y_pred5 = knn5.predict(X_test_scaled)
acc5 = accuracy_score(y_test, y_pred5)
f1_5 = f1_score(y_test, y_pred5)
print("SMOTE + KNN (k=5) -> Accuracy:", acc5, " F1:", f1_5)

# ---------- Cell 8: Tune k using GridSearchCV ----------
# KNN is slow on a large dataset, so GridSearchCV is run on a random sample of 20,000 rows
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_sm), 20000, replace=False)
X_tune, y_tune = X_train_sm[idx], y_train_sm[idx]

param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, cv=3, scoring='f1')
grid.fit(X_tune, y_tune)

best_k = grid.best_params_['n_neighbors']
print("Best value of k:", best_k)
print("Best CV F1-score:", grid.best_score_)

# ---------- Cell 9: Plot F1-score vs k values ----------
k_values = param_grid['n_neighbors']
cv_scores = grid.cv_results_['mean_test_score']
plt.scatter(k_values, cv_scores)
plt.xlabel("k")
plt.ylabel("F1-Score (CV)")
plt.title("F1-Score vs. k values")
plt.show()

# ---------- Cell 10: Train the final KNN with the best k ----------
knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_best.fit(X_train_sm, y_train_sm)
y_pred_best = knn_best.predict(X_test_scaled)
acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
print("SMOTE + KNN (best k=%d) -> Accuracy: %s  F1: %s" % (best_k, acc_best, f1_best))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Fraud']))

# ---------- Cell 11: Compare model performance (Accuracy and F1-Score) ----------
results = pd.DataFrame({
    'Model': ['KNN k=5 (no SMOTE)', 'SMOTE + KNN k=5', 'SMOTE + KNN best k=%d' % best_k],
    'Accuracy': [acc_base, acc5, acc_best],
    'F1-Score': [f1_base, f1_5, f1_best]
})
print(results.round(4).to_string(index=False))

results.set_index('Model')[['Accuracy', 'F1-Score']].plot(kind='bar', figsize=(8, 5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1.05)
plt.show()
