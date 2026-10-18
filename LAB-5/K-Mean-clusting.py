# Import all relevant libraries
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('income_evaluation.csv')

# --- FIX: Dataset ko chota kar lein taake code hang na ho ---
df = df.sample(
    n=5000, random_state=101
)  # Sirf 5000 rows par train karein ge taake fast chale

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

# Apply one-hot encoding to categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Separate features and target variable
X = df.drop('income_ >50K', axis=1)
y = df['income_ >50K']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train SVC model and evaluate using cross_val_score
svc = SVC(random_state=101)
svc.fit(X_train, y_train)
accuracies = cross_val_score(svc, X_train, y_train, cv=5)
print('Train Score:', np.mean(accuracies))
print('Test Score:', svc.score(X_test, y_test))

# --- OPTIMIZED GRID SEARCH (Kam combinations taake fast chale) ---
grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1]}
svm = SVC()
svm_cv = GridSearchCV(svm, grid, cv=3)  # cv=3 kar diya hai speed ke liye
svm_cv.fit(X_train, y_train)

print('Best Parameters:', svm_cv.best_params_)
print('Train Score:', svm_cv.best_score_)
print('Test Score:', svm_cv.score(X_test, y_test))