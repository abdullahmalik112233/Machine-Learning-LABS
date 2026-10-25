import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Task 1: Load dataset, display shape, first 10 rows & class distribution
# ---------------------------------------------------------
# Update path if using local Kaggle CSV: pd.read_csv('fashion-mnist_train.csv')
df_fashion = pd.read_csv('fashion-mnist_train.csv')

print("Dataset Shape:", df_fashion.shape)
print("\nFirst 10 Rows:")
display(df_fashion.head(10))

print("\nClass Distribution:")
print(df_fashion['label'].value_counts().sort_index())

# ---------------------------------------------------------
# Task 2: Separate features & target, Normalize and apply StandardScaler
# ---------------------------------------------------------
X_fashion = df_fashion.drop(columns=['label'])
y_fashion = df_fashion['label']

# Normalize pixel values to [0, 1] then standardize to mean=0, variance=1
X_norm = X_fashion / 255.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

# ---------------------------------------------------------
# Task 3 & 4: Apply PCA using 3 approaches and display metrics
# ---------------------------------------------------------
# Method 1: Fixed 50 components
pca_50 = PCA(n_components=50, random_state=42)
X_pca_50 = pca_50.fit_transform(X_scaled)
var_50 = np.sum(pca_50.explained_variance_ratio_) * 100

# Method 2: 95% Variance
pca_95 = PCA(n_components=0.95, random_state=42)
X_pca_95 = pca_95.fit_transform(X_scaled)
var_95 = np.sum(pca_95.explained_variance_ratio_) * 100

# Method 3: All components
pca_all = PCA(n_components=None, random_state=42)
X_pca_all = pca_all.fit_transform(X_scaled)
var_all = np.sum(pca_all.explained_variance_ratio_) * 100

# Comparison Table (Task 7)
comparison_pca = pd.DataFrame({
    'Approach': ['Method 1: Fixed (n=50)', 'Method 2: 95% Variance (0.95)', 'Method 3: All Components (None)'],
    'Original Features': [X_fashion.shape[1]] * 3,
    'Retained Components': [pca_50.n_components_, pca_95.n_components_, pca_all.n_components_],
    'Cumulative Explained Variance (%)': [round(var_50, 2), round(var_95, 2), round(var_all, 2)]
})
print("\n--- PCA Approaches Comparison Table ---")
display(comparison_pca)

# ---------------------------------------------------------
# Task 5: Plot Cumulative Explained Variance
# ---------------------------------------------------------
cumsum_var = np.cumsum(pca_all.explained_variance_ratio_)

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, color='teal', linewidth=2)
plt.axhline(y=0.95, color='red', linestyle='--', label='95% Explained Variance')
plt.axvline(x=pca_95.n_components_, color='orange', linestyle='--', label=f'{pca_95.n_components_} Components')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---------------------------------------------------------
# Task 6: Visualize dataset in 2D using first two principal components
# ---------------------------------------------------------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_pca_50[:, 0],
    X_pca_50[:, 1],
    c=y_fashion,
    cmap='tab10',
    alpha=0.6,
    s=10
)
plt.colorbar(scatter, label='Clothing Class (0-9)')
plt.title('2D PCA Visualization of Fashion MNIST')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.grid(alpha=0.2)
plt.show()