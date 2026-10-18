from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 1. Load dataset (aap yahan apni dataset file ka path de dein)
df = pd.read_csv('income_evaluation.csv')

# Chota sample lein taake fast run ho (optional, agar full data chalana hai toh yeh line hata sakte hain)
df = df.sample(n=5000, random_state=101)

# 2. Preprocessing
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

X = df.drop('income_ >50K', axis=1)
y = df['income_ >50K']

# Train-test split (Yahan X_train aur y_train define ho rahay hain)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# 3. Define individual classifiers
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(random_state=101)
svm_clf = SVC()

# 4. Create and fit Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard',
)
voting_clf.fit(X_train, y_train)

# 5. Evaluate and compare accuracies[cite: 2]
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(clf.__class__.__name__, accuracy_score(y_test, y_pred))