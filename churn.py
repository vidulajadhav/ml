import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- 1. READ THE DATASET ----------
df = pd.read_csv('Churn_Modelling.csv')
print(df.shape)
print(df['Exited'].value_counts())               # 1 = Leaves, 0 = Stays
print("Missing values:", df.isnull().sum().sum())

# ---------- 2. FEATURES, TARGET, SPLIT ----------
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)   # useless columns
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)  # text -> numbers

X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 3. NORMALIZE ----------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)               # learn from train only
X_test = sc.transform(X_test)                     # apply same scaling to test

# ---------- 4. BUILD MODEL ----------
# Basic model: one small hidden layer
base = MLPClassifier(hidden_layer_sizes=(6,), max_iter=300, random_state=42)
base.fit(X_train, y_train)
p1 = base.predict(X_test)
print("\nBasic model accuracy   :", round(accuracy_score(y_test, p1), 4))

# Improved model: more layers/neurons, early stopping, regularization
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                      alpha=0.001, early_stopping=True, max_iter=500, random_state=42)
model.fit(X_train, y_train)
p2 = model.predict(X_test)

# ---------- 5. EVALUATION ----------
print("Improved model accuracy:", round(accuracy_score(y_test, p2), 4))
cm = confusion_matrix(y_test, p2)
print("\nConfusion Matrix:\n", cm)
print(classification_report(y_test, p2, target_names=['Stays', 'Leaves']))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stays', 'Leaves'], yticklabels=['Stays', 'Leaves'])
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

plt.plot(model.loss_curve_); plt.title('Training Loss'); plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.show()

# Copy Churn_Modelling.csv into it, run nano churn.py, paste the code, and save with Ctrl+O, Enter, Ctrl+X. Then:

# python3 churn.py

# If a library is missing: pip3 install pandas numpy matplotlib seaborn scikit-learn (add --break-system-packages if it complains).

# 6. The “improvements” (they will ask this)

# Step 4 of the question says to identify points of improvement. In the code, the improved model changes:

# More hidden layers and neurons (32, 16) instead of (6,), so it can learn more complex patterns.
# ReLU activation in the hidden layers, which trains faster and avoids vanishing gradients.
# Adam optimizer, an adaptive learning-rate optimizer.
# Early stopping, which stops training when the validation score stops improving and prevents overfitting.
# L2 regularization (alpha), which penalizes large weights.