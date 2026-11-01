import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ---------- 1. LOAD AND PRE-PROCESS ----------
df = pd.read_csv('diabetes.csv')
print(df.shape)
print(df['Outcome'].value_counts())               # 1 = Diabetic, 0 = Not diabetic

# In these columns, 0 is impossible, so it means "missing". Replace with the median.
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

# ---------- 2. FEATURES, TARGET, SPLIT ----------
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 3. SCALE (KNN uses distances, so scaling is important) ----------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------- 4. KNN MODEL ----------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
p = knn.predict(X_test)

# ---------- 5. EVALUATION ----------
cm = confusion_matrix(y_test, p)
acc = accuracy_score(y_test, p)
print("\nConfusion Matrix:\n", cm)
print("Accuracy   :", round(acc, 4))
print("Error rate :", round(1 - acc, 4))
print("Precision  :", round(precision_score(y_test, p), 4))
print("Recall     :", round(recall_score(y_test, p), 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('KNN Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

# ---------- BONUS: best K ----------
scores = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
          for k in range(1, 21)]
plt.plot(range(1, 21), scores, marker='o'); plt.title('Accuracy vs K')
plt.xlabel('K'); plt.ylabel('Accuracy'); plt.show()

# mkdir ml_exam4
# cd ml_exam4

# Copy diabetes.csv in, run nano diabetes.py, paste, then Ctrl+O, Enter, Ctrl+X. Run with python3 diabetes.py. If a library is missing, use pip3 install pandas numpy matplotlib seaborn scikit-learn (add --break-system-packages if it complains).