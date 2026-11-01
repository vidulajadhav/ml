import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# ---------- 1. LOAD AND PRE-PROCESS ----------
df = pd.read_csv('emails.csv')
print(df.shape)
print(df['Prediction'].value_counts())           # 0 = Not Spam, 1 = Spam
print("Missing values:", df.isnull().sum().sum())
df = df.dropna().drop('Email No.', axis=1)       # remove useless ID column

X = df.drop('Prediction', axis=1)                # 3000 word-count columns
y = df['Prediction']                             # target

sns.countplot(x=y); plt.title('0 = Not Spam, 1 = Spam'); plt.show()

# ---------- 2. TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------- 3. MODELS ----------
models = {'KNN': KNeighborsClassifier(n_neighbors=5),
          'SVM': SVC(kernel='linear')}

# ---------- 4. EVALUATION ----------
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    print(f"\n===== {name} =====")
    print("Accuracy :", round(accuracy_score(y_test, p), 4))
    print("Precision:", round(precision_score(y_test, p), 4))
    print("Recall   :", round(recall_score(y_test, p), 4))
    print("F1 score :", round(f1_score(y_test, p), 4))
    print(classification_report(y_test, p, target_names=['Not Spam', 'Spam']))
    sns.heatmap(confusion_matrix(y_test, p), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    plt.title(name + ' Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

# Copy emails.csv into it, then run nano spam.py, paste the code, and save with Ctrl+O, Enter, Ctrl+X. Check the libraries with:

# python3 -c "import pandas, seaborn, sklearn; print('OK')"

# Then run python3 spam.py. If a library is missing, use pip3 install pandas numpy matplotlib seaborn scikit-learn (add --break-system-packages if it complains).