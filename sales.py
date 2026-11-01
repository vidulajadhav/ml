import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------- 1. LOAD AND PRE-PROCESS ----------
df = pd.read_csv('sales_data_sample.csv', encoding='latin1')   # latin1 avoids a decode error
print(df.shape)
print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Use only the numeric columns that describe sales behaviour
X = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP']]
X = X.dropna()

# ---------- 2. SCALE ----------
X_scaled = StandardScaler().fit_transform(X)

# ---------- 3. ELBOW METHOD ----------
wcss = []
for k in range(1, 11):
    wcss.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method'); plt.xlabel('Number of clusters (K)'); plt.ylabel('WCSS (inertia)')
plt.show()

# ---------- 4. K-MEANS WITH CHOSEN K ----------
k = 3                                              # read this from the elbow graph
km = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
X = X.copy()
X['Cluster'] = labels
print("\nCluster sizes:\n", X['Cluster'].value_counts())
print("\nCluster averages:\n", X.groupby('Cluster').mean().round(2))
print("\nSilhouette score:", round(silhouette_score(X_scaled, labels), 4))

plt.scatter(X['QUANTITYORDERED'], X['SALES'], c=labels, cmap='viridis', s=15)
plt.title('K-Means Clusters'); plt.xlabel('QUANTITYORDERED'); plt.ylabel('SALES')
plt.show()

# ---------- 5. HIERARCHICAL (optional second method) ----------
sample = X_scaled[:200]                            # small sample so the dendrogram is readable
dendrogram(linkage(sample, method='ward'))
plt.title('Dendrogram'); plt.xlabel('Samples'); plt.ylabel('Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
print("Hierarchical cluster sizes:\n", pd.Series(hc.fit_predict(X_scaled)).value_counts())

# mkdir ml_exam5
# cd ml_exam5

# Copy the CSV in, run nano sales.py, paste, then Ctrl+O, Enter, Ctrl+X. Run with python3 sales.py. If a library is missing, use pip3 install pandas numpy matplotlib seaborn scikit-learn scipy (add --break-system-packages if it complains).