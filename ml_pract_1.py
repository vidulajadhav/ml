import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- 1. PRE-PROCESSING ----------
df = pd.read_csv('uber.csv')
df = df.drop(['Unnamed: 0', 'key'], axis=1)      # useless columns
print(df.isnull().sum())                          # check missing values
df = df.dropna()                                  # remove missing rows
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['hour'] = df.pickup_datetime.dt.hour
df['day'] = df.pickup_datetime.dt.dayofweek
df['month'] = df.pickup_datetime.dt.month
df['year'] = df.pickup_datetime.dt.year
df = df.drop('pickup_datetime', axis=1)

# distance (km) from lat/long using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

df['distance'] = haversine(df.pickup_latitude, df.pickup_longitude,
                           df.dropoff_latitude, df.dropoff_longitude)

# remove impossible values
df = df[(df.fare_amount > 0) & (df.distance > 0) & (df.passenger_count.between(1, 6))]
df = df[df.pickup_latitude.between(-90, 90) & df.dropoff_latitude.between(-90, 90)]
df = df[df.pickup_longitude.between(-180, 180) & df.dropoff_longitude.between(-180, 180)]

# ---------- 2. OUTLIERS ----------
sns.boxplot(data=df[['fare_amount', 'distance']]); plt.title('Before'); plt.show()

def remove_outliers(d, col):                      # IQR method
    q1, q3 = d[col].quantile([.25, .75])
    iqr = q3 - q1
    return d[d[col].between(q1 - 1.5*iqr, q3 + 1.5*iqr)]

for c in ['fare_amount', 'distance']:
    df = remove_outliers(df, c)

sns.boxplot(data=df[['fare_amount', 'distance']]); plt.title('After'); plt.show()

# ---------- 3. CORRELATION ----------
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# ---------- 4. MODELS ----------
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {'Linear Regression': LinearRegression(),
          'Random Forest': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)}

# ---------- 5. EVALUATION ----------
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    print(f"\n{name}")
    print("R2   :", round(r2_score(y_test, p), 4))
    print("RMSE :", round(np.sqrt(mean_squared_error(y_test, p)), 4))
    print("MAE  :", round(mean_absolute_error(y_test, p), 4))