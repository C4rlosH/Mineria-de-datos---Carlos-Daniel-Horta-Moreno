import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

df = pd.read_csv('csv/listings.csv')
df.dropna(subset=['latitude', 'longitude', 'accommodates', 'room_type'], inplace=True)

X = df[['latitude', 'longitude', 'accommodates']]
y = df['room_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DATA CLASSIFICATION
# Modelo k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Precisión
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# DATA CLUSTERING
# Modelo k-means
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

# Gráfico de clusters
plt.scatter(df['latitude'], df['longitude'], c=df['cluster'], cmap='viridis')
plt.title('Property Clusters')
plt.show()