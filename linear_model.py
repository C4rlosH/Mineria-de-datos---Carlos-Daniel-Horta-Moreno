import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('csv/listings.csv')
df.dropna(subset=['latitude', 'longitude', 'accommodates', 'price'], inplace=True)

# Variables independientes y dependientes
X = df[['latitude', 'longitude', 'accommodates']]
y = df['price']

# Modelo lineal
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# R² score
r2 = r2_score(y, y_pred)
print('R² score:', r2)

# Gráfico de predicción
plt.scatter(y, y_pred)
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Precio real vs Precio predicho')
plt.show()
