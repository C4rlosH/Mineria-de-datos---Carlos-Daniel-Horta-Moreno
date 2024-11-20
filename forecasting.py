import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('csv/listings.csv')
df.dropna(inplace=True)
df['host_since'] = pd.to_datetime(df['host_since'], format='%d/%m/%y')
df = df.sort_values('host_since')
df.set_index('host_since', inplace=True)
df['days_since'] = (df.index - df.index.min()).days

# Modelo de regresión lineal
X = df['days_since'].values.reshape(-1, 1)
y = df['price']
model = LinearRegression()
model.fit(X, y)

# Predicción
df['price_pred'] = model.predict(X)

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['price'], label='Real Price')
plt.plot(df.index, df['price_pred'], label='Prediction', linestyle='--')
plt.title('Price prediction based on time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()