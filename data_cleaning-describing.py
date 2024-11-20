import pandas as pd
import numpy as np

# 1) Data Cleaning
# Carga el dataset 
df = pd.read_csv('csv/listings.csv')

# Mostrar un resumen de los datos 
print(df.info())

# Limpiar datos nulos o faltantes 
df.dropna(inplace=True)

# Convertir fechas 
df['host_since'] = pd.to_datetime(df['host_since'], format='%d/%m/%y')

# Ver los primeros 5 registros después de la limpieza 
print(df.head())


# 2) Descriptive Statistics
# Estadísticas descriptivas
print(df.describe())

# Agrupar datos y obtener estadísticas
# Seleccionar solo las columnas numéricas 
numeric_columns = df.select_dtypes(include=[np.number]).columns 
# Agrupar por 'neighbourhood_cleansed' y calcular la media solo de las columnas numéricas 
grouped = df.groupby('neighbourhood_cleansed')[numeric_columns].mean() 
# Mostrar las primeras filas del agrupamiento 
print(grouped.head())
