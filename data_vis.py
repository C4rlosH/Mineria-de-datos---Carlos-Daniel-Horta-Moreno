import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('csv/listings.csv')

df['host_since'] = pd.to_datetime(df['host_since'], format='%d/%m/%y')

df.dropna(subset=['neighbourhood_cleansed', 'price'], inplace=True)

charts = ['histogram', 'boxplot', 'scatter', 'pie', 'line']

for chart in charts:
    plt.figure(figsize=(10, 6))
    
    if chart == 'histogram':
        sns.histplot(df['price'], bins=30)
        plt.title('Distribución de precios')
        
    elif chart == 'boxplot':
        sns.boxplot(x='room_type', y='price', data=df)
        plt.title('Precio por tipo de habitación')
        
    elif chart == 'scatter':
        sns.scatterplot(x='longitude', y='latitude', hue='price', data=df)
        plt.title('Ubicación de propiedades y precios')
        
    elif chart == 'pie':
        df['room_type'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Distribución de tipos de habitación')
        
    elif chart == 'line':
        df.groupby('host_since')['price'].mean().plot()
        plt.title('Precio medio a lo largo del tiempo')
        
    plt.show()
