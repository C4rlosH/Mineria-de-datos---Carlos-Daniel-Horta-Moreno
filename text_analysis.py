import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('csv/listings.csv')
df.dropna(subset = ['host_name'], inplace=True)

# Concatenar todos los nombres en un solo texto
text = ' '.join(df['host_name'].astype(str).tolist())

# Crear la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

