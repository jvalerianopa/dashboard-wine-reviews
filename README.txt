WINE REVIEWS — DASHBOARD CON LEMATIZACIÓN Y LDA

1. Copia winemag-data-130k-v2.csv en esta carpeta, junto a app.py.
2. Abre CMD en esta carpeta.
3. Ejecuta:
   py -3.10 -m pip install -r requirements.txt
   py -3.10 app.py
4. Abre la dirección local que aparece en la consola (puerto 8060).

Se reemplazó únicamente el gráfico de proyección SVD por un gráfico de tópicos LDA.
Los demás gráficos, filtros, lematización, K-Means, TF-IDF y Sunburst interactivo se conservan.
LDA se entrena con frecuencias de palabras (CountVectorizer), usando cinco tópicos y la columna lematizada.
Los seis términos mostrados por tópico son los de mayor peso dentro de cada tópico. El orden de los tópicos cambia según su prevalencia en los datos filtrados.
