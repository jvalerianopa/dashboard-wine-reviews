import os, glob, re
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

px.defaults.template = "simple_white"
# Buscar el CSV junto a este archivo, no en la carpeta desde la que se abrió CMD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
files = [
    "winemag-data-130k-v2.csv",
    "winemag-data_first150k.csv",
    "wine-reviews.csv",
]
path = next((os.path.join(BASE_DIR, name) for name in files
             if os.path.isfile(os.path.join(BASE_DIR, name))), None)
if path is None:
    raise FileNotFoundError(
        "No se encontró el CSV de Wine Reviews. Copia winemag-data-130k-v2.csv "
        "en la misma carpeta que app.py. No se cargarán CSV auxiliares para "
        "evitar resultados incorrectos."
    )
print(f"CSV encontrado: {path}")

# Leer por bloques y conservar una muestra aleatoria uniforme de hasta 4.000
# reseñas válidas. Así no se carga todo el CSV en memoria antes de muestrear.
COLUMNAS_NECESARIAS = {
    "country", "description", "points", "price",
    "title", "variety", "province"
}
TAMANO_MUESTRA = 4000
TAMANO_BLOQUE = 5000
rng = np.random.default_rng(42)
reservoir = []
vistos = 0
columnas_csv = None

for bloque in pd.read_csv(
    path,
    usecols=lambda c: str(c).strip().lower().replace(" ", "_") in COLUMNAS_NECESARIAS,
    chunksize=TAMANO_BLOQUE,
    low_memory=True,
    encoding="utf-8"
):
    bloque.columns = [str(c).strip().lower().replace(" ", "_") for c in bloque.columns]
    if "description" not in bloque.columns:
        raise ValueError("El CSV debe contener la columna description.")
    if columnas_csv is None:
        columnas_csv = list(bloque.columns)

    # Asegurar columnas esperadas aunque alguna no exista en el CSV.
    for c in ["country", "variety", "title", "province"]:
        if c not in bloque.columns:
            bloque[c] = "Desconocido"
    for c in ["points", "price"]:
        if c not in bloque.columns:
            bloque[c] = np.nan
    bloque["description"] = bloque["description"].fillna("").astype(str)
    bloque = bloque[bloque["description"].str.strip().ne("")]

    # Algoritmo reservoir sampling: cada reseña válida tiene la misma
    # probabilidad de formar parte de la muestra final.
    if columnas_csv is None:
        columnas_csv = list(bloque.columns)
    for fila in bloque[columnas_csv].itertuples(index=False, name=None):
        vistos += 1
        if len(reservoir) < TAMANO_MUESTRA:
            reservoir.append(fila)
        else:
            j = int(rng.integers(0, vistos))
            if j < TAMANO_MUESTRA:
                reservoir[j] = fila

if not reservoir:
    raise ValueError("El CSV no contiene reseñas con descripción válida.")

df = pd.DataFrame(reservoir, columns=columnas_csv)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
for c in ["country", "variety", "title", "province"]:
    if c not in df:
        df[c] = "Desconocido"
    df[c] = df[c].fillna("Desconocido").astype(str)
for c in ["points", "price"]:
    if c not in df:
        df[c] = np.nan
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["description"] = df["description"].fillna("").astype(str)
print(f"Muestra aleatoria cargada: {len(df):,} reseñas válidas de {vistos:,} revisadas.")

# Lematización en inglés para el análisis de texto. Se conserva description
# sin modificar para sentimiento, tablas emergentes y demás datos originales.
try:
    import nltk
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    recursos = [
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for ruta_recurso, nombre_recurso in recursos:
        try:
            nltk.data.find(ruta_recurso)
        except LookupError:
            print(f"Descargando recurso NLTK necesario: {nombre_recurso}...")
            if not nltk.download(nombre_recurso, quiet=False):
                raise RuntimeError(f"No se pudo descargar el recurso NLTK {nombre_recurso}.")

    lemmatizer = WordNetLemmatizer()
    def etiqueta_wordnet(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def lematizar_descripcion(texto):
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", texto.lower())
        etiquetados = pos_tag(tokens, lang="eng")
        lemas = [lemmatizer.lemmatize(token, etiqueta_wordnet(tag))
                 for token, tag in etiquetados]
        return " ".join(lemas)

    df["description_clean"] = df["description"].map(lematizar_descripcion)
    print("Lematización activada: TF-IDF, LDA y K-Means usarán description_clean.")
except Exception as exc:
    raise RuntimeError(
        "No se pudo preparar la lematización. Verifica la instalación de nltk "
        "y que haya conexión a Internet para descargar los recursos WordNet y "
        "averaged_perceptron_tagger_eng. No se ha continuado sin lematizar."
    ) from exc

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df["score_sent"] = df.description.map(lambda t: analyzer.polarity_scores(t)["compound"])
except Exception:
    positive = {"elegant","fresh","rich","beautiful","excellent","balanced","delicious","lovely","complex","smooth","ripe","bright","silky","great"}
    negative = {"bitter","harsh","thin","weak","bland","rough","sour","poor","tart","flat","disappointing","dry"}
    def lex(text):
        words = re.findall(r"[a-z]+", text.lower())
        return float(np.clip((sum(w in positive for w in words)-sum(w in negative for w in words))/max(1, len(words)**0.5), -1, 1))
    df["score_sent"] = df.description.map(lex)
df["sentimiento"] = np.select([df.score_sent >= .05, df.score_sent <= -.05], ["Positivo", "Negativo"], default="Neutral")
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1,2), min_df=3, max_df=.85, sublinear_tf=True)
X = vectorizer.fit_transform(df.description_clean)
df["cluster"] = KMeans(n_clusters=5, random_state=42, n_init=3, max_iter=100).fit_predict(X).astype(str)
terms = vectorizer.get_feature_names_out(); means = np.asarray(X.mean(axis=0)).ravel(); ix = np.argsort(means)[-15:][::-1]
words = pd.DataFrame({"palabra": terms[ix], "peso_tfidf": means[ix]})

# LDA necesita frecuencias de términos (conteos), no pesos TF-IDF.
# Se entrena sobre la misma columna lematizada para que ambos análisis sean coherentes.
count_vectorizer = CountVectorizer(stop_words="english", max_features=1000,
                                   ngram_range=(1, 2), min_df=3, max_df=0.85)
X_counts = count_vectorizer.fit_transform(df.description_clean)
lda = LatentDirichletAllocation(n_components=5, random_state=42,
                                learning_method="batch", max_iter=8,
                                doc_topic_prior=None, topic_word_prior=None)
doc_topic = lda.fit_transform(X_counts)
lda_terms = count_vectorizer.get_feature_names_out()
print("Modelo LDA entrenado: 5 tópicos sobre las descripciones lematizadas.")
countries = sorted(df.country.unique()); varieties = sorted(df.variety.unique())
pmax = max(1, int(df.price.quantile(.95))) if df.price.notna().any() else 1000
AZUL, NARANJA = "#2B6CB0", "#E69F00"
app = Dash(__name__); app.title = "Wine Reviews — Dashboard"
def card(label, ident):
    return html.Div([html.Small(label, style={"color":"#475569"}), html.H3(id=ident, style={"margin":"4px 0"})], style={"flex":"1", "padding":"8px 14px", "background":"#E7EDF4", "borderRadius":"4px", "minWidth":"150px"})
app.layout = html.Div(style={"fontFamily":"Arial,sans-serif", "padding":"8px", "background":"white", "color":"#111827"}, children=[
    html.H3("Wine Reviews: puntuaciones, sentimientos y patrones de texto", style={"margin":"4px 0 10px"}),
    dcc.Store(id="sun_country_selected", data=None),
    html.Div([card("reseñas analizadas", "k_n"), card("puntuación promedio", "k_points"), card("sentimiento positivo", "k_pos")], style={"display":"flex", "gap":"10px", "flexWrap":"wrap"}),
    html.Div(style={"display":"flex", "alignItems":"stretch", "gap":"10px", "marginTop":"8px"}, children=[
        dcc.Graph(id="g_points", style={"width":"72%", "height":"300px"}),
        html.Div(style={"width":"28%", "minWidth":"230px", "padding":"10px"}, children=[
            html.B("País"), dcc.Dropdown(id="f_country", options=[{"label":x,"value":x} for x in countries], placeholder="Todos los países", clearable=True),
            html.Br(), html.B("Variedad"), dcc.Dropdown(id="f_variety", options=[{"label":x,"value":x} for x in varieties], placeholder="Todas las variedades", clearable=True),
            html.Br(), html.B("Precio máximo (USD)"), dcc.Slider(id="f_price", min=0, max=pmax, value=pmax, step=1, marks=None, tooltip={"placement":"bottom","always_visible":True}),
            html.P("Los filtros actualizan todos los gráficos. Selecciona un país y una variedad para comparar patrones.", style={"fontSize":"12px","color":"#6B7280"})
        ])
    ]),
    html.Div([
        html.Button("Ver todos los países", id="reset_sun", n_clicks=0,
                    style={"margin":"4px 0 8px", "padding":"6px 12px", "cursor":"pointer"}),
        html.Span(id="sun_status", style={"marginLeft":"10px", "color":"#64748b", "fontSize":"13px"})
    ]),
    html.Div(style={"display":"flex","gap":"8px", "flexWrap":"wrap"}, children=[
        dcc.Graph(id="g_sun", style={"width":"32%","height":"320px","minWidth":"280px"},
                  clear_on_unhover=True),
        dcc.Graph(id="g_scatter", style={"width":"36%","height":"320px","minWidth":"300px"}),
        dcc.Graph(id="g_sent", style={"width":"30%","height":"320px","minWidth":"260px"})
    ]),
    html.Div(style={"display":"flex","gap":"8px","flexWrap":"wrap"}, children=[
        dcc.Graph(id="g_words", style={"width":"49%","height":"360px","minWidth":"320px"}),
        dcc.Graph(id="g_clusters", style={"width":"49%","height":"540px","minWidth":"320px"})
    ]),
    html.Div(style={"display":"flex", "gap":"8px", "marginTop":"8px"}, children=[
        dcc.Graph(id="g_cluster_terms", style={"width":"100%", "height":"520px"})
    ]),

])
def filtered(country, variety, price):
    d = df
    if country: d = d[d.country == country]
    if variety: d = d[d.variety == variety]
    if price is not None: d = d[d.price.isna() | (d.price <= price)]
    return d
@app.callback(
    Output("sun_country_selected", "data"),
    Input("g_sun", "clickData"),
    Input("reset_sun", "n_clicks"),
    State("sun_country_selected", "data"),
    prevent_initial_call=True
)
def select_sun_country(click_data, reset_clicks, selected_country):
    # El botón siempre restablece todos los países.
    from dash import ctx
    if ctx.triggered_id == "reset_sun":
        return None

    if not click_data or not click_data.get("points"):
        return selected_country

    label = str(click_data["points"][0].get("label", ""))
    # Solo reaccionar a países; un clic en una variedad no cambia el país.
    if label not in set(df["country"].unique()):
        return selected_country

    # Segundo clic en el mismo país vuelve a la vista general.
    if selected_country == label:
        return None
    return label


def filtered(country, variety, price):
    d = df
    if country:
        d = d[d.country == country]
    if variety:
        d = d[d.variety == variety]
    if price is not None:
        d = d[d.price.isna() | (d.price <= price)]
    return d


@app.callback(
    Output("k_n","children"),
    Output("k_points","children"),
    Output("k_pos","children"),
    Output("g_points","figure"),
    Output("g_sun","figure"),
    Output("g_scatter","figure"),
    Output("g_sent","figure"),
    Output("g_words","figure"),
    Output("g_clusters","figure"),
    Output("g_cluster_terms","figure"),
    Output("sun_status","children"),
    Input("sun_country_selected","data"),
    Input("f_country","value"),
    Input("f_variety","value"),
    Input("f_price","value")
)
def update(sun_country, dropdown_country, variety, price):
    # El clic del sunburst tiene prioridad; si no hay país seleccionado, usar el desplegable.
    country = sun_country or dropdown_country
    d = filtered(country, variety, price)
    if d.empty:
        empty = px.scatter(title="No hay datos con estos filtros")
        return "0", "N/D", "N/D", empty, empty, empty, empty, empty, empty, empty, "No hay datos para los filtros seleccionados."

    n = f"{len(d):,}"
    avg = f"{d.points.mean():.2f}" if d.points.notna().any() else "N/D"
    pos = f"{(d.sentimiento == 'Positivo').mean():.1%}"

    # Histograma de puntuaciones enteras: una barra por puntuación.
    point_data = d.dropna(subset=["points"]).copy()
    f_points = px.histogram(
        point_data, x="points", nbins=max(1, int(point_data["points"].max() - point_data["points"].min() + 1)),
        title="¿Cómo se distribuyen las puntuaciones?",
        labels={"points":"Puntuación", "count":"Reseñas"}
    )
    if point_data["points"].nunique() > 1:
        f_points.update_traces(xbins=dict(
            start=float(point_data["points"].min()) - 0.5,
            end=float(point_data["points"].max()) + 0.5,
            size=1
        ))
    f_points.add_vline(x=d.points.mean(), line_dash="dot", line_color=NARANJA,
                       annotation_text=f"Promedio {avg}")

    # Sunburst de dos niveles: país -> 8 categorías (7 variedades + Otros).
    # Para cada país se agrupan las variedades menos frecuentes en "Otras variedades".
    base_sun = df.copy()
    # El sunburst muestra solo el país seleccionado; sin selección muestra todos.
    if country:
        base_sun = base_sun[base_sun.country == country]
    if variety:
        base_sun = base_sun[base_sun.variety == variety]
    if price is not None:
        base_sun = base_sun[base_sun.price.isna() | (base_sun.price <= price)]

    pair_counts = (
        base_sun.groupby(["country", "variety"], dropna=False)
        .size().reset_index(name="reseñas")
    )
    if not pair_counts.empty:
        pair_counts["rank"] = pair_counts.groupby("country")["reseñas"].rank(
            method="first", ascending=False
        )
        pair_counts["categoria"] = np.where(
            pair_counts["rank"] <= 7,
            pair_counts["variety"],
            "Otras variedades"
        )
        sun_data = (
            pair_counts.groupby(["country", "categoria"], as_index=False)["reseñas"]
            .sum()
        )
        f_sun = px.sunburst(
            sun_data, path=["country", "categoria"], values="reseñas",
            title="¿Qué países concentran más reseñas? Haz clic en un país para filtrar",
            maxdepth=2
        )
        f_sun.update_traces(
            insidetextorientation="radial",
            hovertemplate="%{label}<br>Reseñas: %{value:,}<extra></extra>"
        )
    else:
        f_sun = px.sunburst(title="No hay países para mostrar")

    scatter_data = d.dropna(subset=["points","price"])
    scatter_data = scatter_data.sample(min(1200, len(scatter_data)), random_state=42) if len(scatter_data) else scatter_data
    f_scatter = px.scatter(
        scatter_data, x="price", y="points", color="cluster", opacity=.55,
        hover_data=["country","variety","title"],
        title="¿Cómo se relacionan precio y puntuación? (muestra)",
        labels={"price":"Precio (USD)","points":"Puntuación","cluster":"Clúster"}
    )

    sent = d.sentimiento.value_counts().reindex(["Positivo","Neutral","Negativo"], fill_value=0).rename_axis("sentimiento").reset_index(name="reseñas")
    f_sent = px.bar(
        sent, x="reseñas", y="sentimiento", orientation="h",
        title="¿Qué sentimiento predomina?",
        labels={"reseñas":"Número de reseñas","sentimiento":""},
        color="sentimiento",
        color_discrete_map={"Positivo":AZUL,"Neutral":"#A0AEC0","Negativo":NARANJA}
    )

    # Mantener los términos calculados sobre la muestra original, como en el dashboard correcto.
    w = words.sort_values("peso_tfidf").tail(15)
    f_words = px.bar(w, x="peso_tfidf", y="palabra", orientation="h",
                     title="¿Qué palabras y bigramas destacan por TF-IDF?",
                     labels={"peso_tfidf":"Peso TF-IDF","palabra":"Término"})

    # Reemplazo del gráfico SVD por un gráfico de tópicos LDA.
    # Los términos describen cada tópico; el orden de los tópicos se actualiza
    # según su proporción promedio entre las reseñas que dejan los filtros.
    posiciones = d.index.to_numpy()
    proporciones = doc_topic[posiciones].mean(axis=0) if len(posiciones) else np.zeros(lda.n_components)
    orden_topicos = np.argsort(proporciones)[::-1]
    topico_rows = []
    for topic_idx in orden_topicos:
        pesos = lda.components_[topic_idx]
        top_idx = pesos.argsort()[-6:][::-1]
        for term_idx in top_idx:
            topico_rows.append({
                "tópico": f"Tópico {topic_idx + 1}",
                "término": lda_terms[term_idx],
                "etiqueta": f"T{topic_idx + 1}: {lda_terms[term_idx]}",
                "peso": float(pesos[term_idx] / pesos.sum()),
                "prevalencia": float(proporciones[topic_idx])
            })
    topics_df = pd.DataFrame(topico_rows)
    f_clusters = px.bar(
        topics_df, x="peso", y="etiqueta", color="tópico", orientation="h",
        barmode="overlay",
        title="¿Qué temas aparecen en las reseñas? Tópicos LDA (6 términos por tema)",
        labels={"peso":"Peso del término en el tópico", "etiqueta":"Tópico: término", "tópico":"Tema"},
        hover_data={"término":True, "tópico":True, "peso":":.4f", "prevalencia":":.1%", "etiqueta":False}
    )
    f_clusters.update_layout(
        height=520, bargap=0.18,
        yaxis={"type":"category", "categoryorder":"array", "categoryarray":topics_df["etiqueta"].tolist(), "automargin":True},
        legend_title_text="Tópico",
        margin=dict(t=65, l=150, r=25, b=45)
    )

    # Palabras y bigramas representativos de CADA clúster, calculados
    # usando los documentos que permanecen tras aplicar los filtros actuales.
    # d conserva los índices originales de df, por lo que podemos seleccionar
    # exactamente las filas correspondientes de la matriz TF-IDF X.
    term_rows = []
    feature_names = vectorizer.get_feature_names_out()
    row_positions = d.index.to_numpy()
    X_filtered = X[row_positions]
    d_clusters = d["cluster"].astype(str).to_numpy()
    for cluster_id in sorted(d["cluster"].astype(str).unique(), key=lambda value: int(value)):
        local_mask = d_clusters == cluster_id
        if not local_mask.any():
            continue
        mean_tfidf = np.asarray(X_filtered[local_mask].mean(axis=0)).ravel()
        top_indices = mean_tfidf.argsort()[-8:][::-1]
        for idx in top_indices:
            if mean_tfidf[idx] > 0:
                term = feature_names[idx]
                term_rows.append({
                    "clúster": f"Clúster {cluster_id}",
                    "término": term,
                    "etiqueta": f"C{cluster_id}: {term}",
                    "peso_tfidf": float(mean_tfidf[idx])
                })
    if term_rows:
        terms_by_cluster = pd.DataFrame(term_rows).sort_values(["clúster", "peso_tfidf"])
        # Cada combinación clúster-término tiene una etiqueta única (por ejemplo, C0: fruit).
        # Usamos barmode="overlay" para evitar los huecos que "group" deja cuando
        # una categoría del eje Y solo tiene una barra perteneciente a un clúster.
        terms_by_cluster = terms_by_cluster.reset_index(drop=True)
        etiquetas_ordenadas = terms_by_cluster["etiqueta"].tolist()
        altura_terminos = max(560, 28 * len(terms_by_cluster) + 120)
        f_cluster_terms = px.bar(
            terms_by_cluster, x="peso_tfidf", y="etiqueta", color="clúster",
            orientation="h", barmode="overlay",
            title="¿Qué palabras y bigramas caracterizan cada clúster? (hasta 8 términos por grupo)",
            labels={"peso_tfidf":"TF-IDF promedio dentro del clúster", "etiqueta":"Clúster: término", "clúster":"Grupo"},
            hover_data={"término":True, "clúster":True, "peso_tfidf":":.4f", "etiqueta":False}
        )
        f_cluster_terms.update_layout(
            height=altura_terminos,
            bargap=0.18,
            yaxis={
                "type":"category",
                "categoryorder":"array",
                "categoryarray":etiquetas_ordenadas,
                "automargin":True,
                "title":"Clúster: término"
            },
            margin=dict(l=190, r=30, t=65, b=45),
            showlegend=True
        )
    else:
        f_cluster_terms = px.bar(title="No hay términos representativos para los filtros seleccionados")

    for fig in [f_points, f_sun, f_scatter, f_sent, f_words, f_clusters, f_cluster_terms]:
        fig.update_layout(margin=dict(t=55,l=10,r=10,b=10), legend=dict(font=dict(size=10)))

    status = f"País seleccionado: {country}" if country else "Vista general: todos los países"
    return n, avg, pos, f_points, f_sun, f_scatter, f_sent, f_words, f_clusters, f_cluster_terms, status

#if __name__ == "__main__":
#    app.run(debug=True, host="127.0.0.1", port=8062)
server = app.server

if __name__ == "__main__":
    import os
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8062))
    )