"""
=========================================================================
 Radiohead Explorer — Web App de análisis de audio y letras
 -----------------------------------------------------------------------
 Práctica universitaria: visualización interactiva del catálogo de
 estudio de Radiohead combinando features de Spotify y análisis lírico.
 Autor: Rubo
=========================================================================
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================================
#  CONFIGURACIÓN GLOBAL DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Radiohead Explorer",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Orden cronológico canónico de los 9 álbumes de estudio
ALBUM_ORDER = [
    'Pablo Honey', 'The Bends', 'OK Computer', 'Kid A', 'Amnesiac',
    'Hail to the Thief', 'In Rainbows', 'The King of Limbs',
    'A Moon Shaped Pool',
]

# ============================================================================
#  CARGA DE DATOS (con caché de Streamlit → solo se ejecuta una vez)
# ============================================================================
@st.cache_data
def load_data():
    """Carga los CSVs pre-computados desde la carpeta data/."""
    base = Path(__file__).parent / "data"
    audio = pd.read_csv(base / "radiohead_studio.csv")
    lyrics = pd.read_csv(base / "radiohead_lyrics.csv")
    # Tipado: el orden cronológico se garantiza con Categorical
    audio['short_album_name'] = pd.Categorical(
        audio['short_album_name'], categories=ALBUM_ORDER, ordered=True
    )
    lyrics['album'] = pd.Categorical(
        lyrics['album'], categories=ALBUM_ORDER, ordered=True
    )
    return audio, lyrics

audio_df, lyrics_df = load_data()

# ============================================================================
#  SIDEBAR — controles interactivos
# ============================================================================
st.sidebar.title("🎛️ Controles")
st.sidebar.markdown(
    "Filtra los álbumes y las gráficas de toda la app se actualizan en vivo."
)

selected_albums = st.sidebar.multiselect(
    "Álbumes a incluir",
    options=ALBUM_ORDER,
    default=ALBUM_ORDER,
    help="Por defecto se muestran los 9 álbumes de estudio.",
)

if not selected_albums:
    st.sidebar.warning("Selecciona al menos un álbum.")
    st.stop()

# Aplicamos el filtro a ambos datasets
audio_f  = audio_df[audio_df['short_album_name'].isin(selected_albums)].copy()
lyrics_f = lyrics_df[lyrics_df['album'].isin(selected_albums)].copy()

# Métrica auxiliar mostrada en la sidebar
st.sidebar.divider()
st.sidebar.metric("Pistas totales", len(audio_f))
st.sidebar.metric("Pistas con letra disponible",
                  int(lyrics_f['has_lyrics'].sum()))

st.sidebar.divider()
st.sidebar.caption(
    "📊 Datos: [Spotify 1.2M Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)  \n"
    "🎤 Letras: [lyrics.ovh](https://lyrics.ovh)  \n"
    "💬 Sentiment: VADER (NLTK)"
)

# ============================================================================
#  CABECERA
# ============================================================================
st.title("🎸 Radiohead Explorer")
st.markdown(
    "Análisis interactivo de las **features de audio** y las **letras** "
    "de los 9 álbumes de estudio de Radiohead (1993 – 2016)."
)

# KPIs de cabecera
c1, c2, c3, c4 = st.columns(4)
c1.metric("Álbumes seleccionados", len(selected_albums))
c2.metric("Canciones",            len(audio_f))
c3.metric("Sentimiento medio",    f"{lyrics_f['compound'].mean():.3f}",
          help="VADER compound score (rango [-1, +1])")
c4.metric("Duración media",       f"{audio_f['duration_ms'].mean()/60000:.2f} min")

st.divider()

# ============================================================================
#  TABS PRINCIPALES
# ============================================================================
tab_audio, tab_letras, tab_data = st.tabs(
    ["🎶 Features de audio", "🎤 Análisis de letras", "📋 Tabla de datos"]
)

# ----------------------------------------------------------------------------
#  TAB 1 — Features de audio
# ----------------------------------------------------------------------------
with tab_audio:
    st.subheader("Mapa sonoro: Acousticness vs Valence")
    st.caption(
        "Cada burbuja es una canción. **Tamaño** = duración, **color** = álbum. "
        "Las esquinas inferiores izquierdas son canciones tristes y eléctricas."
    )

    fig_scatter = px.scatter(
        audio_f, x='valence', y='acousticness',
        color='short_album_name',
        category_orders={'short_album_name': selected_albums},
        size='duration_ms', size_max=35,
        hover_data={'name': True, 'year': True, 'duration_ms': False},
        labels={'valence': 'Valence (positividad musical)',
                'acousticness': 'Acousticness',
                'short_album_name': 'Álbum'},
        height=600,
    )
    fig_scatter.update_layout(template='plotly_white', legend_title_text='Álbum')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # --- Radar chart por álbum ---
    st.subheader("Huella sonora por álbum (radar)")
    st.caption(
        "Media de cada feature por álbum, normalizada al rango [0, 1] "
        "para poder comparar magnitudes muy distintas en un mismo eje."
    )

    radar_features = ['acousticness', 'danceability', 'energy',
                      'instrumentalness', 'liveness', 'valence']

    means = audio_f.groupby('short_album_name', observed=True)[radar_features].mean()
    means = means.loc[[a for a in selected_albums if a in means.index]]
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)

    fig_radar = go.Figure()
    for album, row in norm.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row.tolist() + [row.tolist()[0]],
            theta=radar_features + [radar_features[0]],
            name=album, fill='toself', opacity=0.35,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=600, template='plotly_white',
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # --- Pistas por álbum ---
    st.subheader("Número de pistas por álbum")
    counts = (audio_f.groupby('short_album_name', observed=True)
                     .size()
                     .reindex([a for a in selected_albums]))
    fig_count = px.bar(
        x=counts.values, y=counts.index, orientation='h',
        labels={'x': 'Número de pistas', 'y': ''},
        color=counts.values, color_continuous_scale='Blues',
    )
    fig_count.update_layout(template='plotly_white', height=400,
                            coloraxis_showscale=False,
                            yaxis={'categoryorder': 'array',
                                   'categoryarray': list(reversed(selected_albums))})
    st.plotly_chart(fig_count, use_container_width=True)

# ----------------------------------------------------------------------------
#  TAB 2 — Análisis de letras
# ----------------------------------------------------------------------------
with tab_letras:
    st.subheader("Evolución del sentimiento lírico por álbum")
    st.caption(
        "**Barras** = sentimiento medio del álbum (rojo si es negativo, "
        "verde si es positivo).  **Puntos** = canciones individuales (hover "
        "para ver el título).  Línea discontinua = sentimiento neutro."
    )

    ld = lyrics_f[lyrics_f['has_lyrics']].copy()
    if len(ld) == 0:
        st.warning("No hay letras disponibles para los álbumes seleccionados.")
    else:
        album_mean = ld.groupby('album', observed=True)['compound'].mean()
        album_std  = ld.groupby('album', observed=True)['compound'].std().fillna(0)
        album_mean = album_mean.reindex([a for a in selected_albums if a in album_mean.index])
        album_std  = album_std.reindex(album_mean.index)

        bar_colors = ['#d62728' if v < 0 else '#2ca02c' for v in album_mean]

        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(
            x=album_mean.index.astype(str), y=album_mean.values,
            error_y=dict(type='data', array=album_std.values, color='#444'),
            marker_color=bar_colors, opacity=0.55,
            name='Media por álbum',
            hovertemplate='<b>%{x}</b><br>Sentimiento: %{y:.3f}<extra></extra>',
        ))
        fig_sent.add_trace(go.Scatter(
            x=ld['album'].astype(str), y=ld['compound'], mode='markers',
            marker=dict(size=9, color=ld['compound'], colorscale='RdYlGn',
                        cmin=-1, cmax=1, line=dict(width=0.5, color='black'),
                        colorbar=dict(title='Compound')),
            text=ld['name'], name='Canciones',
            hovertemplate='<b>%{text}</b><br>Álbum: %{x}<br>Sentimiento: %{y:.3f}<extra></extra>',
        ))
        fig_sent.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_sent.update_layout(
            xaxis_title='Álbum', yaxis_title='Compound score (VADER)',
            yaxis_range=[-1.05, 1.05], height=550,
            template='plotly_white', hovermode='closest',
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    st.divider()

    # --- Word cloud reactivo ---
    st.subheader("Nube de palabras por álbum")
    st.caption("Selecciona un álbum para regenerar la nube de palabras.")

    album_choice = st.selectbox(
        "Álbum",
        options=[a for a in selected_albums if a in lyrics_f['album'].unique()],
        index=0,
    )

    # Stopwords inline (no necesitamos descargar NLTK en producción)
    BASE_STOP = {
        'the','a','an','and','or','but','if','then','of','to','in','on',
        'at','for','with','by','from','as','is','are','was','were','be',
        'been','being','have','has','had','do','does','did','will','would',
        'could','should','may','might','must','can','i','you','he','she',
        'it','we','they','me','him','her','us','them','my','your','his',
        'its','our','their','this','that','these','those','what','which',
        'who','whom','where','when','why','how','all','some','any','no',
        'not','only','own','same','so','than','too','very','s','t','don',
        'now','oh','yeah','la','na','hey','mm','ah','uh','ooh','whoa',
        'ya','gonna','wanna','gotta','just','like','get','got','one',
        'know','say','said','see','let','make','made','want','take',
        'come','go','went','back','still',
    }

    def tokenize(text: str) -> list[str]:
        return [w for w in re.findall(r"\b[a-zA-Z']+\b", str(text).lower())
                if w not in BASE_STOP and len(w) > 2]

    text_album = ' '.join(
        lyrics_f.loc[lyrics_f['album'] == album_choice, 'lyrics'].fillna('')
    )
    tokens = tokenize(text_album)

    col_wc, col_top = st.columns([2, 1])

    with col_wc:
        if len(tokens) > 0:
            wc = WordCloud(
                width=900, height=500, background_color='white',
                colormap='viridis', max_words=80, relative_scaling=0.4,
            ).generate(' '.join(tokens))
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"{album_choice}", fontsize=14, fontweight='bold')
            st.pyplot(fig_wc)
        else:
            st.info("No hay suficientes letras para generar una nube.")

    with col_top:
        st.markdown(f"**Top 15 palabras — {album_choice}**")
        top = Counter(tokens).most_common(15)
        if top:
            top_df = pd.DataFrame(top, columns=['palabra', 'frecuencia'])
            fig_top = px.bar(
                top_df, x='frecuencia', y='palabra', orientation='h',
                color='frecuencia', color_continuous_scale='Viridis',
            )
            fig_top.update_layout(
                template='plotly_white', height=500,
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_top, use_container_width=True)

    st.divider()

    # --- Tabla resumen por álbum ---
    st.subheader("Estadísticas líricas por álbum")
    summary = (
        lyrics_f[lyrics_f['has_lyrics']]
        .groupby('album', observed=True)
        .agg(canciones        = ('name', 'count'),
             sentim_medio     = ('compound', 'mean'),
             sentim_std       = ('compound', 'std'),
             palabras_medias  = ('word_count', 'mean'),
             riqueza_lexica   = ('lexical_richness', 'mean'))
        .round(3)
    )
    st.dataframe(summary, use_container_width=True)

# ----------------------------------------------------------------------------
#  TAB 3 — Tabla de datos cruda (transparencia para el lector)
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Audio features (filtradas)")
    st.dataframe(audio_f, use_container_width=True, height=300)

    st.subheader("Letras y sentimiento (filtradas)")
    cols_show = ['name', 'album', 'year', 'has_lyrics',
                 'compound', 'word_count', 'lexical_richness']
    st.dataframe(lyrics_f[cols_show], use_container_width=True, height=300)

# ============================================================================
#  PIE DE PÁGINA
# ============================================================================
st.divider()
st.caption(
    "Práctica universitaria — Análisis de datos musicales con Spotify y "
    "lyrics.ovh.  Construido con Streamlit + Plotly."
)
