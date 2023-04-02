# LIBRERÍAS
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
# from branca.colormap import LinearColormap

#to make the plotly graphs
import plotly.graph_objs as go
# import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

from plotly.subplots import make_subplots
import plotly.io as pio

from utils.milibreria import *

# CONFIGURACIÓN DE LA PÁGINA# 
layout='centered' or 'wide' 
st.set_page_config(page_title="Manchester", layout="wide", page_icon="🏠")
# Esto es para no mostrar los errores de deprecation en pyplot al usuario
st.set_option("deprecation.showPyplotGlobalUse", False)

df = pd.read_csv("data/df.csv")

# Establecemos la imagen de fondo de la app

# import base64
# import streamlit as st

# def set_background_image(file_path):
#     with open(file_path, "rb") as f:
#         image_bytes = f.read()
#     encoded_image = base64.b64encode(image_bytes).decode()
#     page_bg_img = '''
#         <style>
#             body {
#                 background-image: url("data:Modulo2/16-Data StoryTelling/MANCHESTER/img/manchester1.jpg");
#                 background-size: cover;
#             }
#         </style>
#     ''' % encoded_image
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# PORTADA MANCHESTER
# col1, col2, col3 = st.columns(3)
# with col1:
#      st.title("")
# with col2:
#     st.title("")
# st.image("img/Manchester_portada.png", use_column_width=True)

# with col3:
#     st.title("")

# MANCHESTER - PORTADA
if st.sidebar.button("Airbnb en Manchester"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.title("")
    with col2:
        st.title("")

    st.image("img/Manchester_portada.png", use_column_width=True)
    
    
# CONOCIENDO LA CIUDAD
if st.sidebar.button("Conociendo la ciudad"):
    
    st.markdown("<h3 style='text-align: center;'>CONOCIENDO LA CIUDAD<br><br></h3>", unsafe_allow_html=True)

    with st.expander("Ubicación"):
        st.write("Manchester es una ciudad ubicada en el noroeste de Inglaterra, a unos 260 km de Londres. Conocida por su rica historia industrial, la ciudad ha evolucionado en los últimos años convirtiéndose en un centro cultural y artístico.")
        st.image("img/manchesterlocationmap.jpg", use_column_width=True)

    with st.expander("Historia"):
        st.write("La ciudad de Manchester se originó como un pequeño pueblo romano llamado Mamucium.") 
        st.image("img/manc_edadmedia.jpg", use_column_width=True)
        st.write("Durante la Edad Media, se convirtió en un importante centro de producción de lana y algodón, gracias a su ubicación estratégica cerca de ríos y canales que permitían el transporte de mercancías.") 
        st.image("img/manchester1.jpg", use_column_width=True)
        st.write ("Durante la Revolución Industrial del siglo XIX, Manchester experimentó un gran crecimiento y se convirtió en un centro textil líder en el mundo y fue conocida como 'Cottonopolis' debido a su papel en la producción de algodón y otras telas.")
        st.image("img/Manchester_Cotton_Mill.png", use_column_width=True)
        st.write ("La ciudad también tuvo una importancia fundamental en la lucha por los derechos laborales, siendo el lugar de nacimiento del movimiento sindical británico. ")
        st.image("img/man-strike.jpg", use_column_width=True)
        st.write ("*Foto de 1934, durante la huelga de nueve meses de duración llevada a cabo por 650 trabajadores de una fábrica de alambre en Bradford, Manchester.*")
        st.image("img/emmeline.jpg", use_column_width=True)
        st.write ("*Foto de Emmeline Pankhurst, líder del movimiento sufragista británico*")

    with st.expander("Sitios de interés"):
        st.write("Entre los puntos más importantes de Manchester se encuentra la Catedral de Manchester, que data del siglo XV, así como el Ayuntamiento de Manchester, un edificio impresionante construido en estilo neogótico.")
        st.image("img/manc_cathedral.jpg", use_column_width=True)
        st.image("img/Manchester_Town_Hall.jpg", use_column_width=True)
        st.write("Manchester es conocida mundialmente por sus equipos de fútbol, el Manchester United y el Manchester City. Aquí podemos ver su emblemático estadio de fútbol, el Old Trafford, hogar del Manchester United.")
        st.image("img/old-trafford-manchester.jpg", use_column_width=True)
        st.write("Entre los sitios más emblemáticos de Manchester se encuentra la Chetham Library (Siglo XVII), la biblioteca donde Karl Marx y Friedrich Engels escribieron el Manifiesto Comunista en 1848.")
        st.image("img/chethams_1.jpg", use_column_width=True)
        st.image("img/640px-Marx_and_Engels_in_Berlin.jpg", use_column_width=True)

    with st.expander("Música"):
        st.write("Manchester ha sido una ciudad icónica en la historia musical británica, siendo el epicentro de varios movimientos musicales importantes. En los años 70, surgieron bandas post-punk y new wave como Joy Division, The Fall y Buzzcocks. En los 80, el sonido de Manchester se centró en la música electrónica, con grupos como New Order y The Smiths.")
        st.image("img/manchester-collage.jpg", use_column_width=True)
        st.write("En los años 90, nació el movimiento *Madchester*, que fusionó el rock alternativo con la música electrónica y el dance, destacando bandas como Stone Roses, Happy Mondays y Inspiral Carpets. A finales de los años 90, surgió el Britpop, que tuvo a Oasis como banda líder, junto a otros grupos como Blur y Pulp. En resumen, Manchester ha sido fundamental en la evolución y desarrollo de diversos géneros musicales y ha sido cuna de algunos de los artistas más influyentes de las últimas décadas.")
        st.image("img/bands.jpg", use_column_width=True)

#DATASET

if st.sidebar.button("Dataset"):

    st.markdown("<h3 style='font-size: 16px;'>Este es el dataset con los datos sobre los alojamientos de AirBnB en Manchester:</h3>", unsafe_allow_html=True)

    st.dataframe(df)
    
    # Regulations AirBnB
    st.image("img/regulationsABNB.png", use_column_width=True)
    
    st.plotly_chart(fig5, use_container_width=True)

    st.plotly_chart(fig6, use_container_width=True)
    
if st.sidebar.button("Neighbourhoods"):

    st.header("Neighbourhoods")

    # Mostrar gráficos
    st.plotly_chart(fig, use_container_width=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    #folium_static(map7)
    
if st.sidebar.button("A city for everyone"):
    
    st.markdown("<h3 style='font-size: 16px;'>Teniendo en cuenta los datos de la columna Amenities del dataset de AirBnB en Manchester, hemos extraído algunos alojamientos idóneos para diferentes tipos de viajeros:</h3>", unsafe_allow_html=True)

    # FAMILIES
    st.image("img/kids.png", use_column_width=True)
    
    st.plotly_chart(figFAM1, use_container_width=True)
    
    st.write("El eje x representa el precio de los alojamientos y el eje y muestra los diferentes barrios de Manchester donde se encuentran estos alojamientos para familias con las comodidades de 'cocina', 'cuna', 'patio trasero' y 'niños' (según el filtro que se aplicó previamente). Cada punto en el gráfico representa un alojamiento y su posición en el eje x e y indica su precio y ubicación, respectivamente. Los puntos más a la derecha en el eje x indican alojamientos más caros, mientras que los puntos más arriba en el eje y representan alojamientos en barrios específicos de Manchester. Este tipo de gráfico puede ayudar a identificar patrones en la relación entre el precio y la ubicación de los alojamientos para familias con estas comodidades. Por ejemplo, se pueden observar agrupaciones de puntos que indican que los alojamientos en ciertos barrios tienden a tener precios más altos o más bajos que otros.")

    st.plotly_chart(figFAM2, use_container_width=True)
    
    # st.markdown(map1._repr_html_(), unsafe_allow_html=True)

    # DIVERSIDAD
    
    st.image("img/diversidad.png", use_column_width=True)

    st.plotly_chart(figDIV1, use_container_width=True)

    st.plotly_chart(figDIV1, use_container_width=True)
    
    # st.markdown(map2._repr_html_(), unsafe_allow_html=True)

    # BUSINESS
    
    st.image("img/business.png", use_column_width=True)

    st.plotly_chart(figBS1, use_container_width=True)
    
    st.plotly_chart(figBS2, use_container_width=True)
    
    # st.markdown(map3._repr_html_(), unsafe_allow_html=True)

    # LONG TERM
    
    st.image("img/longterm.png", use_column_width=True)

    st.plotly_chart(figLT1, use_container_width=True)
    
    st.plotly_chart(figLT2, use_container_width=True)

    # st.markdown(map3._repr_html_(), unsafe_allow_html=True)
    
if st.sidebar.button("Reviews"):
    
    st.markdown("<h3 style='font-size: 16px;'>Además de las reseñas escritas, los invitados pueden enviar una calificación de estrellas general y un conjunto de calificaciones de estrellas de categoría. Los huéspedes pueden dar calificaciones sobre: experiencia general, limpieza, precisión, valor, comunicación, llegada y ubicación. A continuación puede ver la distribución de puntajes de todas esas categorías.</h3>", unsafe_allow_html=True)

    st.pyplot(fig9)
    
    st.plotly_chart(fig20)
    
    st.plotly_chart(fig21)
