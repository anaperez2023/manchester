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

from streamlit_folium import folium_static

# Defino DF
df = pd.read_csv("data/df.csv")
calendar = pd.read_csv('http://data.insideairbnb.com/united-kingdom/england/greater-manchester/2022-12-27/data/calendar.csv.gz')

# GRÁFICAS

# NEIGHBOURHOOD
# Gráfica N1: Listings by neighbourhood

    # Contar número de alojamientos por barrio
feq = df['neighbourhood'].value_counts()

    # Generar gráfico de barras
fig = px.bar(feq,
                orientation='h',
                width=1000,
                height=800,
                color_discrete_sequence=['#DA291C'],
                title='Number of listings by neighbourhood',
                labels={'value': 'Number of listings', 'index': 'Neighbourhood'},
                template='simple_white',
                category_orders={'index': feq.index},
                )

fig.update_traces(marker_line_color='#FBE122',
                    marker_line_width=1,
                    opacity=0.8,
                    width=0.8)

fig.update_layout(xaxis=dict(showgrid=False, zeroline=False))

fig.update_yaxes(tickfont=dict(size=12))

fig.update_layout(plot_bgcolor='white')

fig.update_layout(title={
                        'text': "Number of listings by neighbourhood",
                        'x':0.5,
                        'xanchor': 'center',
                        'font':{'size': 24}})


# Gráfica N2: Room type by neighbourhood

color_list = ['#FBE122', '#DA291C', '#000000', '#918181']

fig2 = px.histogram(df, x="neighbourhood", color="room_type", barmode="group",
                    color_discrete_sequence=color_list)

fig2.update_layout(xaxis=dict(showgrid=False, zeroline=False))

fig2.update_layout(plot_bgcolor='white')

fig2.update_layout(title={
                    'text': "Room type by neighbourhood",
                    'x':0.5,
                    'xanchor': 'center',
                    'font':{'size': 24}})

fig2.show(renderer='colab')

# Gráfica N3: 
feq = df['accommodates'].value_counts().sort_index()

fig6 = px.bar(feq, x=feq.index, y=feq.values, labels={'x': 'Accommodates', 'y': 'Number of listings'},
              color_discrete_sequence=['#000000'])

fig6.update_layout(
    font=dict(size=12),
    width=800,
    height=600,
    showlegend=False
)
fig6.update_layout(title={
                    'text': "Accommodates (number of people)",
                    'x':0.5,
                    'xanchor': 'center',
                    'font':{'size': 24}})


fig6.show(renderer = 'colab')

# Mapa N1: Average price by neighbourhood: ¡¡¡¡ARREGLAR!!!!

map7 = folium.Map(location=[53.4808, -2.2426], zoom_start=11)
mcter = gpd.read_file("data/neighbourhoods.geojson")
folium.GeoJson(data=mcter,
               name='Manchester',
               tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                                      labels=True,
                                                      sticky=True),
               style_function= lambda feature: {
                   'fillColor': get_color(feature),
                   'color': 'black',
                   'weight': 1,
                   'dashArray': '5, 5',
                   'fillOpacity':0.5
                   },
               highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map7)


# Gráfica N4: Tipos de propiedades:

prop = df.groupby(['property_type','room_type']).room_type.count()
prop = prop.unstack()
prop['total'] = prop.iloc[:,0:3].sum(axis = 1)
prop = prop.sort_values(by=['total'])
prop = prop[prop['total']>=100]
prop = prop.drop(columns=['total'])

fig5 = px.bar(prop, orientation='h', color_discrete_sequence=['#FBE122'],
              width=900, height=500)

fig5.update_layout(title={
                    'text': "Property types in Manchester",
                    'x':0.5,
                    'xanchor': 'center',
                    'font':{'size': 24}})


fig5.update_layout(showlegend=False)

fig5.show(renderer = 'colab')

# Mapa N2: Average price by neighbourhood (map) ¡¡¡¡ARREGLAR!!!!

# feq = df[df['accommodates']==2] 
# feq = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True) 
# mcter = gpd.read_file("/content/drive/MyDrive/BOOTCAMP_Data_Analytics/MANCHESTER/data/neighbourhoods.geojson") 
# feq = pd.DataFrame([feq]) 
# feq = feq.transpose() 

# mcter = pd.merge(mcter, feq, on='neighbourhood', how='left') 
# mcter.rename(columns={'price': 'average_price'}, inplace=True) 
# mcter.average_price = mcter.average_price.round(decimals=0) 
# mcter = mcter.dropna(subset=['average_price']) 

# map_dict = mcter.set_index('neighbourhood')['average_price'].to_dict() 
# color_scale = LinearColormap(['#da291c','#fbe122','#918181'], vmin=min(map_dict.values()), vmax=max(map_dict.values()), caption='Average price') 
# def get_color(feature): 
#   value = map_dict.get(feature['properties']['neighbourhood'])
#   if value is None: 
#     return '#WHATEVER' 
#   else: 
#     return color_scale(value) 

# map10 = folium.Map(location=[53.4808, -2.2426], zoom_start=11) 
# folium.GeoJson(data=mcter, 
#                name='Manchester', 
#                tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'], 
#                                                       labels=True, 
#                                                       sticky=False), 
#                style_function= lambda feature: { 
#                    'fillColor': get_color(feature), 
#                    'color': 'black', 
#                    'weight': 1, 
#                    'dashArray': '5, 5', 
#                    'fillOpacity':0.9 
#                    }, 
#                highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map10) 
# map10.add_child(color_scale)


# A CITY FOR EVERYONE

# 1. FAMILIES WITH KIDS

# Gráfica FAM1 (precio menor 400e):

familias_df = df.loc[df['amenities'].str.contains('kitchen', case=False) & df['amenities'].str.contains('crib', case=False) & df['amenities'].str.contains('backyard', case=False) & df['amenities'].str.contains('children', case=False)]
familias_df

# Filtro los datos por precios menores o iguales a 400 euros
familias_df1= familias_df.query('price <= 400')

# Crear figura para precios menores o iguales a 400 euros
figFAM1 = px.scatter(familias_df1, x='price', y='neighbourhood')

figFAM1.update_layout(
    title={
        'text': "Ideal accommodations for families: Children amenities + Backyard + Crib + Kitchen (Prices <= 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 20}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figFAM1.update_traces(
    name='Comodidades',
    marker=dict(color='red')
)

figFAM1.show(renderer = 'colab')


# Gráfica FAM2 (precio mayor 400e):

# Valores filtro Familias < 400€# Filtrar los datos por precios mayores a 400 euros
familias_df2 = familias_df.query('price > 400')

# Crear figura para precios mayores a 400 euros
figFAM2 = px.scatter(familias_df2, x='price', y='neighbourhood')

figFAM2.update_layout(
    title={
        'text': "Ideal accommodations for families: Children amenities + Backyard + Crib + Kitchen (Prices > 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 20}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figFAM2.update_traces(
    name='Comodidades',
    marker=dict(color='red')
)

figFAM2.show(renderer='colab')

# Mapa Familias:

# # Crear mapa   ¡¡¡¡ ARREGLAR !!! 
# map1 = folium.Map(location=[53.4808, -2.2426], zoom_start=11)

# # Uso FastMarkerCluster para agrupar los marcadores
# marker_cluster1 = FastMarkerCluster([], name='marker_cluster1')

# # Iterar sobre cada fila de familias_df y añadir un marcador al objeto MarkerCluster
# for index, row in familias_df.iterrows():
#     folium.Marker(
#         location=[row['latitude'], row['longitude']],
#         tooltip=row['name'],
#         icon=folium.features.CustomIcon('img/family_icon.png', icon_size=(30, 30))
#     ).add_to(marker_cluster1)

# # Añadir el objeto MarkerCluster al mapa
# marker_cluster1.add_to(map1)

# # Añadir el control de capas al mapa
# folium.LayerControl().add_to(map1)

# 2. PEOPLE WITH DISABILITIES

# Gráfica DIV1 (precio menor 400e):

diversidad_df = df[df['amenities'].str.contains('elevator', case=False) | df['amenities'].str.contains('single level', case=False)]
diversidad_df 

diversidad_df1 = diversidad_df.query('price <= 400')

# Crear figura para precios menores o iguales a 400 euros
figDIV1 = px.scatter(diversidad_df1, x='price', y='neighbourhood')

figDIV1.update_layout(
    title={
        'text': "Ideal accommodations for people with disabilities: Elevator or Single level accomodations (Prices <= 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 16}
    },
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figDIV1.update_traces(
    name='Comodidades',
    marker=dict(color='black')
)

# Gráfica DIV2 (precio mayor 400e):

# Filtro los datos por precios menores o iguales a 400 euros
diversidad_df2= diversidad_df.query('price > 400')

# Crear figura para precios menores o iguales a 400 euros
figDIV2 = px.scatter(diversidad_df2, x='price', y='neighbourhood')

figDIV2.update_layout(
    title={
        'text': "Ideal accommodations for people with disabilities: Elevator or Single level (Prices > 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 18}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figDIV2.update_traces(
    name='Comodidades',
    marker=dict(color='black')
)

figDIV2.show(renderer = 'colab')

# Mapa Diversidad: ¡¡¡ ARREGLAR !!!

# map2 = folium.Map(location=[53.4808, -2.2426], zoom_start=11) 

# # uso MarkerCluster para agrupar los marcadores
# marker_cluster2 = FastMarkerCluster([], name='marker_cluster2')

# # Iterar sobre cada fila de diversidad_df1 y añadir un marcador al objeto MarkerCluster
# for index, row in diversidad_df.iterrows():
#     folium.Marker(
#         location=[row['latitude'], row['longitude']],
#         tooltip=row['name'],
#         icon=folium.features.CustomIcon('/content/drive/MyDrive/BOOTCAMP_Data_Analytics/MANCHESTER/img/disability_icon-min.png', icon_size=(25, 25))
#     ).add_to(marker_cluster2)

# # Añadir el objeto MarkerCluster al mapa
# marker_cluster2.add_to(map2)

# # Mostrar el mapa
# map2

# 3. BUSINESS TRAVELLERS

business_df = df[df['amenities'].str.contains('wifi', case=False) & df['amenities'].str.contains('dedicated workspace', case=False)]

# Gráfica BS1 (precio menor 400e):

# Filtro los datos por precios menores o iguales a 400 euros
business_df1= business_df.query('price <= 400')

# Crear figura para precios menores o iguales a 400 euros
figBS1 = px.scatter(business_df1, x='price', y='neighbourhood')

figBS1.update_layout(
    plot_bgcolor='grey',
    title={
        'text': "Ideal accommodations for business travelers: Wifi + Dedicated Workspace (Prices <= 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 16}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figBS1.update_traces(
    name='Comodidades',
    marker=dict(color='yellow')
)

# Gráfica BS2 (precio mayor 400e):

# Filtro los datos por precios menores o iguales a 400 euros
business_df2= business_df.query('price > 400')

# Crear figura para precios menores o iguales a 400 euros
figBS2 = px.scatter(business_df2, x='price', y='neighbourhood')

figBS2.update_layout(
    plot_bgcolor='grey',
    title={
        'text': "Ideal accommodations for business travelers: Wifi + Dedicated Workspace (Prices > 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 18}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figBS2.update_traces(
    name='Comodidades',
    marker=dict(color='yellow')
)

# # Mapa BUSINESS: ¡¡¡ ARREGLAR !!!

# map3 = folium.Map(location=[53.4808, -2.2426], zoom_start=11) 

# marker_cluster3 = FastMarkerCluster([], name='marker_cluster3')

# # Iterar sobre cada fila de business_df y añadir un marcador al objeto MarkerCluster
# for index, row in business_df.iterrows():
#     folium.Marker(
#         location=[row['latitude'], row['longitude']],
#         tooltip=row['name'],
#         icon=folium.features.CustomIcon('/content/drive/MyDrive/BOOTCAMP_Data_Analytics/MANCHESTER/img/business_trip.png', icon_size= (30, 30))
#     ).add_to(marker_cluster3)

# # Añadir el objeto MarkerCluster al mapa
# marker_cluster3.add_to(map3)

# # Mostrar el mapa
# map3

# 4. LONG TERM STAYS

longterm_df = df[df['amenities'].str.contains('long term', case=False)]

# Gráfica LT1 (precio menor 400e):

longterm_df1= longterm_df.query('price <= 400')

figLT1 = px.scatter(longterm_df1, x='price', y='neighbourhood')

figLT1.update_layout(
    plot_bgcolor='grey',
    title={
        'text': "Ideal accommodations for long-term stays (Prices <= 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 20}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figLT1.update_traces(
    name='Comodidades',
    marker=dict(color='red')
)

# Gráfica LT2 (precio mayor 400e):

longterm_df2= longterm_df.query('price > 400')

# Crear figura para precios menores o iguales a 400 euros
figLT2 = px.scatter(longterm_df2, x='price', y='neighbourhood')

figLT2.update_layout(
    plot_bgcolor='grey',
    title={
        'text': "Ideal accommodations for long-term stays (Prices > 400€)",
        'x': 0.5,
        'xanchor': 'center',
        'font':{'size': 20}},
    xaxis_title='Price',
    yaxis_title='Neighbourhood'
)

figLT2.update_traces(
    name='Comodidades',
    marker=dict(color='red')
)

# # Mapa long term: ¡¡¡ ARREGLAR !!!

# map4 = folium.Map(location=[53.4808, -2.2426], zoom_start=11) 

# marker_cluster4 = FastMarkerCluster([], name='marker_cluster4')

# # Iterar sobre cada fila de business_df y añadir un marcador al objeto MarkerCluster
# for index, row in longterm_df.iterrows():
#     folium.Marker(
#         location=[row['latitude'], row['longitude']],
#         tooltip=row['name'],
#         icon=folium.features.CustomIcon('/content/drive/MyDrive/BOOTCAMP_Data_Analytics/MANCHESTER/img/family_icon.png', icon_size= (30, 30))
#     ).add_to(marker_cluster4)

# # Añadir el objeto MarkerCluster al mapa
# marker_cluster4.add_to(map4)

# # Mostrar el mapa
# map4


# REVIEWS

# Gráficas limpieza, comunicación, etc

listings_details = pd.read_csv('data/listings_details.csv')

# Filter the listings with at least 10 reviews
listings10 = listings_details[listings_details['number_of_reviews'] >= 10]

# Create a figure with subplots
fig9, axs = plt.subplots(2, 3, figsize=(20, 15))
fig9.suptitle('Average Review Scores by Category', fontsize=24)

# Define the review categories
categories = ['review_scores_location', 'review_scores_cleanliness', 'review_scores_value',
              'review_scores_communication', 'review_scores_checkin', 'review_scores_accuracy']
colors = ['#DA291C', '#FBE122', 'k', '#DA291C', '#FBE122', 'k']
titles = ['Location', 'Cleanliness', 'Value', 'Communication', 'Arrival', 'Accuracy']

# Plot each category
for i in range(len(categories)):
    # Calculate the frequency of each score
    freq = listings10[categories[i]].value_counts().sort_index()
    # Plot the bar graph in the corresponding subplot
    axs[i//3, i%3].bar(freq.index, freq.values, color=colors[i])
    # Set the title and axis labels
    axs[i//3, i%3].set_title(titles[i], fontsize=18)
    axs[i//3, i%3].set_xlabel('Average Review Score', fontsize=14)
    axs[i//3, i%3].set_ylabel('Number of Listings', fontsize=14)
    # Set the font size of the tick labels
    axs[i//3, i%3].tick_params(axis='both', labelsize=12)

# Adjust the spacing between the subplots
fig9.tight_layout(pad=3.0)

# Average review score location

# # Leer el archivo de datos
# @st.cache
# def load_data():
#     df = pd.read_csv("data/listings.csv")
#     return df

# df = load_data()

# # Filtrar por listados con al menos 60 reseñas
# df_filtered = df[df['number_of_reviews'] >= 60]

# # Agrupar por barrio y obtener la media de las reseñas de ubicación
# feq20 = df_filtered.groupby('neighbourhood')['review_scores_location'].mean().sort_values(ascending=True)

# # Definir los colores para la gráfica
# colors = [(251, 225, 34),(145,129,129),(218, 41, 28)]
# colors = ['#%02x%02x%02x' % c for c in colors]

# # Crear la gráfica con Plotly Express
# fig20 = px.bar(feq20, orientation='h', color=feq20.values, color_continuous_scale=colors)

# # Configurar el diseño de la gráfica
# fig20.update_layout(title_text='Average review score location (at least 10 reviews)', xaxis_title='Score (scale 1-10)', yaxis_title='')

# fig20.update_layout(xaxis_tickfont_size=16, yaxis_tickfont_size=16)

# # Filtrar por alojamientos con capacidad para dos personas
# df_filtered = df[df['accommodates'] == 2]

# # Filtrar por barrios con al menos 20 alojamientos
# df_filtered = df_filtered[df_filtered.groupby('neighbourhood')['neighbourhood'].transform('size') >= 20]

# # Agrupar por barrio y obtener la media de los precios
# feq21 = df_filtered.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)

# # Crear la segunda gráfica con Plotly Express
# fig21 = px.bar(feq21, orientation='h', color=feq21.values, color_continuous_scale=colors)

# # Configurar el diseño de la segunda gráfica
# fig21.update_layout(title_text='Average daily price for a 2-person accommodation with at least 5 entries', xaxis_title='Average daily price (Euro)', yaxis_title='')

# fig21.update_layout(xaxis_tickfont_size=16, yaxis_tickfont_size=20)

#  # HOSTS

# # Replace 't' and 'f' with 'True' and 'False'
# listings_details.host_is_superhost = listings_details.host_is_superhost.replace({"t": "True", "f": "False"})

# # Count values and plot bar chart
# feq30 = listings_details['host_is_superhost'].value_counts()
# fig30, ax = plt.subplots(figsize=(10, 8))
# ax.bar(feq30.index, feq30.values, width=0.5, color='#DA291C')
# ax.set_title("Number of listings with Superhost", fontsize=20)
# ax.set_ylabel('Number of listings', fontsize=12)
# ax.tick_params(axis='x', labelsize=12, rotation=0)
# ax.tick_params(axis='y', labelsize=12)

# # OTRA HOST

# # Load data
# listings_details = pd.read_csv("listings_details.csv")
# listings10 = listings_details[listings_details['number_of_reviews'] >= 10]

# # Clean data
# listings_details.host_is_superhost = listings_details.host_is_superhost.replace({"t": "True", "f": "False"})

# # Plot bar chart with Manchester red color
# fig40, ax1 = plt.subplots(figsize=(10, 8))
# feq40 = listings_details['host_is_superhost'].value_counts()
# ax1.bar(feq40.index, feq40.values, width=0.5, color='#da291c')
# ax1.set_title("Number of listings with Superhost", fontsize=20)
# ax1.set_ylabel('Number of listings', fontsize=12)
# ax1.tick_params(axis='both', labelsize=16)

# # Plot histograms with Manchester red color
# fig50, (ax2, ax3) = plt.subplots(ncols=2, figsize=(20, 10))
# ax2.hist(listings10['host_response_rate'].dropna(), color='#da291c')
# ax2.set_title("Response rate (at least 10 reviews)", fontsize=20)
# ax2.set_ylabel("number of listings")
# ax2.set_xlabel("percent", fontsize=20)
# ax2.tick_params(axis='both', labelsize=16)

# feq50 = listings10['host_response_time'].value_counts()
# ax3.bar(feq50.index, feq50.values, color='#da291c', width=0.5)
# ax3.set_title("Response time (at least 10 reviews)", fontsize=20)
# ax3.set_ylabel("number of listings")
# ax3.tick_params(axis='both', labelsize=16)
# plt.xticks(rotation=45)
