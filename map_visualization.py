import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.express as px
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# Color mapping
color_mapping = {
    '1': [255, 0, 0], '2': [255, 0, 0],
    '3': [255, 0, 0], '4': [255, 165, 0],
    '5': [255, 165, 0], '6': [255, 165, 0],
    '7': [0, 0, 255], '8': [0, 0, 255],
    '9': [0, 128, 0], '10': [0, 128, 0]
}
#========================================================
def extract_chainage(image_name):
    """Extract chainage from image name."""
    parts = image_name.split()
    chainage = parts[1].replace('.', '')
    return int(chainage)
#=======================================================
# This function is used if the summary csv file is used.
def preprocess_ratings_data(ratings_data):
    """Preprocess the ratings data to generate a format similar to the GPS data."""
    rows = []
    for _, row in ratings_data.iterrows():
        start_chainage = extract_chainage(row['Start Image Name'])
        end_chainage = extract_chainage(row['End Image Name'])
        for chainage in range(start_chainage, end_chainage + 5, 5):  # Assuming each image represents 5 meters
            rows.append({
                'Chainage': chainage,
                'Most Common Rating': row['Most Common Rating']
            })
    return pd.DataFrame(rows)
#=================================================
def plot_ratings_bar_chart(data, color_mapping):
    # Convert the color mapping to a format suitable for Plotly
    color_scale = []
    for rating, color in color_mapping.items():
        normalized_color = [x/255 for x in color]
        color_scale.append((float(rating)/10, f'rgb({normalized_color[0]},{normalized_color[1]},{normalized_color[2]})'))

    # Plotting a bar graph using Plotly with custom color scale
    fig = px.bar(data, x='Chainage', y='Image Rating', color='Image Rating', 
                 labels={'Image Rating': 'Rating', 'Chainage': 'Sections'}, 
                 title='Ratings per Section', color_continuous_scale=color_scale)

    st.plotly_chart(fig)
#==================================================
def map_visualization():
    st.title("GPS Location Viewer with Ratings")

    # Upload the GPS data CSV
    gps_file = st.file_uploader("Upload GPS CSV file", type="csv")
    if gps_file:
        gps_data = pd.read_csv(gps_file)
        total_km = gps_data ['Chainage'].max()/1000
        # Upload the ratings CSV
        ratings_file = st.file_uploader("Upload Ratings CSV file", type="csv")
        if ratings_file:
            ratings_data = pd.read_csv(ratings_file)
            
            # Warning about the Image Index starting from zero
            st.warning("Please note: The Image Index in the uploaded Ratings CSV starts from zero.")
            # Convert 'Chainage' columns to int64
            gps_data['Chainage'] = gps_data['Chainage'].astype('int64')
            ratings_data['Chainage'] = ratings_data['Chainage'].astype('int64')

            # Merge the ratings data with the GPS data on Chainage
            merged_data = pd.merge(gps_data, ratings_data, left_on='Chainage', right_on='Chainage')

            # Assign colors based on ratings
            merged_data['color'] = merged_data['Image Rating'].apply(lambda x: color_mapping[str(int(x))])
            # Display the merged data for debugging
            
            st.write(f"Total KM Covered: {total_km}")
            st.write(merged_data[['Chainage', 'Lat', 'Lng', 'Image Rating']])
            plot_ratings_bar_chart(merged_data, color_mapping)

            # Define the layer for the map
            layer = pdk.Layer(
                "ScatterplotLayer",
                merged_data,
                get_position='[Lng, Lat]',
                get_color='color',
                get_radius=20,  # Adjust this value to change the size of the markers
                pickable=True,
                opacity=0.8,
            )

            # Define the view for the map
            view_state = pdk.ViewState(
                latitude=merged_data['Lat'].mean(),
                longitude=merged_data['Lng'].mean(),
                zoom=13,
                pitch=0,
            )

            # Render the map using pydeck within Streamlit
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
            ))
