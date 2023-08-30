import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# Color mapping
color_mapping = {
    '1': [255, 0, 0], '2': [255, 0, 0],
    '3': [255, 0, 0], '4': [255, 165, 0],
    '5': [255, 165, 0], '6': [255, 165, 0],
    '7': [0, 0, 255], '8': [0, 0, 255],
    '9': [0, 128, 0], '10': [0, 128, 0]
}

def extract_chainage(image_name):
    """Extract chainage from image name."""
    parts = image_name.split()
    chainage = parts[1].replace('.', '')
    return int(chainage)

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

def main():
    st.title("GPS Location Viewer with Ratings")

    # Upload the GPS data CSV
    gps_file = st.file_uploader("Upload GPS CSV file", type="csv")
    if gps_file:
        gps_data = pd.read_csv(gps_file)

        # Upload the ratings CSV
        ratings_file = st.file_uploader("Upload Ratings CSV file", type="csv")
        if ratings_file:
            ratings_data = pd.read_csv(ratings_file)

            # Preprocess the ratings data
            preprocessed_ratings = preprocess_ratings_data(ratings_data)

            # Merge the preprocessed ratings data with the GPS data on Chainage
            merged_data = pd.merge(gps_data, preprocessed_ratings, on='Chainage')

            # Assign colors based on ratings
            merged_data['color'] = merged_data['Most Common Rating'].apply(lambda x: color_mapping[str(int(x))])

            # Display the merged data for debugging
            st.write(merged_data[['Chainage', 'Lat', 'Lng', 'Most Common Rating']])

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

if __name__ == "__main__":
    main()


'''
          # Define the layer for the map
            layer = pdk.Layer(
                "LineLayer",
                gps_df,
                get_source_position='[source_lon, source_lat]',
                get_target_position='[target_lon, target_lat]',
                get_color=random_color,
                get_width=5,  # Adjust this value to change the width of the line
                pickable=True,
                opacity=0.8,
            )
'''



#streamlit run gps_viewer.py --server.address 192.168.1.65 --server.port 8503
