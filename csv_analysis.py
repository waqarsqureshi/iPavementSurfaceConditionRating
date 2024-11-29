# Necessary imports for the scripts
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import re
import os
import base64
from datetime import datetime

# Global variables

# Set folder paths and server variables from the main Streamlit app
TEMP_DIR = st.session_state.get('TEMP_DIR', "/home/pms/streamlit-example/temp/")
SERVER_IP = st.session_state.get('SERVER_IP', "192.168.1.80")
PORT = st.session_state.get('PORT', 8502)
#===============================================
# Function to extract chainage value from image name
def extract_chainage(image_name):
    """
    Extract the chainage value from the image name.

    Parameters:
    - image_name: The name of the image.

    Returns:
    - The chainage value as a float or None if not found.
    """
    # Use a regular expression to extract the chainage value for different formats
    match = re.search(r' (\d+) \d+\.(jpg|png)$', image_name)
    if match:
        return float(match.group(1)) / 1000

    match = re.search(r' (\d+\.\d+)\.(jpg|png)$', image_name)
    if match:
        return float(match.group(1)) / 1000

    match = re.search(r' (\d{7})\.jpg$', image_name)
    if match:
        chainage_str = match.group(1)
        return float(chainage_str) / 1000

    return None

#=================================================
def apply_filters(data, window_size):
    """
    Apply the median filter to the data.

    Parameters:
    - data: The input data series that needs to be filtered.
    - window_size: The size of the window for the filter.

    Returns:
    - Filtered data series.
    """
    # Ensure the window size is odd for the median filter
    if window_size % 2 == 0:
        window_size += 1
        st.warning(f"Adjusted window size for Median Filter to {window_size} to ensure it's odd.")
    return np.floor(medfilt(data, kernel_size=window_size))

#==================================================
# Function used if the chainage is in 0000.000 format
def format_chainage(chainage_float):
    formatted_str = "{:.3f}".format(chainage_float).replace('.', '')
    return float(formatted_str)

#=================================================
def determine_chainage_format(chainage):
    """
    Determine the format of the chainage value and format it accordingly.

    Parameters:
    - chainage: The chainage value extracted from the image name.

    Returns:
    - The formatted chainage value.
    """
    if "." in str(chainage):
        return format_chainage(chainage)
    return chainage

#=================================================
def analyze_data(df, window_size, section_size, x_axis_type, folder_path):
    """
    Analyze the data based on user-defined parameters and chosen filter.

    Parameters:
    - df: The input dataframe containing the data.
    - window_size: The size of the window for the filter.
    - section_size: The size of the section for determining the most common rating.
    - x_axis_type: The type of x-axis to be used for plotting ("Image Index" or "Image Chainage").
    - folder_path: The path to the folder where the CSV will be saved.

    Displays:
    - An interactive plot showing the most common pavement rating for each section.
    - A dropdown to select the desired CSV format.
    - A button to save the data to CSV.
    """
    # Apply the chosen filter
    df['Smoothed Prediction'] = apply_filters(df['Prediction 1'], window_size)

    # Determine the most common rating for each user-defined section size
    most_common_ratings = []
    probabilities = []
    for i in range(0, len(df), section_size):
        section_end = min(i + section_size, len(df))
        section = df.iloc[i:section_end]

        # Calculate the weighted mode
        unique_ratings = section['Smoothed Prediction'].unique()
        weighted_counts = {rating: sum(section[section['Smoothed Prediction'] == rating]['Probability 1']) for rating in unique_ratings}
        most_common_rating = max(weighted_counts, key=weighted_counts.get)
        most_common_prob = weighted_counts[most_common_rating] / sum(section['Probability 1'])

        most_common_ratings.extend([most_common_rating] * section_size)
        probabilities.extend([most_common_prob] * section_size)

    # Determine x-axis values based on user's choice
    if x_axis_type == "Image Index":
        x_values = df.index
    else:  # "Distance in KMs"
        x_values = df['Image Name'].apply(extract_chainage)
        if x_values.isnull().any():
            st.error("Some images have missing or invalid chainage values. Cannot plot with respect to Distance in KMs.")
            return

    # Create an interactive plot using Plotly
    st.write(f"### Most Common Pavement Rating for Each {section_size}-Image Section vs {x_axis_type}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=most_common_ratings, 
        mode='markers+lines',
        hovertext=df['Image Name']
    ))
    fig.update_layout(
        title=f'Most Common Pavement Rating for Each {section_size}-Image Section vs {x_axis_type}',
        xaxis_title=x_axis_type,
        yaxis_title='Most Common Pavement Rating',
        yaxis=dict(range=[1, 10], dtick=1)
    )
    st.plotly_chart(fig)

    # Button to save the data to CSV
    csv_format = st.radio("Choose CSV Format:", ["Summary", "Full Ratings"])
    if st.button("Save Data to CSV"):
        csv_data = {}
        if csv_format == "Summary":
            csv_data = {
                "Most Common Rating": [],
                "Start Image Index": [],
                "End Image Index": [],
                "Start Image Name": [],
                "End Image Name": []
            }
            for i in range(0, len(df), section_size):
                section_end = min(i + section_size, len(df))
                section = df.iloc[i:section_end]
                unique_ratings = section['Smoothed Prediction'].unique()
                weighted_counts = {rating: sum(section[section['Smoothed Prediction'] == rating]['Probability 1']) for rating in unique_ratings}
                most_common_rating = max(weighted_counts, key=weighted_counts.get)

                csv_data["Most Common Rating"].append(most_common_rating)
                csv_data["Start Image Index"].append(i)
                csv_data["End Image Index"].append(section_end - 1)
                csv_data["Start Image Name"].append(df.iloc[i]['Image Name'])
                csv_data["End Image Name"].append(df.iloc[section_end - 1]['Image Name'])
            output_csv_name = "output_data_summary.csv"
        else:
            most_common_ratings = [int(float(rating)) for rating in most_common_ratings]
            csv_data = {
                "Image Index": list(df.index),
                "Image Name": list(df['Image Name']),
                "Chainage": [determine_chainage_format(val) for val in df['Image Name'].apply(extract_chainage)],
                "Actual Rating": list(df['Prediction 1']),
                "Smoothed Rating": most_common_ratings,
                "Probability": list(df['Probability 1'])
            }
            max_len = max(len(lst) for lst in csv_data.values())
            for key, lst in csv_data.items():
                if len(lst) < max_len:
                    placeholder = ' ' if isinstance(lst[0], str) else -1
                    lst.extend([placeholder] * (max_len - len(lst)))
            output_csv_name = "output_data_detailed.csv"

        output_path = os.path.join(folder_path, output_csv_name)
        output_df = pd.DataFrame(csv_data)
        output_df.to_csv(output_path, index=False)
        csv_url = f"http://{SERVER_IP}:{PORT}/temp/{os.path.basename(folder_path)}/{output_csv_name}"
        st.markdown(f"[Download the CSV file here]({csv_url})")

#=================================================
def csv_analysis():
    """Main function for Streamlit app."""
    st.title("Pavement Rating Analysis")

    # User-defined parameters
    window_size = st.sidebar.slider("Median Filter Window Size", 1, 11, 3)
    section_size = st.sidebar.slider("Section Size for Most Common Rating", 1, 100, 20)
    x_axis_options = ["Image Index", "Distance in KMs"]
    x_axis_type = st.sidebar.selectbox("Choose x-axis type:", x_axis_options)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        if 'csv_analysis_temp_dir' not in st.session_state:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            csv_analysis_temp_dir = os.path.join(TEMP_DIR, timestamp)
            os.makedirs(csv_analysis_temp_dir)
            file_path = os.path.join(csv_analysis_temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            st.session_state.csv_analysis_temp_dir = csv_analysis_temp_dir
        else:
            file_path = os.path.join(st.session_state.csv_analysis_temp_dir, uploaded_file.name)

        df = pd.read_csv(file_path)

        if all(col in df.columns for col in ['Image Name', 'Prediction 1', 'Probability 1']):
            analyze_data(df, window_size, section_size, x_axis_type, st.session_state.csv_analysis_temp_dir)
        else:
            st.error("The uploaded CSV file does not have the required columns.")
