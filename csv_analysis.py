# necessary imports for the scripts
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
#===============================================
def extract_chainage(image_name):
    """
    Extract the chainage value from the image name.

    Parameters:
    - image_name: The name of the image.

    Returns:
    - The chainage value as a float or None if not found.
    """
    # Use a regular expression to extract the chainage value for the format "LOOP1A 0000.000 1.jpg"
    match = re.search(r' (\d+\.\d+) ', image_name)
    if match:
        return float(match.group(1))

    # Use a regular expression to extract the chainage value for the format "LOOP1A 0000000.jpg"
    match = re.search(r' (\d{7})\.jpg$', image_name)
    if match:
        # Convert the extracted value to the float format
        chainage_str = match.group(1)
        return float(chainage_str)/1000

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

    # Median Filter:
    # This filter replaces each data point with the median of neighboring data points defined by 'window_size'.
    # It's particularly effective against 'salt and pepper' noise in an image. It preserves edges while removing noise.
    
    # Ensure the window size is odd for the median filter.
    # An odd window size ensures that there's a clear median value.
    if window_size % 2 == 0:
        window_size += 1
        st.warning(f"Adjusted window size for Median Filter to {window_size} to ensure it's odd.")
    return np.floor(medfilt(data, kernel_size=window_size))
#==================================================
#This function is used if the chainage is in this format i.e. 0000.000
# do not use it if 0000000 format is used.
def format_chainage(chainage_float):
    # Format the float to have three digits after the decimal point
    formatted_str = "{:.3f}".format(chainage_float)
    # Replace the decimal point
    formatted_str = formatted_str.replace('.', '')
    # Convert back to float and return
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
    # If the chainage contains a decimal point, format it
    if "." in str(chainage):
        return format_chainage(chainage)
    return chainage
#=================================================

def analyze_data(df, prob_threshold, window_size, section_size, x_axis_type, folder_path):
    """
    Analyze the data based on user-defined parameters and chosen filter.

    Parameters:
    - df: The input dataframe containing the data.
    - prob_threshold: The probability threshold for adjusting predictions.
    - window_size: The size of the window for the filter.
    - section_size: The size of the section for determining the most common rating.

    - x_axis_type: The type of x-axis to be used for plotting ("Image Index" or "Image Chainage").
    - folder_path: The path to the folder where the CSV will be saved.

    Displays:
    - An interactive plot showing the most common pavement rating for each section.
    - A dropdown to select the desired CSV format.
    - A button to save the data to CSV.
    """

    # Adjust predictions based on probability threshold
    df['Adjusted Prediction 1'] = df['Prediction 1'].where(df['Probability 1'] >= prob_threshold, df['Prediction 1'].shift())

    # Apply the chosen filter
    df['Smoothed Prediction'] = apply_filters(df['Adjusted Prediction 1'], window_size)

    # Determine the most common rating for each user-defined section size
    #############################
    #most_common_ratings = []
    #for i in range(0, len(df), section_size):
        #section_end = min(i + section_size, len(df))
        #mode_series = df.iloc[i:section_end]['Smoothed Prediction'].mode()
        #most_common_rating = mode_series.iloc[0] if not mode_series.empty else df.iloc[i]['Smoothed Prediction']
        #most_common_ratings.extend([most_common_rating]*section_size)
    ############################
#################################
    most_common_ratings = []

    for i in range(0, len(df), section_size):
        section_end = min(i + section_size, len(df))
        
        # Extract the section of interest
        section = df.iloc[i:section_end]
        
        # Calculate the weighted mode
        unique_ratings = section['Smoothed Prediction'].unique()
        weighted_counts = {rating: 0 for rating in unique_ratings}
        
        for rating in unique_ratings:
            weighted_counts[rating] = sum(section[section['Smoothed Prediction'] == rating]['Probability 1'])
        
        most_common_rating = max(weighted_counts, key=weighted_counts.get)
        
        most_common_ratings.extend([most_common_rating] * section_size)

################################

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
    
    # Initialize a Plotly scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=most_common_ratings, 
        mode='markers+lines',
        hovertext=df['Image Name']  # Display the image name when hovering over a data point
    ))
    
    # Update the layout of the plot
    fig.update_layout(
        title=f'Most Common Pavement Rating for Each {section_size}-Image Section vs {x_axis_type}',
        xaxis_title=x_axis_type,
        yaxis_title='Most Common Pavement Rating',
        yaxis=dict(range=[1, 10], dtick=1)  # Set the y-axis range to [1, 10]
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)


    # Button to save the data to CSV
    csv_format = st.radio("Choose CSV Format:", ["Summary", "Full Ratings"])
    if st.button("Save Data to CSV"):
        if csv_format == "Summary":
            # Logic for saving in current saved format
            csv_data = {
                "Most Common Rating": [],
                "Start Image Index": [],
                "End Image Index": [],
                "Start Image Name": [],
                "End Image Name": []
            }

            #for i in range(0, len(df), section_size):
            #    section_end = min(i + section_size, len(df))
            #    mode_series = df.iloc[i:section_end]['Smoothed Prediction'].mode()
            #    most_common_rating = mode_series.iloc[0] if not mode_series.empty else df.iloc[i]['Smoothed Prediction']
                
            #################################
            for i in range(0, len(df), section_size):
                section_end = min(i + section_size, len(df))
                
                # Extract the section of interest
                section = df.iloc[i:section_end]
                # Calculate the weighted mode
                unique_ratings = section['Smoothed Prediction'].unique()
                weighted_counts = {rating: 0 for rating in unique_ratings}
                
                for rating in unique_ratings:
                    weighted_counts[rating] = sum(section[section['Smoothed Prediction'] == rating]['Probability 1'])
                
                most_common_rating = max(weighted_counts, key=weighted_counts.get) 
                most_common_ratings.extend([most_common_rating] * section_size)
                
                csv_data["Most Common Rating"].append(most_common_rating)
                csv_data["Start Image Index"].append(i)
                csv_data["End Image Index"].append(section_end - 1)  # -1 because it's inclusive
                csv_data["Start Image Name"].append(df.iloc[i]['Image Name'])
                csv_data["End Image Name"].append(df.iloc[section_end - 1]['Image Name'])
            ################################
                
            output_csv_name = "output_data_summary.csv"
        else:
            most_common_ratings = [int(float(rating)) for rating in most_common_ratings]    
            csv_data = {
                "Image Index": list(df.index),
                "Image Name": list(df['Image Name']),
                "Chainage": [determine_chainage_format(val) for val in df['Image Name'].apply(extract_chainage)],
                "Image Rating": most_common_ratings
            }

            # Ensure all columns have the same length
            max_len = max(len(lst) for lst in csv_data.values())  # Find the maximum length among all lists

            for key, lst in csv_data.items():
                if len(lst) < max_len:
                    # Determine the appropriate placeholder based on the data type
                    placeholder = ' ' if isinstance(lst[0], str) else -1
                    # Extend the list with the placeholder value
                    lst.extend([placeholder] * (max_len - len(lst))) 
            output_csv_name = "output_data_detailed.csv"
        
        output_path = os.path.join(folder_path, output_csv_name)
        output_df = pd.DataFrame(csv_data)
        output_df.to_csv(output_path, index=False)
        
        # Generate a URL link for downloading the CSV file
        csv_url = f"http://192.168.1.65:8502/{os.path.basename(folder_path)}/{output_csv_name}"
        st.markdown(f"[Download the CSV file here]({csv_url})")

#=================================================
def csv_analysis():
    """Main function for Streamlit app."""
    st.title("Pavement Rating Analysis")

    # User-defined parameters
    prob_threshold = st.sidebar.slider("Probability Threshold (0-1)", 0.2, 1.0, 0.8)
    window_size = st.sidebar.slider("Median Filter Window Size", 2, 11, 3)
    section_size = st.sidebar.slider("Section Size for Most Common Rating", 5, 100, 20)
    x_axis_options = ["Image Index", "Distance in KMs"]
    x_axis_type = st.sidebar.selectbox("Choose x-axis type:", x_axis_options)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Define the base directory for saving the uploaded CSV file
        temp_dir_path = "/home/pms/streamlit-example/temp/"

        # Check if the csv_analysis_temp_dir is already in the session state
        if 'csv_analysis_temp_dir' not in st.session_state:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get the current date and time
            csv_analysis_temp_dir = os.path.join(temp_dir_path, timestamp)  # Use the timestamp as the directory name
            os.makedirs(csv_analysis_temp_dir)
            
            # Save the uploaded CSV file to the unique directory
            file_path = os.path.join(csv_analysis_temp_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            st.session_state.csv_analysis_temp_dir = csv_analysis_temp_dir  # Store the directory path in the session state
        else:
            file_path = os.path.join(st.session_state.csv_analysis_temp_dir, uploaded_file.name)

        # Read the CSV file from the unique directory
        df = pd.read_csv(file_path)

        if all(col in df.columns for col in ['Image Name', 'Prediction 1', 'Probability 1']):
            analyze_data(df, prob_threshold, window_size, section_size, x_axis_type, st.session_state.csv_analysis_temp_dir)
        else:
            st.error("The uploaded CSV file does not have the required columns.")
