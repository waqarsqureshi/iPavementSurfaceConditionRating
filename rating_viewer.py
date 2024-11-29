# Rating Viewer Module by TU Dublin
# Written by Waqar Shahid Qureshi

import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import time
from datetime import datetime
import zipfile

# Set folder paths as variables
# Set folder paths and server variables from the main Streamlit app
TEMP_DIR  = st.session_state.get('TEMP_DIR ', "/home/pms/streamlit-example/temp/")
SERVER_IP = st.session_state.get('SERVER_IP', "192.168.1.80")
PORT = st.session_state.get('PORT', 8502)
df = 0
# Utility function to check and extract ZIP files
def check_and_extract_zip(uploaded_zip, temp_dir_path):
    """
    Checks and extracts a ZIP file into a temporary directory.
    
    Parameters:
    - uploaded_zip: The uploaded ZIP file.
    - temp_dir_path: Path for temporary extraction.
    
    Returns:
    - Path to the extracted folder.
    """
    if 'image_analysis_temp_dir' not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get the current date and time
        image_analysis_temp_dir = os.path.join(temp_dir_path, timestamp)  # Use the timestamp as the directory name
        os.makedirs(image_analysis_temp_dir, exist_ok=True)

        # Extract the ZIP file
        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(image_analysis_temp_dir)
            st.session_state.image_analysis_temp_dir = image_analysis_temp_dir  # Store the directory path in the session state
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP file. Please upload a valid ZIP file.")

    return st.session_state.image_analysis_temp_dir

# Display selected image and its results
def display_results(df, selected_image_index, folder_path):
    """
    Display results for the selected image.
    
    Parameters:
    - df: DataFrame containing image data.
    - selected_image_name: Index of the selected image.
    - folder_path: Path to the folder containing the images.
    """
    # Find the row in the DataFrame corresponding to the selected image
    selected_image_row = df[df['Image Index'] == selected_image_index]

    if not selected_image_row.empty:
        # Extract the ratings and probability from the selected row
        original_rating = selected_image_row['Actual Rating'].values[0]
        smoothed_rating = selected_image_row['Smoothed Rating'].values[0]
        probability = selected_image_row['Probability'].values[0]
        selected_image_name = selected_image_row['Image Name'].values[0]
        # Load the image
        image_path = os.path.join(folder_path, selected_image_name)
        image = Image.open(image_path)

        # Draw the ratings and probability on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = f"Original Rating: {original_rating}, Smoothed Rating: {smoothed_rating}, Confidence: {probability * 100:.1f}%"
        text_size = draw.textsize(text, font=font)
        circle_radius = 20

        # Draw a circle to indicate rating color
        color = "green" if smoothed_rating > 7 else "yellow" if smoothed_rating > 4 else "red"
        draw.ellipse((10, 10, 10 + 2 * circle_radius, 10 + 2 * circle_radius), fill=color)
        draw.text((10 + 2 * circle_radius + 10, 10), text, fill="black", font=font)

        # Display the image with its ratings and confidence
        st.image(image, caption=f"Original Rating: {original_rating}, Smoothed Rating: {smoothed_rating}, Confidence: {probability * 100:.1f}%", use_column_width=True)
    else:
        st.write("Image not found in the DataFrame.")

# Main function for the rating viewer
def rating_viewer():
    st.subheader("Rating Viewer")

    # Upload CSV file
    csv_file = st.file_uploader("Upload the detailed CSV File containing image rating", type=["csv"])

    # Upload ZIP file containing images
    zip_file = st.file_uploader("Upload ZIP File (Containing Images)", type=["zip"])

    if csv_file is not None and zip_file is not None:
        # Extract images from the zip file to a temporary folder
        temp_folder = check_and_extract_zip( zip_file, TEMP_DIR)
        # Read CSV file
        csv_path = os.path.join(temp_folder, "uploaded_data.csv")
        with open(csv_path, 'wb') as f:
            f.write(csv_file.getvalue())
        df = pd.read_csv(csv_path)

        # Display the CSV data
        st.subheader("CSV Data:")
        st.dataframe(df, use_container_width=True)
        # List all images from the CSV file
        images_from_csv = df['Image Index'].tolist()

        # Select image file from the extracted folder based on CSV list
        selected_image_index = st.selectbox("Select Image Index", images_from_csv)

        # Display the selected image and its ratings
        display_results(df, selected_image_index, temp_folder)

        # Get the current rating from the CSV file and allow the user to modify it
        current_rating = df.loc[df['Image Index'] == selected_image_index, 'Smoothed Rating'].values[0]
        selected_rating = st.slider("Rate the Selected Image:", 1, 10, value=int(current_rating))
        st.write(f"You rated the image '{selected_image_index}' with {selected_rating}.")

        # Update the DataFrame with the new rating
        df.loc[df['Image Index'] == selected_image_index, 'User Rating'] = selected_rating

        # Save button to save the updated ratings
        if st.button("Save Updated Ratings"):
            output_csv_path = os.path.join(temp_folder, "modified_output_data_detailed.csv")
            df.to_csv(output_csv_path, index=False)
            st.success(f"Updated ratings saved to {output_csv_path}")
            if os.path.exists(output_csv_path):
                output_csv_url = f"http://{SERVER_IP}:{PORT}/temp/{os.path.basename(temp_folder)}/{os.path.basename(output_csv_path)}"
                st.markdown(f"[Download the updated CSV file]({output_csv_url})")
            else:
                st.warning("Updated CSV file does not exist.")

        # Start/Stop video player buttons
        start_button = st.button("Start Video Player")
        stop_button = st.button("Stop Video Player")

        # Initialize last_stopped_frame in session state if not present
        if 'last_stopped_frame' not in st.session_state:
            st.session_state.last_stopped_frame = None

        # Get the last stopped frame from session state
        last_stopped_frame = st.session_state.last_stopped_frame

        if start_button:
            # Embed images as video frames at 5 frames per second
            video_frame = st.empty()

            # Determine the starting index based on whether the "Start" button was pressed for the first time or not
            start_index = 0 if last_stopped_frame is None else images_from_csv.index(last_stopped_frame) + 1

            for index in range(start_index, len(images_from_csv)):
                if stop_button:
                    st.write("Player stopped at frame:", images_from_csv[index])
                    image_path = os.path.join(temp_folder, images_from_csv[index])
                    image = Image.open(image_path)
                    video_frame.image(image, caption=f"Image: {images_from_csv[index]}", use_column_width=True)
                    # Update the last stopped frame in session state
                    st.session_state.last_stopped_frame = images_from_csv[index]
                    break
                else:
                    image_path = os.path.join(temp_folder, images_from_csv[index])
                    image = Image.open(image_path)
                    video_frame.image(image, caption=f"Image: {images_from_csv[index]}", use_column_width=True)

                    # Sleep to achieve 5 frames per second
                    time.sleep(0.5)
