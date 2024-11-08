import streamlit as st
import pandas as pd
from PIL import Image
import zipfile
import os
import tempfile
from datetime import datetime

#============================================================
def check_and_extract_zip(uploaded_zip, temp_dir_path):
    # Check if the image_analysis_temp_dir is already in the session state
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

#============================================================

import streamlit as st
import pandas as pd
from PIL import Image
import zipfile
import os
import tempfile
from datetime import datetime
import time

#============================================================
def check_and_extract_zip(uploaded_zip, temp_dir_path):
    # Check if the image_analysis_temp_dir is already in the session state
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

#============================================================
#============================================================

def main():
    st.title("Image Viewer with Rating")

    # Upload CSV file
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])

    # Upload ZIP file containing images
    zip_file = st.file_uploader("Upload ZIP File (Containing Images)", type=["zip"])

    if csv_file is not None and zip_file is not None:
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Display the CSV data
        st.write("CSV Data:")
        st.write(df)

        # Extract images from the zip file to a temporary folder
        temp_dir_path = "/home/pms/streamlit-example/temp/"
        temp_folder = check_and_extract_zip(zip_file, temp_dir_path)

        # List all images from the CSV file
        images_from_csv = df['Image Name'].tolist()

        # Select image file from the extracted folder based on CSV list
        selected_image = st.selectbox("Select Image File", images_from_csv)

        # Display the selected image
        display_results(df, selected_image, temp_folder)

        # Get the rating for the selected image
        # Get the rating for the selected image using buttons
        selected_rating = st.button("1") + 2 * st.button("2") + 3 * st.button("3") + 4 * st.button("4") + 5 * st.button("5") + 6 * st.button("6") + 7 * st.button("7") + 8 * st.button("8") + 9 * st.button("9") + 10 * st.button("10")

        st.write(f"You rated the image '{selected_image}' with {selected_rating}.")

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

#============================================================



#============================================================



#============================================================
def extract_images(zip_file,temp_dir_path):
    # Create a temporary folder
    temp_folder = os.makedirs(temp_dir_path, exist_ok=True)

    # Extract images from the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_folder)

    return temp_folder

#============================================================
def display_results(df, selected_image_name, folder_path):
    # Find the row in the DataFrame corresponding to the selected image
    selected_image_row = df[df['Image Name'] == selected_image_name]

    if not selected_image_row.empty:
        # Extract the rating from the selected row
        image_rating = selected_image_row['Image Rating'].values[0]

        # Load the image
        image_path = os.path.join(folder_path, selected_image_name)
        image = Image.open(image_path)

        # Display the image with its rating
        st.image(image, caption=f"Image Rating: {image_rating}", use_column_width=True)
    else:
        st.write("Image not found in the DataFrame.")

#============================================================


if __name__ == "__main__":
    main()
