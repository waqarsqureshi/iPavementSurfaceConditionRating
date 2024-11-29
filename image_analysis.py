# Image Analysis Module by TU Dublin
# Written by Waqar Shahid Qureshi

# Instructions:
# 1. Run the following server in the 'temp' folder:
#    python3 -m http.server 8502 --bind 192.168.1.65
# 2. Run this Streamlit app in the directory where this code is residing:
#    streamlit run streamlit_app.py --server.address 192.168.1.65 --server.port 8501

import streamlit as st
import zipfile
from datetime import datetime
import glob, os
from pavement_utils import (classify, gradCam, PavementExtraction, initialize_model)

# Define paths for model checkpoints and configurations
# Set folder paths and server variables from the main Streamlit app
TEMP_DIR = st.session_state.get('TEMP_DIR', "/home/pms/streamlit-example/temp/")
SERVER_IP = st.session_state.get('SERVER_IP', "192.168.1.80")
PORT = st.session_state.get('PORT', 8502)
CHECKPOINT_PATH_SWIM = st.session_state.get('CHECKPOINT_PATH_SWIM', '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar')
CONFIG_DEEPLABV3PLUS = st.session_state.get('CONFIG_DEEPLABV3PLUS', "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py")
CHECKPOINT_DEEPLABV3PLUS = st.session_state.get('CHECKPOINT_DEEPLABV3PLUS', "/home/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth")

def image_analysis():
    # Define the options for pavement analysis
    options = ["Pavement Surface Extraction", "Pavement Rating", "Pavement Distress Analysis"]
    selected_option = st.sidebar.selectbox("Choose a process:", options)
    
    # Allow users to upload a ZIP file
    uploaded_zip = st.file_uploader("Upload a ZIP file of images:", type=['zip'])

    # If a ZIP file is uploaded
    if uploaded_zip:
        temp_dir_path = TEMP_DIR 
        os.makedirs(temp_dir_path, exist_ok=True)

        # Check if the image_analysis_temp_dir is already in the session state
        if 'image_analysis_temp_dir' not in st.session_state:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get the current date and time
            image_analysis_temp_dir = os.path.join(temp_dir_path, timestamp)  # Use the timestamp as the directory name
            os.makedirs(image_analysis_temp_dir)
            
            # Extract the ZIP file
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(image_analysis_temp_dir)
                st.session_state.image_analysis_temp_dir = image_analysis_temp_dir  # Store the directory path in the session state
            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP file. Please upload a valid ZIP file.")

        # Get the list of images from the extracted ZIP file
        images = [f for f in glob.glob(os.path.join(st.session_state.image_analysis_temp_dir, '**', '*.[jJ][pP][eE][gG]'), recursive=True)]
        images += [f for f in glob.glob(os.path.join(st.session_state.image_analysis_temp_dir, '**', '*.[jJ][pP][gG]'), recursive=True)]

        # If images are found
        if images:
            # For each option, initialize the model and process the images accordingly
            if selected_option == "Pavement Rating" or selected_option == "Pavement Distress Analysis":
                model_name = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
                st.session_state.image_analysis_model, st.session_state.image_analysis_device = initialize_model(model_name, CHECKPOINT_PATH_SWIM)
                
                if selected_option == "Pavement Rating":
                    classify(st.session_state.image_analysis_temp_dir, st.session_state.image_analysis_model, st.session_state.image_analysis_device)
                else:
                    gradCam(st.session_state.image_analysis_temp_dir, st.session_state.image_analysis_model, st.session_state.image_analysis_device)

            elif selected_option == "Pavement Surface Extraction":
                model_name = 'deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512'
                st.session_state.pavement_surface_extraction_model, st.session_state.pavement_surface_extraction_device = initialize_model(model_name, CHECKPOINT_DEEPLABV3PLUS, CONFIG_DEEPLABV3PLUS)
                PavementExtraction(st.session_state.image_analysis_temp_dir, st.session_state.pavement_surface_extraction_model, st.session_state.pavement_surface_extraction_device)
        else:
            st.error("No images found in the ZIP file. Please upload a valid ZIP file containing images.")
    else:
        st.warning("Please upload a ZIP file containing images.")

