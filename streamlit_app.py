# by TU Dublin written by Waqar Shahid Qureshi
#Run the following server in the folder temp folder
#python3 -m http.server 8502 --bind 192.168.1.65
# and run the following in the directory where this code is residing
#streamlit run streamlit_app.py --server.address 192.168.1.65 --server.port 8501

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime
import glob, os
import pydeck as pdk
import numpy as np

from image_analysis import image_analysis
from csv_analysis import csv_analysis
from map_visualization import map_visualization
from help import display_help
# Define paths for model checkpoints and configurations
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'
config_deeplabv3plus = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py"
check_point_deeplabv3plus = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth"

def main():
    st.title("iPSCI Application")
    # Display logos in two columns
    left_column, right_column = st.columns(2)
    left_column.image("/home/pms/Pictures/TU Dublin Logo_resized.jpg", width=200)
    right_column.image("/home/pms/Pictures/pms.png", width=200)
        # Display the title of the application

    # Checkboxes for user confirmation on paths
    st.subheader("Configuration Confirmation")
    checkpoint_confirmed = st.checkbox("I have set the models checkpoint paths for both the models or I'm okay with using the default paths.")
    config_confirmed = st.checkbox("I have set the pavement extraction model configuration path or I'm okay with using the default paths.")
    config_confirmed = st.checkbox("I will read the Help first if I am the first time user of the application.")
    config_confirmed = st.checkbox("Make sure the file server is running on the server machine: python3 -m http.server 8502 --bind 192.168.1.65")

    # Only show the dropdown if both checkboxes are ticked
    if checkpoint_confirmed and config_confirmed:
        # Ask the user for the type of analysis or to view the help
        analysis_type = st.selectbox("Select an option:", [" ", "Help", "Image Analysis", "CSV Data Analysis", "Map Visualization"])

        if analysis_type == "Image Analysis":
            with zipfile.ZipFile("images.zip") as zip_file:
                for file in zip_file.namelist():
                    image_analysis(file)
        elif analysis_type == "CSV Data Analysis":
            csv_analysis()
        elif analysis_type == "Help":
            display_help()
        elif analysis_type == "Map Visualization":
            map_visualization()
    else:
        st.warning("Please confirm the paths for checkpoints and configurations to proceed.")

if __name__ == "__main__":
    main()
