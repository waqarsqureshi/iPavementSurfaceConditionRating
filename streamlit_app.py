# iPSCI Application by TU Dublin
# Written by Waqar Shahid Qureshi

# Instructions:
# 1. Run the following server in the 'temp' folder:
#    python3 -m http.server 8502 --bind 192.168.1.65
# 2. Run this Streamlit app in the directory where this code is residing:
#    streamlit run streamlit_app.py --server.address 192.168.1.65 --server.port 8501

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
CHECKPOINT_PATH_SWIM = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'
CONFIG_DEEPLABV3PLUS = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py"
CHECKPOINT_DEEPLABV3PLUS = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth"

def main():
    st.set_page_config(page_title="iPSCI Application", layout="wide")
    st.title("iPSCI Application")
    
    # Display logos in two columns
    left_column, right_column = st.columns(2)
    with left_column:
        st.image("/home/pms/Pictures/TU Dublin Logo_resized.jpg", width=200)
    with right_column:
        st.image("/home/pms/Pictures/pms.png", width=200)

    # Display configuration confirmation
    st.subheader("Configuration Confirmation")
    checkpoint_confirmed = st.checkbox("I have set the model checkpoints or I am okay with using the default paths.")
    config_confirmed = st.checkbox("I have set the pavement extraction model configuration path or I am okay with using the default paths.")
    help_read = st.checkbox("I will read the Help section if I am a first-time user of the application.")
    server_running = st.checkbox("I confirm that the file server is running on the server machine: python3 -m http.server 8502 --bind 192.168.1.65")

    # Ensure all confirmations are checked
    if checkpoint_confirmed and config_confirmed and help_read and server_running:
        # Ask the user for the type of analysis or to view the help
        analysis_type = st.selectbox("Select an option:", [" ", "Help", "Image Analysis", "CSV Data Analysis", "Map Visualization"])

        if analysis_type == "Image Analysis":
            image_analysis()
        elif analysis_type == "CSV Data Analysis":
            csv_analysis()
        elif analysis_type == "Help":
            display_help()
        elif analysis_type == "Map Visualization":
            map_visualization()
    else:
        st.warning("Please confirm all the requirements to proceed.")

if __name__ == "__main__":
    main()
