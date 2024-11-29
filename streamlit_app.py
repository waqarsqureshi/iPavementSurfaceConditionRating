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
from rating_viewer import rating_viewer

# Set session variables for paths and server details
if 'TEMP_DIR' not in st.session_state:
    st.session_state.TEMP_DIR = "/home/pms/streamlit-example/temp/"

if 'SERVER_IP' not in st.session_state:
    st.session_state.SERVER_IP = "192.168.1.80"

if 'PORT' not in st.session_state:
    st.session_state.PORT = 8502

# Set session variables for model checkpoint paths and configurations
if 'CHECKPOINT_PATH_SWIM' not in st.session_state:
    st.session_state.CHECKPOINT_PATH_SWIM = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'

if 'CONFIG_DEEPLABV3PLUS' not in st.session_state:
    st.session_state.CONFIG_DEEPLABV3PLUS = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py"

if 'CHECKPOINT_DEEPLABV3PLUS' not in st.session_state:
    st.session_state.CHECKPOINT_DEEPLABV3PLUS = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth"


def main():
    st.set_page_config(page_title="Password Protected iPSCI Application", layout="wide")

    # Password protection
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter Password:", type="password")
        if password == "pms":  # Replace with your secure password
            st.session_state.authenticated = True
        else:
            st.warning("Please enter the correct password.")
            return
    
    # Sidebar for configuration and instructions
    st.sidebar.title("Configuration & Instructions")
    st.sidebar.image("Pictures/TU Dublin Logo_resized.jpg", width=150)
    st.sidebar.image("Pictures/pms.png", width=150)

    st.sidebar.subheader("Configuration Confirmation")
    checkpoint_confirmed = st.sidebar.checkbox("I have set the model checkpoints or I am okay with using the default paths.")
    config_confirmed = st.sidebar.checkbox("I have set the pavement extraction model configuration path or I am okay with using the default paths.")
    help_read = st.sidebar.checkbox("I will read the Help section if I am a first-time user of the application.")
    server_running = st.sidebar.checkbox("I confirm that the file server is running on the server machine.")

    # Ensure all confirmations are checked
    if checkpoint_confirmed and config_confirmed and help_read and server_running:
        st.title("Intelligent Pavement Surface Condition Index - Application")
        
        # Use tabs for better navigation between sections
        tabs = st.tabs(["Image Analysis", "Rating Data Analysis", "Image Rating Viewer", "Map Visualization", "Help"])

        try:
            with tabs[0]:
                st.header("Image Analysis")
                image_analysis()  # Call the function that performs image analysis

            with tabs[1]:
                st.header("Rating Data Analysis")
                csv_analysis()  # Call the function that performs CSV data analysis

            with tabs[2]:
                st.header("Image Rating Viewer")
                rating_viewer()

            with tabs[3]:
                st.header("Map Visualization")
                map_visualization()  # Call the function for map visualization

            with tabs[4]:
                st.header("Help")
                display_help()  # Call the help display function

        except NameError as e:
            st.error(f"An error occurred while accessing a function: {e}")

        # Improved feedback for process status
        st.sidebar.subheader("Processing Status")
        if 'processing' in st.session_state and st.session_state.processing:
            st.sidebar.info("Processing is currently running...")
        else:
            st.sidebar.success("Ready for analysis.")

    else:
        st.warning("Please confirm all the requirements in the sidebar to proceed.")


if __name__ == "__main__":
    main()
