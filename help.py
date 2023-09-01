
import streamlit as st
def display_help():
    st.subheader("User Help Guide for Pavement Surface Condition Index Application")
    st.write("""
    ### Welcome to the Pavement Surface Condition Index Application!

    
1. The Application:
This application, developed by TU Dublin and authored by Waqar Shahid Qureshi, provides a comprehensive solution for analyzing pavement conditions. Here's a step-by-step guide to help you navigate and use the application effectively:

2. Main Interface:

Upon launching, you'll be greeted with the title "Pavement Surface Condition Index" and logos of TU Dublin and PMS.

    Analysis Selection: Use the dropdown menu labeled "Select the type of Analysis" to choose between "Image Analysis" and "CSV Data Analysis".

3. Image Analysis:

If you select "Image Analysis":

    Upload ZIP File: Use the provided interface to upload a ZIP file containing the images you wish to analyze.

    Analysis Type: Once the ZIP file is uploaded, you'll be prompted to select the specific type of image analysis:
        Pavement Rating: Classifies the images based on pavement rating.
        Pavement Distress Analysis: Analyzes the distress in the pavement.
        Pavement Surface Extraction: Extracts the pavement surface from the images.

4. CSV Data Analysis:

If you select "CSV Data Analysis":

    Parameter Definition: On the sidebar, you can define various parameters that will influence the analysis.

    Upload CSV File: Use the provided interface to upload a CSV file containing the data you wish to analyze.

    Interactive Plot: Once the CSV file is uploaded and processed, an interactive plot will be displayed based on the data. You can hover over the plot for detailed insights.

    Save Data: If you wish to save the processed data, click on the "Save Data to CSV" button. A link will be generated, allowing you to download the CSV file.

5. Troubleshooting:

    Invalid ZIP File: If no images are found in the uploaded ZIP file, an error message will be displayed. Ensure the ZIP file contains valid images and try again.

    Invalid CSV File: If the uploaded CSV file doesn't have the required columns, an error message will be displayed. Ensure the CSV file is in the correct format and try again.

6. Feedback & Support:

If you encounter any issues or have suggestions for improvements, please reach out to the development team. Your feedback is invaluable to us!

    Thank you for using the Pavement Surface Condition Index Application. We hope it serves your needs effectively and provides valuable insights into pavement conditions!
    """)