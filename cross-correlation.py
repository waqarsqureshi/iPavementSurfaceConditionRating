import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial.distance import euclidean, cityblock
from fastdtw import fastdtw
from sklearn.metrics import mutual_info_score
from scipy.signal import correlate
import plotly.graph_objects as go
from scipy.stats import mode

from scipy.ndimage import median_filter
from scipy.stats import mode
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.signal import correlate

def apply_filters(series, window_size=5, mode_section=5):
    """
    Apply mode and moving median filters to a series.

    Parameters:
    - series: The series to filter.
    - window_size: The size of the window to use for the moving median filter.
    - mode_section: The size of the section to use for the mode filter.

    Returns:
    - The filtered series.
    """
    # Apply moving median filter
    median_filtered = median_filter(series, size=window_size)
    
    # Apply mode filter
    sections = [median_filtered[i:i+mode_section] for i in range(0, len(median_filtered), mode_section)]
    mode_values = [mode(section)[0][0] for section in sections]
    mode_filtered = []
    for value in mode_values:
        mode_filtered.extend([value] * mode_section)
    mode_filtered = mode_filtered[:len(median_filtered)]
    
    return mode_filtered



def compute_metrics(series1, series2):
    """
    Compute and display various metrics between two series.

    Parameters:
    - series1: First series.
    - series2: Second series.

    Displays:
    - Normalized Mutual Information.
    - Peak value of the Cross-Correlation Function.
    - Cross-Correlation between the two series.
    """
    
    # Ensure the series are 1-D
    series1 = np.array(series1).ravel()
    series2 = np.array(series2).ravel()

    # Mutual Information
    nmi = mutual_info_score(series1, series2)
    st.write(f"Mutual Information: {nmi}")

    # Cross-Correlation
    cross_correlation = np.corrcoef(series1, series2)[0, 1]
    st.write(f"Cross-Correlation: {cross_correlation}")

#==================================================
    
def plot_difference(data):
    # Calculate the difference between the two ratings
    #data['Difference'] = data['Image Rating_1'] - data['Image Rating_2']

    # Plotting the difference using Plotly
    fig = go.Figure()

    # Plotting the ratings of series 1
    fig.add_trace(go.Scatter(x=data['Chainage'], y=data['Image Rating_1'],
                    mode='lines',
                    name='Manual Rating'))

    # Plotting the ratings of series 2
    fig.add_trace(go.Scatter(x=data['Chainage'], y=data['Image Rating_2'],
                    mode='lines',
                    name='Automated Rating'))

    # Plotting the difference
    #fig.add_trace(go.Scatter(x=data['Chainage'], y=data['Difference'],
    #                mode='lines+markers',
    #                name='Difference'))

    fig.update_layout(title='Difference between Manual Rating(1) and Automated Rating(2)',
                   xaxis_title='Chainage',
                   yaxis_title='Rating / Difference')

    st.plotly_chart(fig)

def main():
    st.title("Series Relationship Analysis")

    # Upload the first CSV
    csv1 = st.file_uploader("Upload the Manual Rating CSV file", type="csv")
    if csv1:
        df1 = pd.read_csv(csv1)
        df1 = df1[df1['Image Rating'] != -1]  # Filter out rows with -1

        # Upload the second CSV
        csv2 = st.file_uploader("Upload the Automated Rating CSV file", type="csv")
        if csv2:
            df2 = pd.read_csv(csv2)
            df2 = df2[df2['Image Rating'] != -1]  # Filter out rows with -1

            # Align the series based on chainage
            merged_df = pd.merge(df1, df2, on="Chainage", how="inner", suffixes=('_1', '_2'))

            # Display the merged dataframe for debugging
            st.write(merged_df)
            
            plot_difference(merged_df)
            # Compute metrics on the Image Rating columns
            compute_metrics(merged_df['Image Rating_1'].values, merged_df['Image Rating_2'].values)

if __name__ == "__main__":
    main()




