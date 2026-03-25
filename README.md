# Statistical Analysis Suite

A professional data science dashboard built with Python and Streamlit. This application automates core procedures from the AP Statistics curriculum, providing a centralized interface for data cleaning, distribution analysis, and formal inference.

## Project Overview

The Statistical Analysis Suite is designed to bridge the gap between theoretical statistics and functional software development. By integrating industry-standard libraries, the suite allows users to upload raw data and instantly generate the visualizations and test statistics required for rigorous analysis.

## Core Features

### 1. Descriptive Statistics and Data Integrity
* **Central Tendency and Spread:** Calculates Mean, Median, Standard Deviation, and the 5-number summary.
* **Automated Outlier Detection:** Implements the 1.5xIQR rule to identify and flag potential outliers.
* **Distribution Metrics:** Provides Skewness and Kurtosis values to assess the shape of the dataset.

### 2. Visual Analysis Suite
* **Histograms:** Visualizes frequency distributions to identify peaks, clusters, and gaps.
* **Normal Probability Plots (Q-Q Plots):** A critical tool for verifying the Normal condition before proceeding with T-procedures.
* **Box Plots:** Side-by-side visualization of quartiles and interquartile range (IQR).

### 3. Inferential Statistics
* **1-Sample T-Tests:** Automates hypothesis testing by calculating T-statistics and P-values against a user-defined null hypothesis (mu-0).
* **Confidence Intervals:** Generates intervals for 90%, 95%, and 99% confidence levels, including the calculation of critical values (t*) based on degrees of freedom.

## Technical Stack
* **Language:** Python 3.12
* **Interface:** Streamlit
* **Data Processing:** Pandas and NumPy
* **Mathematics and Visualization:** SciPy and Matplotlib

## Installation and Usage

To run this project locally, ensure you have Python installed, then follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
