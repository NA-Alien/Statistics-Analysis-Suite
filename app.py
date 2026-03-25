import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Page Configuration
st.set_page_config(page_title="Statistical Analysis Suite", layout="wide")

# Custom CSS for a professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Statistical Analysis Suite")
st.markdown("Advanced Descriptive Statistics and Inference Engine for AP Statistics.")

# 1. Data Input System
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if st.sidebar.button("Generate Synthetic Data"):
    # Generates a normally distributed set of 100 points
    data = np.random.normal(75, 12, 100).round(2)
    sample_df = pd.DataFrame({'Observation_Data': data})
    sample_df.to_csv("synthetic_data.csv", index=False)
    st.sidebar.success("Synthetic data generated: 'synthetic_data.csv'")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("Error: No numerical data detected in the uploaded file.")
    else:
        target_var = st.sidebar.selectbox("Analysis Variable", numeric_cols)
        clean_series = df[target_var].dropna()

        # 2. Summary Metrics Dashboard
        st.header(f"Descriptive Statistics: {target_var}")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Mean (x-bar)", f"{clean_series.mean():.2f}")
        m2.metric("Median", f"{clean_series.median():.2f}")
        m3.metric("Std Deviation (s)", f"{clean_series.std():.2f}")
        m4.metric("Sample Size (n)", len(clean_series))

        # 3. Shape and Spread Analysis
        with st.expander("Detailed Distribution Metrics"):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Variance:** {clean_series.var():.2f}")
            c2.write(f"**Skewness:** {clean_series.skew():.2f}")
            c3.write(f"**Kurtosis:** {clean_series.kurt():.2f}")
            
            # AP Stats Outlier Logic (1.5xIQR)
            q1 = clean_series.quantile(0.25)
            q3 = clean_series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = clean_series[(clean_series < lower) | (clean_series > upper)]
            
            st.write(f"**Interquartile Range (IQR):** {iqr:.2f}")
            if not outliers.empty:
                st.warning(f"Outliers Detected: {len(outliers)} values outside the interval [{lower:.2f}, {upper:.2f}]")
                st.write(outliers.values)
            else:
                st.success("Condition Met: No outliers detected via 1.5xIQR rule.")

        # 4. Visualization Suite (Histogram, Q-Q, Box Plot)
        st.header("Visual Analysis")
        v1, v2, v3 = st.columns(3)

        with v1:
            st.subheader("Histogram")
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(clean_series, bins=12, color='#2c3e50', edgecolor='white')
            st.pyplot(fig_hist)

        with v2:
            st.subheader("Q-Q Plot")
            fig_qq, ax_qq = plt.subplots()
            stats.probplot(clean_series, dist="norm", plot=ax_qq)
            st.pyplot(fig_qq)

        with v3:
            st.subheader("Box Plot")
            fig_box, ax_box = plt.subplots()
            ax_box.boxplot(clean_series, vert=False, patch_artist=True, 
                           boxprops=dict(facecolor="#ecf0f1"))
            st.pyplot(fig_box)

        # 5. Statistical Inference
        st.header("Inference Procedures")
        t1, t2 = st.columns(2)

        with t1:
            st.subheader("One-Sample T-Test")
            null_mean = st.number_input("Null Hypothesis (mu-0)", value=0.0)
            if st.button("Calculate T-Procedure"):
                t_stat, p_val = stats.ttest_1samp(clean_series, null_mean)
                st.write(f"T-Statistic: {t_stat:.4f}")
                st.write(f"P-Value: {p_val:.4f}")