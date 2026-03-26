import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Page Configuration
st.set_page_config(page_title="Statistical Analysis Suite", layout="wide")

# Custom CSS for UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 28px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Statistical Analysis Suite")
st.markdown("Professional Workbench for Descriptive, Visual, and Inferential Statistics.")

# 1. Data Input System
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

if st.sidebar.button("Generate Synthetic Data"):
    data_x = np.random.normal(75, 12, 100).round(2)
    data_y = (data_x * 0.8 + np.random.normal(0, 5, 100)).round(2)
    sample_df = pd.DataFrame({'Explanatory_X': data_x, 'Response_Y': data_y})
    sample_df.to_csv("synthetic_stats_data.csv", index=False)
    st.sidebar.success("Synthetic data generated: 'synthetic_stats_data.csv'")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.error("Error: No numerical data detected.")
    else:
        target_var = st.sidebar.selectbox("Primary Variable (Univariate)", numeric_cols)
        clean_series = df[target_var].dropna()

        # 2. Univariate Dashboard
        st.header(f"Univariate Analysis: {target_var}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean (x-bar)", f"{clean_series.mean():.2f}")
        m2.metric("Median", f"{clean_series.median():.2f}")
        m3.metric("Std Deviation (s)", f"{clean_series.std():.2f}")
        m4.metric("Sample Size (n)", len(clean_series))

        # 3. Z-Score and Formal Normality Testing
        with st.expander("Advanced Tools: Z-Scores and Normality Tests"):
            col_z, col_norm = st.columns(2)
            with col_z:
                st.subheader("Z-Score Calculator")
                obs_value = st.number_input("Enter an Observation Value", value=float(clean_series.mean()))
                z = (obs_value - clean_series.mean()) / clean_series.std()
                st.write(f"The Z-score for {obs_value} is: **{z:.4f}**")
            
            with col_norm:
                st.subheader("Shapiro-Wilk Normality Test")
                shapiro_stat, shapiro_p = stats.shapiro(clean_series)
                st.write(f"P-Value: {shapiro_p:.4f}")
                if shapiro_p > 0.05:
                    st.success("Data appears to be Normally Distributed (p > 0.05)")
                else:
                    st.warning("Data does not appear to be Normal (p <= 0.05)")

        # 4. Visual Analysis Suite
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
            ax_box.boxplot(clean_series, vert=False, patch_artist=True, boxprops=dict(facecolor="#ecf0f1"))
            st.pyplot(fig_box)

        # 5. Bivariate Analysis & Residuals
        if len(numeric_cols) >= 2:
            st.header("Bivariate Analysis & Residual Plot")
            y_var = st.selectbox("Select Y Variable (Response)", numeric_cols, index=0)
            x_var = st.selectbox("Select X Variable (Explanatory)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            if x_var != y_var:
                b_df = df[[x_var, y_var]].dropna()
                slope, intercept, r_value, p_value, std_err = stats.linregress(b_df[x_var], b_df[y_var])
                
                # Calculate Residuals: y - y_hat
                y_hat = intercept + slope * b_df[x_var]
                residuals = b_df[y_var] - y_hat
                
                reg1, reg2 = st.columns(2)
                with reg1:
                    fig_reg, ax_reg = plt.subplots()
                    ax_reg.scatter(b_df[x_var], b_df[y_var], alpha=0.6, color='#3498db')
                    ax_reg.plot(b_df[x_var], y_hat, color='#e74c3c', label='LSRL')
                    ax_reg.set_title("Scatter Plot with LSRL")
                    st.pyplot(fig_reg)
                    st.write(f"**LSRL:** ŷ = {intercept:.4f} + {slope:.4f}x")
                    st.write(f"**r:** {r_value:.4f} | **r²:** {r_value**2:.4f}")

                with reg2:
                    fig_res, ax_res = plt.subplots()
                    ax_res.scatter(b_df[x_var], residuals, color='#8e44ad', alpha=0.6)
                    ax_res.axhline(0, color='black', linestyle='--')
                    ax_res.set_title("Residual Plot")
                    st.pyplot(fig_res)
                    st.info("Check for a random scatter of points with no clear pattern to verify the linear model.")

        # 6. Inference & Export
        st.header("Inference Procedures")
        t1, t2 = st.columns(2)
        with t1:
            st.subheader("One-Sample T-Test")
            null_mean = st.number_input("Null Hypothesis (mu-0)", value=0.0)
            if st.button("Calculate T-Procedure"):
                t_stat, p_val = stats.ttest_1samp(clean_series, null_mean)
                st.write(f"T-Statistic: {t_stat:.4f} | P-Value: {p_val:.4f}")
                if p_val < 0.05:
                    st.error("Reject H0.")
                else:
                    st.warning("Fail to Reject H0.")
        with t2:
            st.subheader("Confidence Interval")
            conf = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
            t_crit = stats.t.ppf((1 + conf) / 2, len(clean_series) - 1)
            margin_error = t_crit * (clean_series.std() / np.sqrt(len(clean_series)))
            st.info(f"Interval: ({clean_series.mean() - margin_error:.2f}, {clean_series.mean() + margin_error:.2f})")

        # 7. Data Export
        st.sidebar.markdown("---")
        summary_df = clean_series.describe().to_frame()
        csv_export = summary_df.to_csv().encode('utf-8')
        st.sidebar.download_button("Download Summary Stats", data=csv_export, file_name="summary_stats.csv", mime="text/csv")

else:
    st.info("System Ready. Please upload a CSV file or generate synthetic data to begin analysis.")
