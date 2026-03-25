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
    # Generates a normally distributed set of 100 points with a related second variable
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
        # Sidebar Selection
        target_var = st.sidebar.selectbox("Primary Variable (Univariate)", numeric_cols)
        clean_series = df[target_var].dropna()

        # 2. Univariate Dashboard
        st.header(f"Univariate Analysis: {target_var}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean (x-bar)", f"{clean_series.mean():.2f}")
        m2.metric("Median", f"{clean_series.median():.2f}")
        m3.metric("Std Deviation (s)", f"{clean_series.std():.2f}")
        m4.metric("Sample Size (n)", len(clean_series))

        # 3. Z-Score and Transformation Tools
        with st.expander("Advanced Tools: Z-Scores and Transformations"):
            col_z, col_trans = st.columns(2)
            with col_z:
                st.subheader("Z-Score Calculator")
                obs_value = st.number_input("Enter an Observation Value", value=float(clean_series.mean()))
                # Calculation: z = (x - mean) / std
                z = (obs_value - clean_series.mean()) / clean_series.std()
                st.write(f"The Z-score for {obs_value} is: **{z:.4f}**")
            
            with col_trans:
                st.subheader("Data Transformation")
                trans_type = st.radio("Apply Transformation (Check for Linearity/Normality):", ["None", "Log10", "Square Root"])
                if trans_type == "Log10":
                    transformed = np.log10(clean_series[clean_series > 0])
                elif trans_type == "Square Root":
                    transformed = np.sqrt(clean_series[clean_series >= 0])
                else:
                    transformed = clean_series
                st.write(f"Original Skewness: {clean_series.skew():.4f}")
                st.write(f"Transformed Skewness: {transformed.skew():.4f}")

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

        # 5. Bivariate Analysis (Least Squares Regression Line)
        if len(numeric_cols) >= 2:
            st.header("Bivariate Analysis (Regression)")
            y_var = st.selectbox("Select Y Variable (Response)", numeric_cols, index=0)
            x_var = st.selectbox("Select X Variable (Explanatory)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            if x_var != y_var:
                b_df = df[[x_var, y_var]].dropna()
                slope, intercept, r_value, p_value, std_err = stats.linregress(b_df[x_var], b_df[y_var])
                
                res1, res2 = st.columns(2)
                with res1:
                    st.write(f"**LSRL Equation:** ŷ = {intercept:.4f} + {slope:.4f}x")
                    st.write(f"**Correlation Coefficient (r):** {r_value:.4f}")
                with res2:
                    st.write(f"**Coefficient of Determination (r²):** {r_value**2:.4f}")
                    st.write(f"**Standard Error of Estimate:** {std_err:.4f}")
                
                fig_reg, ax_reg = plt.subplots()
                ax_reg.scatter(b_df[x_var], b_df[y_var], alpha=0.6, color='#3498db')
                ax_reg.plot(b_df[x_var], intercept + slope*b_df[x_var], color='#e74c3c', label='LSRL')
                ax_reg.set_xlabel(x_var)
                ax_reg.set_ylabel(y_var)
                ax_reg.legend()
                st.pyplot(fig_reg)

        # 6. Inference Procedures
        st.header("Inference Procedures")
        t1, t2 = st.columns(2)
        with t1:
            st.subheader("One-Sample T-Test")
            null_mean = st.number_input("Null Hypothesis (mu-0)", value=0.0)
            if st.button("Calculate T-Procedure"):
                t_stat, p_val = stats.ttest_1samp(clean_series, null_mean)
                st.write(f"T-Statistic: {t_stat:.4f}")
                st.write(f"P-Value: {p_val:.4f}")
                
                alpha = 0.05
                if p_val < alpha:
                    st.error(f"Reject H0. P-value < {alpha}.")
                else:
                    st.warning(f"Fail to Reject H0. P-value >= {alpha}.")
        with t2:
            st.subheader("Confidence Interval")
            conf = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
            # Calculation: x-bar +/- t*(s/sqrt(n))
            df_deg = len(clean_series) - 1
            t_crit = stats.t.ppf((1 + conf) / 2, df_deg)
            margin_error = t_crit * (clean_series.std() / np.sqrt(len(clean_series)))
            
            st.write(f"**Critical Value (t*):** {t_crit:.4f}")
            st.info(f"Interval: ({clean_series.mean() - margin_error:.2f}, {clean_series.mean() + margin_error:.2f})")

else:
    st.info("System Ready. Please upload a CSV file or generate synthetic data to begin analysis.")
