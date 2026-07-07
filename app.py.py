import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Personal Data Summarizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Personal Data Summarizer")
st.markdown("Upload your CSV or Excel file and get an instant data summary with insights.")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# ----------------------------
# Data Quality Score
# ----------------------------
def data_quality_score(df):

    total_cells = df.shape[0] * df.shape[1]

    missing = df.isnull().sum().sum()

    duplicates = df.duplicated().sum()

    score = 100

    if total_cells > 0:
        score -= (missing / total_cells) * 100

    if len(df) > 0:
        score -= (duplicates / len(df)) * 100

    score = max(0, round(score, 2))

    return score

# ----------------------------
# Auto Insights
# ----------------------------
def generate_insights(df):

    insights = []

    insights.append(f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing):
        insights.append(
            f"Found missing values in **{len(missing)} columns**."
        )
    else:
        insights.append("No missing values found.")

    dup = df.duplicated().sum()

    insights.append(f"Duplicate Rows : **{dup}**")

    numeric = df.select_dtypes(include=np.number)

    if len(numeric.columns):

        highest_std = numeric.std().idxmax()

        insights.append(
            f"Highest variation found in **{highest_std}**."
        )

    return insights

# ----------------------------
# Main App
# ----------------------------
if uploaded_file:

    df = load_data(uploaded_file)

    # ----------------------------
    # Dataset Preview
    # ----------------------------
    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    # ----------------------------
    # Metrics
    # ----------------------------
    rows = df.shape[0]
    cols = df.shape[1]
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    quality = data_quality_score(df)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing", missing)
    c4.metric("Duplicates", duplicates)
    c5.metric("Quality Score", f"{quality}%")

    st.divider()

    # ----------------------------
    # Dataset Information
    # ----------------------------
    st.subheader("Dataset Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(info, use_container_width=True)

    memory = df.memory_usage(deep=True).sum() / 1024**2

    st.info(f"Memory Usage : {memory:.2f} MB")

    # ----------------------------
    # Summary Statistics
    # ----------------------------
    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    # ----------------------------
    # Missing Values
    # ----------------------------
    st.subheader("Missing Values")

    missing_df = df.isnull().sum().reset_index()

    missing_df.columns = ["Column", "Missing"]

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing",
        text="Missing",
        title="Missing Values by Column"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Data Types
    # ----------------------------
    st.subheader("Column Data Types")

    st.write(df.dtypes)

    # ----------------------------
    # Key Insights
    # ----------------------------
    st.subheader("Automatic Insights")

    insights = generate_insights(df)

    for insight in insights:
        st.success(insight)

    st.divider()
    # =====================================================
    # NUMERICAL ANALYSIS
    # =====================================================

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:

        st.header("📈 Numerical Analysis")

        selected_num = st.selectbox(
            "Select Numerical Column",
            numeric_cols
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                df,
                x=selected_num,
                nbins=30,
                title=f"Distribution of {selected_num}"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            fig = px.box(
                df,
                y=selected_num,
                title=f"Box Plot of {selected_num}"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    if len(numeric_cols) >= 2:

        st.header("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # SCATTER PLOT
    # =====================================================

    if len(numeric_cols) >= 2:

        st.header("📉 Scatter Plot")

        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox(
                "X Axis",
                numeric_cols,
                key="scatter_x"
            )

        with col2:
            y_axis = st.selectbox(
                "Y Axis",
                numeric_cols,
                index=1,
                key="scatter_y"
            )

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # CATEGORICAL ANALYSIS
    # =====================================================

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_cols) > 0:

        st.header("📊 Categorical Analysis")

        selected_cat = st.selectbox(
            "Select Category Column",
            cat_cols
        )

        value_counts = (
            df[selected_cat]
            .value_counts()
            .reset_index()
        )

        value_counts.columns = [
            selected_cat,
            "Count"
        ]

        fig = px.bar(
            value_counts,
            x=selected_cat,
            y="Count",
            text="Count",
            title=f"{selected_cat} Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # OUTLIER DETECTION
    # =====================================================

    if len(numeric_cols) > 0:

        st.header("🚨 Outlier Detection")

        outlier_summary = []

        for col in numeric_cols:

            Q1 = df[col].quantile(0.25)

            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR

            upper = Q3 + 1.5 * IQR

            outliers = df[
                (df[col] < lower) |
                (df[col] > upper)
            ]

            outlier_summary.append(
                {
                    "Column": col,
                    "Outliers": len(outliers)
                }
            )

        outlier_df = pd.DataFrame(outlier_summary)

        st.dataframe(
            outlier_df,
            use_container_width=True
        )

    # =====================================================
    # DOWNLOAD SUMMARY
    # =====================================================

    st.header("📥 Download Report")

    summary = df.describe(include="all").T

    csv = summary.to_csv().encode("utf-8")

    st.download_button(
        label="⬇ Download Summary CSV",
        data=csv,
        file_name="dataset_summary.csv",
        mime="text/csv"
    )

    # =====================================================
    # FINAL DATASET OVERVIEW
    # =====================================================

    st.header("📌 Final Overview")

    st.info(f"""
    ✅ Dataset Shape : {df.shape}

    ✅ Numerical Columns : {len(numeric_cols)}

    ✅ Categorical Columns : {len(cat_cols)}

    ✅ Missing Cells : {df.isnull().sum().sum()}

    ✅ Duplicate Rows : {df.duplicated().sum()}

    ✅ Data Quality Score : {quality}%
    """)

else:

    st.info("📂 Please upload a CSV or Excel file to start analysis.")


