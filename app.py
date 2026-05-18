import streamlit as st
import pandas as pd

from agents.cleaning_agent import clean_data
from agents.feature_agent import feature_engineering
from agents.model_agent import train_models
from agents.visualization_agent import generate_visuals
from agents.report_agent import generate_report
from agents.llm_agent import dataset_summary

st.set_page_config(page_title="AI Data Scientist Agent")

st.title("AI Autonomous Data Scientist Agent")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")
    st.write(df.head())

    # LLM Analysis
    st.subheader("AI Dataset Analysis")

    with st.spinner("AI is analyzing dataset..."):

        summary = dataset_summary(df)

    st.write(summary)

    # Cleaning
    st.subheader("Cleaning Agent")

    cleaned_df = clean_data(df)

    st.write(cleaned_df.head())

    # Feature Engineering
    st.subheader("Feature Engineering Agent")

    processed_df = feature_engineering(cleaned_df)

    st.write(processed_df.head())

    # Target
    target = st.selectbox(
        "Select Target Column",
        processed_df.columns
    )

    # Model Training
    st.subheader("Model Training Agent")

    results, best_model = train_models(
        processed_df,
        target
    )

    st.write(results)

    # Visualizations
    st.subheader("Visualization Agent")

    generate_visuals(processed_df)

    # Report
    st.subheader("Report Generation Agent")

    generate_report(results)

    st.success("Analysis Completed!")