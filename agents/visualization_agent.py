import matplotlib.pyplot as plt
import streamlit as st

def generate_visuals(df):

    st.subheader("Dataset Correlation")

    correlation = df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    cax = ax.matshow(correlation)

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    fig.colorbar(cax)

    st.pyplot(fig)