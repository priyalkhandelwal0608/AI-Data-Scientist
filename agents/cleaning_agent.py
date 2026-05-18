import pandas as pd

def clean_data(df):

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].mean())

    return df