from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):

    le = LabelEncoder()

    for col in df.columns:

        if df[col].dtype == "object":

            df[col] = le.fit_transform(df[col])

    return df