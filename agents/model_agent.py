import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score,
    r2_score
)

def detect_problem_type(y):

    # Object/String → Classification
    if y.dtype == "object":
        return "classification"

    # Float → Regression
    if pd.api.types.is_float_dtype(y):
        return "regression"

    # Integer with small unique values → Classification
    if pd.api.types.is_integer_dtype(y):

        if y.nunique() <= 10:
            return "classification"

        return "regression"

    return "regression"


def train_models(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = {}

    best_score = -999
    best_model = None

    # ======================================
    # DETECT PROBLEM TYPE
    # ======================================

    problem_type = detect_problem_type(y)

    print("Detected Problem Type:", problem_type)

    # ======================================
    # MODELS
    # ======================================

    if problem_type == "classification":

        models = {

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier()

        }

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree Regressor":
                DecisionTreeRegressor(),

            "Random Forest Regressor":
                RandomForestRegressor()

        }

    # ======================================
    # TRAINING
    # ======================================

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        if problem_type == "classification":

            score = accuracy_score(
                y_test,
                predictions
            )

        else:

            score = r2_score(
                y_test,
                predictions
            )

        results[name] = round(score, 4)

        if score > best_score:

            best_score = score
            best_model = model

    # ======================================
    # SAVE MODEL
    # ======================================

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    return results, best_model