import pandas as pd
import streamlit as st
import numpy as np

def load_csv_data(uploaded_file):
    """
    Loads a CSV file, using all numeric columns except the last as features,
    and the last column as the target.
    Returns (X, y, True) or (None, None, False) on error.
    """
    try:
        df = pd.read_csv(uploaded_file)
        if not isinstance(df, pd.DataFrame) or df.shape[1] < 2:
            st.sidebar.error("CSV must have at least 2 columns (features + target)")
            return None, None, False
        # Use all numeric columns except the last as features
        feature_df = df.iloc[:, 1:-1].select_dtypes(include=[np.number])
        if feature_df.shape[1] < 2:
            st.sidebar.warning("Your data has less than 2 numeric features. Cannot plot a decision boundary.")
        X = feature_df.values
        y = df.iloc[:, -1].values
        st.sidebar.success(f"Loaded CSV with shape {df.shape}")
        return X, y, True
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        return None, None, False

def preview_dataframe(X, y, n=5):
    """
    Returns a DataFrame preview of the first n rows of X and y.
    """
    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        preview_df = pd.DataFrame(X).copy()
        preview_df['target'] = y
        return preview_df.head(n)
    return None 