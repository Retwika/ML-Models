import streamlit as st
import numpy as np
import pandas as pd
from ml_models import (
    get_dataset, get_model, get_metrics, get_model_explanation, plotly_decision_boundary,
    get_metric_definitions, get_available_models, DEFAULT_PARAMS
)
from app_utils import safe_n_features, hyperparam_controls
from data_utils import load_csv_data, preview_dataframe
from io import BytesIO
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from ml_models import bokeh_decision_boundary

metric_defs = get_metric_definitions()

st.set_page_config(page_title="ML Classifier Explorer", layout="wide")
st.title("🧠 ML Classifier Explorer")

# --- Sidebar ---
st.sidebar.header("Options")
dataset_source = st.sidebar.radio("Dataset Source", ["Sample", "Upload CSV"])
hide_conf_matrix = st.sidebar.checkbox("Hide Confusion Matrix", value=False)
hide_data_preview = st.sidebar.checkbox("Hide Data Preview", value=False)

uploaded_data = None
X_upload, y_upload = None, None
if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (last column = target)", type=["csv"])
    if uploaded_file is not None:
        X_upload, y_upload, uploaded_data = load_csv_data(uploaded_file)
    else:
        X_upload, y_upload, uploaded_data = None, None, False

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.9, 0.2, step=0.05)

st.sidebar.subheader("Test Set (optional)")
test_mode = st.sidebar.radio(
    "Test set contains:",
    ["Features only (predict)", "Features + true labels (analyze)"],
    index=1
)
test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"], key="test_csv")
X_test, y_test, test_data_loaded = None, None, False
if test_file is not None:
    df_test = pd.read_csv(test_file)
    if test_mode == "Features only (predict)":
        # Use all columns as features, no y
        X_test = df_test.iloc[:, 1:].values
        y_test = None
        test_data_loaded = True
    else:
        if df_test.shape[1] < 2:
            st.sidebar.error("Test CSV must have at least 2 columns (features + target)")
        else:
            X_test = df_test.iloc[:, :-1].values
            y_test = df_test.iloc[:, -1].values
            test_data_loaded = True

if dataset_source == "Sample":
    dataset_name = st.sidebar.selectbox("Dataset", ["make_moons", "make_circles", "make_classification"])
else:
    dataset_name = None

model_name = st.sidebar.selectbox("Model", get_available_models())
params = hyperparam_controls(model_name)

if dataset_source == "Sample":
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.2)
    n_samples = st.sidebar.slider("Sample Size", 50, 500, 200)
    random_state = st.sidebar.slider("Random Seed", 0, 100, 42)
else:
    noise = None
    n_samples = None
    random_state = st.sidebar.slider("Random Seed", 0, 100, 42)

tabs = st.tabs(["Single Model", "Comparison"])

with tabs[0]:
    # Single model mode
    if dataset_source == "Sample":
        if dataset_name and n_samples is not None and noise is not None:
            X, y = get_dataset(dataset_name, n_samples, noise, random_state)
        else:
            X, y = None, None
    elif uploaded_data:
        X, y = X_upload, y_upload
    else:
        X, y = None, None
    if X is not None and y is not None and not test_data_loaded:
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if X is not None and y is not None:
        if not hide_data_preview:
            st.markdown("### Data Preview (Train)")
            preview_df = preview_dataframe(X, y)
            if preview_df is not None:
                st.dataframe(preview_df, use_container_width=True, height=150)
            if X_test is not None and y_test is not None and len(y_test) > 0:
                st.markdown("### Data Preview (Test)")
                test_preview_df = preview_dataframe(X_test, y_test)
                if test_preview_df is not None:
                    st.dataframe(test_preview_df, use_container_width=True, height=150)
        try:
            if model_name is not None and isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                y = np.asarray(y)
                # Dimensionality reduction for plotting
                if isinstance(X, np.ndarray) and X.shape[1] > 2:
                    use_pca_plot = st.sidebar.checkbox("Use PCA for 2D plot", value=True, help="Project data to 2D using PCA for visualization.")
                    if use_pca_plot:
                        pca = PCA(n_components=2)
                        X_plot = pca.fit_transform(X)
                        X_test_plot = pca.transform(X_test) if X_test is not None and X_test.shape[1] == X.shape[1] else None
                        st.info("Plot shows first 2 principal components (PCA) of your data.")
                    else:
                        feature_indices = st.sidebar.multiselect(
                            "Select 2 features for plotting",
                            options=list(range(X.shape[1])),
                            default=[0, 1],
                            help="Choose which features to use for the 2D decision boundary plot."
                        )
                        if len(feature_indices) == 2:
                            X_plot = X[:, feature_indices]
                            X_test_plot = X_test[:, feature_indices] if X_test is not None and isinstance(X_test, np.ndarray) else None
                            st.info(f"Plotting using features: Feature {feature_indices[0]} and Feature {feature_indices[1]}")
                        else:
                            st.warning("Please select exactly 2 features for plotting.")
                            X_plot = X[:, :2]
                            X_test_plot = X_test[:, :2] if X_test is not None and isinstance(X_test, np.ndarray) else None
                else:
                    X_plot = X
                    X_test_plot = X_test
                if X_plot is None or X_plot.shape[1] != 2 or len(y) == 0:
                    st.warning("Cannot plot decision boundary: need exactly 2 features after PCA/selection and non-empty data.")
                else:
                    # Train a model for metrics and explanations
                    model = get_model(model_name, params, random_state)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                    metrics = get_metrics(y, y_pred, y_proba)
                    explanation = get_model_explanation(model_name, dataset_name or 'CSV', model, X, y)

                    # Train a separate model for the plot on X_plot
                    col1, col2 = st.columns([1, 1])
                    # st.write("X_plot shape:", X_plot.shape)
                    # st.write("y shape:", y.shape)
                    
                    if np.isnan(X_plot).any() or np.isinf(X_plot).any():
                        st.warning("X_plot contains NaN or inf values!")
                    if np.isnan(y).any() or np.isinf(y).any():
                        st.warning("y contains NaN or inf values!")
                    # st.write("X_plot shape:", X_plot.shape)
                    # st.write("y shape:", y.shape)   
                    with col1:
                        st.subheader(f"{model_name} Decision Boundary")
                        model_plot = get_model(model_name, params, random_state)
                        model_plot.fit(X_plot, y)
                        bokeh_fig = bokeh_decision_boundary(model_plot, X_plot, y, title="Decision Boundary")
                        st.bokeh_chart(bokeh_fig, use_container_width=True)
                    with col2:
                        st.markdown("<h5>Metrics (Train)</h5>", unsafe_allow_html=True)
                        st.write("""
                            <style>
                            .compact-metric-table td, .compact-metric-table th {
                                padding: 0.1em 0.4em;
                                font-size: 0.95em;
                                border-bottom: 1px solid #eee;
                            }
                            .compact-metric-table {
                                border-collapse: collapse;
                                width: 100%;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        train_metrics_rows = []
                        for k, v in metrics['main'].items():
                            label = f"<span title='{metric_defs.get(k, '')}'>{k} <span style='color:#888'>ℹ️</span></span>"
                            train_metrics_rows.append({"Metric": label, "Value": v})
                        st.write(pd.DataFrame(train_metrics_rows).to_html(escape=False, index=False, classes='compact-metric-table'), unsafe_allow_html=True)
                        if 'Confusion Matrix' in metrics['extra'] and not hide_conf_matrix:
                            st.subheader("Confusion Matrix (Train)")
                            st.dataframe(metrics['extra']['Confusion Matrix'], use_container_width=True)
                        if X_test is not None and len(X_test) > 0:
                            if test_mode == "Features only (predict)":
                                # Only prediction, no analysis
                                y_test_pred = model.predict(X_test)
                                y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                                pred_df = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])
                                pred_df["Predicted"] = y_test_pred
                                if y_test_proba is not None:
                                    for i in range(y_test_proba.shape[1]):
                                        pred_df[f"Prob_Class_{i}"] = y_test_proba[:, i]
                                
                                st.write("Model used for test prediction:", model)

                                st.write("Test Set Predictions Preview:")
                                st.dataframe(pred_df.head())
                                csv = pred_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Test Set Predictions as CSV",
                                    data=csv,
                                    file_name="test_predictions.csv",
                                    mime="text/csv"
                                )
                                st.info("No true labels provided for the test set. Only predictions are shown.")
                            elif y_test is not None and len(y_test) > 0:
                                # Analysis as before
                                y_test_pred = model.predict(X_test)
                                y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                                y_test_pred = np.asarray(y_test_pred)
                                y_test = np.asarray(y_test)
                                test_metrics = get_metrics(y_test, y_test_pred, y_test_proba)
                                st.subheader("Metrics (Test)")
                                test_metrics_rows = []
                                for k, v in test_metrics['main'].items():
                                    label = f"<span title='{metric_defs.get(k, '')}'>{k} <span style='color:#888'>ℹ️</span></span>"
                                    test_metrics_rows.append({"Metric": label, "Value": v})
                                st.write(pd.DataFrame(test_metrics_rows).to_html(escape=False, index=False, classes='compact-metric-table'), unsafe_allow_html=True)
                                if 'Confusion Matrix' in test_metrics['extra'] and not hide_conf_matrix:
                                    st.subheader("Confusion Matrix (Test)")
                                    st.dataframe(test_metrics['extra']['Confusion Matrix'], use_container_width=True)
                                st.markdown(f"**Test Accuracy:** {test_metrics['main']['Accuracy']*100:.2f}%")
                            else:
                                st.info("No test set available for evaluation.")
                        st.subheader("Explanation")
                        st.write(explanation)
                    if st.button("Download Notes as PDF"):
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.cell(0, 10, "ML Classifier Explorer Notes", ln=1, align="C")
                        pdf.ln(10)
                        # Add model parameters
                        pdf.set_font("Arial", size=11)
                        pdf.cell(0, 10, "Model Parameters:", ln=1)
                        for k, v in params.items():
                            pdf.cell(0, 8, f"{k}: {v}", ln=1)
                        pdf.ln(5)
                        pdf.multi_cell(0, 10, f"Model: {model_name}\nDataset: {dataset_name or 'CSV'}\n\nMetrics:\n{metrics}\n\nExplanation:\n{explanation}")
                        pdf_output = BytesIO()
                        pdf_bytes = pdf.output(dest='S')
                        if isinstance(pdf_bytes, str):
                            pdf_bytes = pdf_bytes.encode('latin1')
                        pdf_output.write(pdf_bytes)
                        pdf_output.seek(0)
                        st.download_button(
                            label="Download PDF",
                            data=pdf_output,
                            file_name="ml_classifier_notes.pdf",
                            mime="application/pdf"
                        )
        except Exception as e:
            st.error(f"Model training or evaluation failed: {e}")
    else:
        st.info("Please select or upload a dataset.")

with tabs[1]:
    # Comparison mode
    model_names = st.multiselect("Select two models to compare", get_available_models(), default=["Logistic Regression (L2)", "KNN"])
    if len(model_names) == 2:
        if dataset_source == "Sample":
            if dataset_name and n_samples is not None and noise is not None:
                X, y = get_dataset(dataset_name, n_samples, noise, random_state)
            else:
                X, y = None, None
        elif uploaded_data:
            X, y = X_upload, y_upload
        else:
            X, y = None, None
        if X is not None and y is not None and not test_data_loaded:
            X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if X is not None and y is not None:
            # Dimensionality reduction for plotting
            if isinstance(X, np.ndarray) and X.shape[1] > 2:
                use_pca_plot = st.sidebar.checkbox("Use PCA for 2D plot (comparison)", value=True, help="Project data to 2D using PCA for visualization.")
                if use_pca_plot:
                    pca = PCA(n_components=2)
                    X_plot = pca.fit_transform(X)
                    X_test_plot = pca.transform(X_test) if X_test is not None and X_test.shape[1] == X.shape[1] else None
                    st.info("Plot shows first 2 principal components (PCA) of your data.")
                else:
                    feature_indices = st.sidebar.multiselect(
                        "Select 2 features for plotting (comparison)",
                        options=list(range(X.shape[1])),
                        default=[0, 1],
                        help="Choose which features to use for the 2D decision boundary plot."
                    )
                    if len(feature_indices) == 2:
                        X_plot = X[:, feature_indices]
                        X_test_plot = X_test[:, feature_indices] if X_test is not None and isinstance(X_test, np.ndarray) else None
                        st.info(f"Plotting using features: Feature {feature_indices[0]} and Feature {feature_indices[1]}")
                    else:
                        st.warning("Please select exactly 2 features for plotting.")
                        X_plot = X[:, :2]
                        X_test_plot = X_test[:, :2] if X_test is not None and isinstance(X_test, np.ndarray) else None
            else:
                X_plot = X
                X_test_plot = X_test
            cols = st.columns(2)
            for i, mname in enumerate(model_names):
                with cols[i]:
                    st.markdown(f"### {mname}")
                    mparams = hyperparam_controls(mname, key_prefix=f"col{i}_", container=cols[i])
                    try:
                        if mname is not None and isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                            y = np.asarray(y)
                            X_plot = np.asarray(X_plot)
                            model = get_model(mname, mparams, random_state)
                            model.fit(X, y)
                            y_pred = model.predict(X)
                            y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                            y_pred = np.asarray(y_pred)
                            metrics = get_metrics(y, y_pred, y_proba)
                            explanation = get_model_explanation(mname, dataset_name or 'CSV', model, X, y)
                            model_plot = get_model(mname, mparams, random_state)
                            model_plot.fit(X_plot, y)
                            bokeh_fig = bokeh_decision_boundary(model_plot, X_plot, y, title="Decision Boundary")
                            st.bokeh_chart(bokeh_fig, use_container_width=True)
                            train_metrics_rows = []
                            for k, v in metrics['main'].items():
                                label = f"<span title='{metric_defs.get(k, '')}'>{k} <span style='color:#888'>ℹ️</span></span>"
                                train_metrics_rows.append({"Metric": label, "Value": v})
                            st.write(pd.DataFrame(train_metrics_rows).to_html(escape=False, index=False, classes='compact-metric-table'), unsafe_allow_html=True)
                            if 'Confusion Matrix' in metrics['extra'] and not hide_conf_matrix:
                                st.subheader("Confusion Matrix (Train)")
                                st.dataframe(metrics['extra']['Confusion Matrix'], use_container_width=True)
                            if X_test is not None and y_test is not None and len(y_test) > 0:
                                y_test_pred = model.predict(X_test)
                                y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                                y_test_pred = np.asarray(y_test_pred)
                                y_test = np.asarray(y_test)
                                test_metrics = get_metrics(y_test, y_test_pred, y_test_proba)
                                test_metrics_rows = []
                                for k, v in test_metrics['main'].items():
                                    label = f"<span title='{metric_defs.get(k, '')}'>{k} <span style='color:#888'>ℹ️</span></span>"
                                    test_metrics_rows.append({"Metric": label, "Value": v})
                                st.write(pd.DataFrame(test_metrics_rows).to_html(escape=False, index=False, classes='compact-metric-table'), unsafe_allow_html=True)
                                if 'Confusion Matrix' in test_metrics['extra'] and not hide_conf_matrix:
                                    st.subheader("Confusion Matrix (Test)")
                                    st.dataframe(test_metrics['extra']['Confusion Matrix'], use_container_width=True)
                                st.markdown(f"**Test Accuracy:** {test_metrics['main']['Accuracy']*100:.2f}%")
                            else:
                                st.info("No test set available for evaluation.")
                            st.write(explanation)
                    except Exception as e:
                        st.error(f"Model training or evaluation failed: {e}")
        else:
            st.info("Please select or upload a dataset.")
    else:
        st.info("Please select exactly two models.")

# --- Educational Tooltips ---
with st.expander("What is Overfitting?"):
    st.write("Overfitting occurs when a model learns the training data too well, including its noise, and performs poorly on new data.")
with st.expander("What is Underfitting?"):
    st.write("Underfitting happens when a model is too simple to capture the underlying pattern of the data.")
with st.expander("What is Non-linearity?"):
    st.write("Non-linearity means the relationship between input and output is not a straight line. Some models can capture non-linear patterns, others cannot.")
with st.expander("Bias-Variance Tradeoff"):
    st.write("Bias is error from erroneous assumptions; variance is error from sensitivity to small fluctuations. Good models balance both.") 