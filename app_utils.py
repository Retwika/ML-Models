import streamlit as st
from ml_models import DEFAULT_PARAMS

def safe_n_features(arr):
    if arr is not None and hasattr(arr, 'shape') and len(arr.shape) > 1:
        return arr.shape[1]
    return '?'

def hyperparam_controls(model_name, key_prefix="", container=st.sidebar):
    params = {}
    # Naive Bayes
    if model_name == "Gaussian NB":
        pass
    elif model_name == "Bernoulli NB":
        pass
    elif model_name == "Multinomial NB":
        pass
    # Ridge
    elif model_name == "Ridge Classifier":
        params["alpha"] = container.slider("alpha (Ridge)", 0.1, 10.0, float(DEFAULT_PARAMS[model_name]["alpha"]), step=0.1, key=f"{key_prefix}{model_name}_alpha")
    # Logistic Regression (L1/L2)
    elif model_name in ["Logistic Regression (L2)", "Logistic Regression (L1)"]:
        penalty = "l2" if model_name == "Logistic Regression (L2)" else "l1"
        params["penalty"] = container.selectbox("Penalty", ["l1", "l2"], index=0 if penalty == "l1" else 1, key=f"{key_prefix}{model_name}_penalty")
        params["C"] = container.slider("C (Inverse Reg)", 0.01, 10.0, float(DEFAULT_PARAMS[model_name]["C"]), key=f"{key_prefix}{model_name}_C")
    # KNN
    elif model_name == "KNN":
        params["n_neighbors"] = container.slider("n_neighbors(KNN)", 1, 15, DEFAULT_PARAMS[model_name]["n_neighbors"], key=f"{key_prefix}{model_name}_n_neighbors")
    # SVM
    elif model_name == "SVM":
        params["C"] = container.slider("C (SVM)", 0.01, 10.0, float(DEFAULT_PARAMS[model_name]["C"]), key=f"{key_prefix}{model_name}_C")
    # Decision Tree
    elif model_name == "Decision Tree":
        params["max_depth"] = container.slider("max_depth(DT)", 1, 10, DEFAULT_PARAMS[model_name]["max_depth"], key=f"{key_prefix}{model_name}_max_depth")
    # Random Forest
    elif model_name == "Random Forest":
        params["n_estimators"] = container.slider("n_estimators(RF)", 10, 200, DEFAULT_PARAMS[model_name]["n_estimators"], key=f"{key_prefix}{model_name}_n_estimators")
        params["max_depth"] = container.slider("max_depth(RF)", 1, 10, DEFAULT_PARAMS[model_name]["max_depth"], key=f"{key_prefix}{model_name}_max_depth")
    # Gradient Boosting
    elif model_name == "Gradient Boosting":
        params["learning_rate"] = container.slider("learning_rate", 0.01, 1.0, float(DEFAULT_PARAMS[model_name]["learning_rate"]), step=0.01, key=f"{key_prefix}{model_name}_lr")
        params["n_estimators"] = container.slider("n_estimators", 10, 200, DEFAULT_PARAMS[model_name]["n_estimators"], key=f"{key_prefix}{model_name}_n_estimators")
        params["max_depth"] = container.slider("max_depth", 1, 10, DEFAULT_PARAMS[model_name]["max_depth"], key=f"{key_prefix}{model_name}_max_depth")
    # XGBoost, LightGBM, CatBoost
    elif model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        params["learning_rate"] = container.slider("learning_rate", 0.01, 1.0, float(DEFAULT_PARAMS[model_name]["learning_rate"]), step=0.01, key=f"{key_prefix}{model_name}_lr")
        params["n_estimators"] = container.slider("n_estimators", 10, 200, DEFAULT_PARAMS[model_name]["n_estimators"], key=f"{key_prefix}{model_name}_n_estimators")
        params["max_depth"] = container.slider("max_depth", 1, 10, DEFAULT_PARAMS[model_name]["max_depth"], key=f"{key_prefix}{model_name}_max_depth")
    # MLP
    elif model_name == "MLP Neural Net":
        hls = container.text_input("hidden_layer_sizes (comma-separated)", value="100", key=f"{key_prefix}{model_name}_hls")
        params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hls.split(",") if x.strip().isdigit())
        params["activation"] = container.selectbox("activation", ["relu", "tanh", "logistic"], key=f"{key_prefix}{model_name}_activation")
        params["max_iter"] = container.slider("max_iter", 100, 1000, DEFAULT_PARAMS[model_name]["max_iter"], step=50, key=f"{key_prefix}{model_name}_max_iter")
    # Voting
    elif model_name == "Voting Ensemble":
        base_opts = ["Logistic Regression (L2)", "Random Forest", "Gradient Boosting"]
        params["estimators"] = container.multiselect("Base Learners", base_opts, default=base_opts, key=f"{key_prefix}{model_name}_base")
        params["voting"] = container.selectbox("Voting", ["soft", "hard"], key=f"{key_prefix}{model_name}_voting")
    # Stacking
    elif model_name == "Stacking Ensemble":
        base_opts = ["Random Forest", "Gradient Boosting"]
        params["estimators"] = container.multiselect("Base Learners", base_opts, default=base_opts, key=f"{key_prefix}{model_name}_base")
        params["final_estimator"] = container.selectbox("Blender", ["Logistic Regression (L2)", "Ridge Classifier"], key=f"{key_prefix}{model_name}_blender")
    # KMeans
    elif model_name == "KMeans (Unsupervised)":
        params["n_clusters"] = container.slider("n_clusters", 2, 10, DEFAULT_PARAMS[model_name]["n_clusters"], key=f"{key_prefix}{model_name}_n_clusters")
    return params 