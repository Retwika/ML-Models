import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend
from bokeh.palettes import Category10

# Optional imports for modern boosting
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# --- Model registry and defaults ---

MODELS: Dict[str, Any] = {
    "Logistic Regression (L2)": LogisticRegression,
    "Logistic Regression (L1)": LogisticRegression,
    "Ridge Classifier": RidgeClassifier,
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Gaussian NB": GaussianNB,
    "Bernoulli NB": BernoulliNB,
    "Multinomial NB": MultinomialNB,
    "MLP Neural Net": MLPClassifier,
    "Voting Ensemble": VotingClassifier,
    "Stacking Ensemble": StackingClassifier,
    "KMeans (Unsupervised)": KMeans,
}
if XGBClassifier:
    MODELS["XGBoost"] = XGBClassifier
if LGBMClassifier:
    MODELS["LightGBM"] = LGBMClassifier
if CatBoostClassifier:
    MODELS["CatBoost"] = CatBoostClassifier

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression (L2)": {"C": 1.0, "penalty": "l2"},
    "Logistic Regression (L1)": {"C": 1.0, "penalty": "l1", "solver": "liblinear"},
    "Ridge Classifier": {"alpha": 1.0},
    "KNN": {"n_neighbors": 5},
    "SVM": {"C": 1.0},
    "Decision Tree": {"max_depth": 3},
    "Random Forest": {"n_estimators": 100, "max_depth": 3},
    "Gradient Boosting": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3},
    "Gaussian NB": {},
    "Bernoulli NB": {},
    "Multinomial NB": {},
    "MLP Neural Net": {"hidden_layer_sizes": (100,), "activation": "relu", "max_iter": 200},
    "Voting Ensemble": {"estimators": ["Logistic Regression (L2)", "Random Forest", "Gradient Boosting"], "voting": "soft"},
    "Stacking Ensemble": {"estimators": ["Random Forest", "Gradient Boosting"], "final_estimator": "Logistic Regression (L2)"},
    "KMeans (Unsupervised)": {"n_clusters": 2},
}
if XGBClassifier:
    DEFAULT_PARAMS["XGBoost"] = {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3}
if LGBMClassifier:
    DEFAULT_PARAMS["LightGBM"] = {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3}
if CatBoostClassifier:
    DEFAULT_PARAMS["CatBoost"] = {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3, "verbose": 0}

# --- Data ---

def get_dataset(
    name: str, n_samples: int, noise: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset."""
    if name == "make_moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "make_circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "make_classification":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, flip_y=noise, random_state=random_state
        )
    else:
        raise ValueError("Unknown dataset")
    return np.asarray(X), np.asarray(y)

# --- Model Factory ---

def get_model(
    name: str, params: Dict[str, Any], random_state: Optional[int] = None
) -> Any:
    """Return an untrained model instance for the given name and params."""
    if name == "Logistic Regression (L2)":
        return LogisticRegression(C=params.get("C", 1.0), penalty="l2", solver="lbfgs", random_state=random_state)
    elif name == "Logistic Regression (L1)":
        return LogisticRegression(C=params.get("C", 1.0), penalty="l1", solver="liblinear", random_state=random_state)
    elif name == "Ridge Classifier":
        return RidgeClassifier(alpha=params.get("alpha", 1.0), random_state=random_state)
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    elif name == "SVM":
        return SVC(C=params.get("C", 1.0), probability=True, random_state=random_state)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", 3), random_state=random_state)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", 3), random_state=random_state)
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            random_state=random_state,
        )
    elif name == "Gaussian NB":
        return GaussianNB()
    elif name == "Bernoulli NB":
        return BernoulliNB()
    elif name == "Multinomial NB":
        return MultinomialNB()
    elif name == "MLP Neural Net":
        return MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
            activation=params.get("activation", "relu"),
            max_iter=params.get("max_iter", 200),
            random_state=random_state,
        )
    elif name == "Voting Ensemble":
        # Build base estimators
        estimators = []
        for est_name in params.get("estimators", ["Logistic Regression (L2)", "Random Forest", "Gradient Boosting"]):
            est = get_model(est_name, DEFAULT_PARAMS.get(est_name, {}), random_state)
            estimators.append((est_name, est))
        return VotingClassifier(estimators=estimators, voting=params.get("voting", "soft"))
    elif name == "Stacking Ensemble":
        base_estimators = []
        for est_name in params.get("estimators", ["Random Forest", "Gradient Boosting"]):
            est = get_model(est_name, DEFAULT_PARAMS.get(est_name, {}), random_state)
            base_estimators.append((est_name, est))
        final_est_name = params.get("final_estimator", "Logistic Regression (L2)")
        final_est = get_model(final_est_name, DEFAULT_PARAMS.get(final_est_name, {}), random_state)
        return StackingClassifier(estimators=base_estimators, final_estimator=final_est)
    elif name == "XGBoost" and XGBClassifier:
        return XGBClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        )
    elif name == "LightGBM" and LGBMClassifier:
        return LGBMClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            random_state=random_state,
        )
    elif name == "CatBoost" and CatBoostClassifier:
        return CatBoostClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            verbose=0,
            random_state=random_state,
        )
    elif name == "KMeans (Unsupervised)":
        return KMeans(n_clusters=params.get("n_clusters", 2), random_state=random_state)
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")

# --- Metrics ---

def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute metrics for classification."""
    def safe_scalar(val):
        if isinstance(val, (np.ndarray, list)):
            return float(np.mean(val))
        return float(val)
    metrics = {}
    metrics["Accuracy"] = round(safe_scalar(accuracy_score(y_true, y_pred)), 3)
    metrics["Precision"] = round(safe_scalar(precision_score(y_true, y_pred, average='binary') if len(np.unique(y_true)) == 2 else precision_score(y_true, y_pred, average='macro')), 3)
    metrics["Recall"] = round(safe_scalar(recall_score(y_true, y_pred, average='binary') if len(np.unique(y_true)) == 2 else recall_score(y_true, y_pred, average='macro')), 3)
    metrics["F1-Score"] = round(safe_scalar(f1_score(y_true, y_pred, average='binary') if len(np.unique(y_true)) == 2 else f1_score(y_true, y_pred, average='macro')), 3)
    # ROC AUC and PR AUC
    try:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics["ROC AUC"] = round(safe_scalar(roc_auc_score(y_true, y_proba[:, 1])), 3)
            metrics["PR AUC"] = round(safe_scalar(average_precision_score(y_true, y_proba[:, 1])), 3)
        elif y_proba is not None:
            metrics["ROC AUC"] = round(safe_scalar(roc_auc_score(y_true, y_proba, multi_class='ovr')), 3)
            metrics["PR AUC"] = round(safe_scalar(average_precision_score(y_true, y_proba)), 3)
    except Exception:
        pass
    # Log Loss
    try:
        if y_proba is not None:
            metrics["Log Loss"] = round(safe_scalar(log_loss(y_true, y_proba)), 3)
    except Exception:
        pass
    # Confusion Matrix (as extra)
    extra_metrics = {}
    try:
        cm = confusion_matrix(y_true, y_pred)
        extra_metrics["Confusion Matrix"] = pd.DataFrame(cm)
    except Exception:
        pass
    return {"main": metrics, "extra": extra_metrics}

def get_metric_definitions() -> Dict[str, str]:
    """Return a dictionary of metric names to short definitions."""
    return {
        "Accuracy": "Proportion of correct predictions out of all predictions.",
        "Precision": "Proportion of positive predictions that are actually positive.",
        "Recall": "Proportion of actual positives that are correctly predicted.",
        "F1-Score": "Harmonic mean of precision and recall, balances the two.",
        "ROC AUC": "Area under the ROC curve; measures ability to distinguish classes.",
        "PR AUC": "Area under the Precision-Recall curve; useful for imbalanced data.",
        "Confusion Matrix": "Table showing counts of true vs. predicted classes.",
        "Log Loss": "Penalty for incorrect predicted probabilities; lower is better."
    }

# --- Explanations ---

def get_model_explanation(
    model_name: str, dataset_name: str, model: Any, X: np.ndarray, y: np.ndarray
) -> str:
    """Return a textual explanation for the model and its decision boundary."""
    explanations = {
        "Logistic Regression (L2)": "Linear model with L2 regularization. Finds a straight line (or hyperplane) to separate classes.",
        "Logistic Regression (L1)": "Linear model with L1 regularization. Can produce sparse solutions (feature selection).",
        "Ridge Classifier": "Linear classifier with L2 regularization, robust to collinearity.",
        "KNN": "Classifies a point based on the majority class among its k closest neighbors. Captures complex boundaries.",
        "SVM": "Finds the optimal boundary (possibly non-linear with kernels) that maximizes the margin between classes.",
        "Decision Tree": "Splits the data into regions by asking a series of questions. Can capture non-linear patterns but may overfit.",
        "Random Forest": "Ensemble of decision trees, reducing overfitting and improving generalization.",
        "Gradient Boosting": "Ensemble of shallow trees built sequentially to correct previous errors. Powerful for tabular data.",
        "Gaussian NB": "Probabilistic model assuming features are normally distributed and independent.",
        "Bernoulli NB": "Naive Bayes for binary/boolean features.",
        "Multinomial NB": "Naive Bayes for count data (e.g., text).",
        "MLP Neural Net": "Feedforward neural network with one or more hidden layers. Can model complex non-linearities.",
        "Voting Ensemble": "Combines predictions from multiple models by majority (hard) or probability (soft) voting.",
        "Stacking Ensemble": "Trains a meta-model (blender) on the outputs of base models for improved performance.",
        "XGBoost": "Efficient, scalable gradient boosting. Often state-of-the-art for tabular data.",
        "LightGBM": "Fast, efficient gradient boosting using histogram-based splits.",
        "CatBoost": "Gradient boosting with excellent categorical feature support.",
        "KMeans (Unsupervised)": "Clusters data into k groups based on feature similarity. No supervision.",
    }
    dataset_explain = {
        "make_moons": "Two interleaving half circles, not linearly separable.",
        "make_circles": "Concentric circles, highly non-linear.",
        "make_classification": "Synthetic dataset with informative and redundant features."
    }
    boundary_interpret = {
        "Logistic Regression (L2)": "The decision boundary is a straight line.",
        "Logistic Regression (L1)": "The decision boundary is a straight line, possibly sparse.",
        "Ridge Classifier": "The decision boundary is a straight line, robust to collinearity.",
        "KNN": "The boundary can be very jagged and follow the data closely.",
        "SVM": "The boundary can be linear or curved depending on the kernel.",
        "Decision Tree": "The boundary is axis-aligned and forms rectangles.",
        "Random Forest": "The boundary is a combination of many axis-aligned splits, smoother than a single tree.",
        "Gradient Boosting": "The boundary is a sum of many shallow trees, can be complex but smooth.",
        "Gaussian NB": "The boundary is quadratic if variances differ, otherwise linear.",
        "Bernoulli NB": "The boundary is linear for binary features.",
        "Multinomial NB": "The boundary is linear for count features.",
        "MLP Neural Net": "The boundary can be highly non-linear and flexible.",
        "Voting Ensemble": "The boundary is a blend of the base models' boundaries.",
        "Stacking Ensemble": "The boundary is learned by a meta-model over base models.",
        "XGBoost": "The boundary is a sum of many shallow trees, can be complex but smooth.",
        "LightGBM": "The boundary is a sum of many shallow trees, can be complex but smooth.",
        "CatBoost": "The boundary is a sum of many shallow trees, can be complex but smooth.",
        "KMeans (Unsupervised)": "No decision boundary; clusters are separated by Voronoi cells.",
    }
    return (
        f"**Model:** {explanations.get(model_name, model_name)}\n\n"
        f"**Dataset:** {dataset_explain.get(dataset_name, dataset_name)}\n\n"
        f"**Boundary:** {boundary_interpret.get(model_name, '')}"
    )

# --- Plotting ---

def plotly_decision_boundary(model, X: np.ndarray, y: np.ndarray, title="Decision Boundary"):
    """
    Plots the decision boundary and data points using Plotly.
    The model must be trained on X (2D).
    """
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    except Exception as e:
        print("Plotly decision boundary error:", e)
        Z = np.zeros_like(xx.ravel())
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    # Decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        showscale=False,
        colorscale="RdBu",
        opacity=0.3,
        contours=dict(showlines=False)
    ))
    # Data points
    for class_value in np.unique(y):
        fig.add_trace(go.Scatter(
            x=X[y == class_value, 0],
            y=X[y == class_value, 1],
            mode='markers',
            marker=dict(size=8, line=dict(width=1, color='black')),
            name=f"Class {class_value}"
        ))
    fig.update_layout(
        xaxis=dict(title='Feature 1'),
        yaxis=dict(title='Feature 2'),
        title=title,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99),
        width=350, height=300
    )
    return fig

def bokeh_decision_boundary(model, X: np.ndarray, y: np.ndarray, title="Decision Boundary"):
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    except Exception as e:
        print("Bokeh decision boundary error:", e)
        Z = np.zeros_like(xx.ravel())
    Z = Z.reshape(xx.shape)

    p = figure(title=title, width=350, height=350, tools="pan,wheel_zoom,box_zoom,reset,save")
    # Decision boundary as image
    p.image(image=[Z], x=x_min, y=y_min, dw=x_max-x_min, dh=y_max-y_min, palette=["#FFAAAA", "#AAAAFF"], level="image", alpha=0.3)
    # Data points
    palette = Category10[10]
    legend_items = []
    for idx, class_value in enumerate(np.unique(y)):
        mask = y == class_value
        source = ColumnDataSource(data=dict(x=X[mask, 0], y=X[mask, 1]))
        r = p.scatter('x', 'y', source=source, size=7, color=palette[idx], legend_label=f"Class {class_value}", line_color="black")
    p.legend.location = "top_left"
    p.xaxis.visible = False
    p.yaxis.visible = False
    return p

def get_available_models() -> list:
    """Return a list of available model names for dropdowns."""
    return list(MODELS.keys()) 