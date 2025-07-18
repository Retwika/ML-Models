# ML Classifier Explorer

An interactive Streamlit app to visualize, compare, and analyze machine learning classifiers and their decision boundaries.

## Features

- **Interactive decision boundary plots** (Bokeh)
- **Supports many model families:** Logistic Regression, Ridge, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Naive Bayes, MLP, Voting/Stacking, KMeans
- **Upload your own CSV** or use sklearn sample datasets
- **Customizable train/test split**
- **Metrics:** Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Log Loss, Confusion Matrix
- **Test set modes:** 
  - **Features only (predict):** Download predictions as CSV
  - **Features + true labels (analyze):** Full metrics and analysis
- **Educational tooltips and explanations**

## Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/Retwika/ML-Models.git
   cd ML-Models
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   or, with conda:
   ```sh
   conda create -n ML-Models python=3.9
   conda activate ML-Models
   pip install -r requirements.txt
   ```

## Usage

```sh
streamlit run app.py
```

- Use the sidebar to select a dataset, model, and hyperparameters.
- Upload your own CSV for custom data.
- In the test set section, choose whether your test CSV is for prediction only or for analysis with true labels.
- Download predictions as CSV if using “Features only (predict)” mode.

## Notes

- For best results, use Bokeh version 2.4.3 (required by Streamlit).
- If you use XGBoost, LightGBM, or CatBoost, ensure they are compatible with your numpy/pandas version.

## License

MIT

---

**Enjoy exploring machine learning classifiers!**
