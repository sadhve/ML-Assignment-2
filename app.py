import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer ML Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM UI / CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Main page */
    .main {
        background: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Header */
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #2563eb 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        font-weight: 750;
        letter-spacing: -0.8px;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.9;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.75rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 105px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }

    /* Info cards */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .info-title {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .info-text {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #e2e8f0 !important;
        font-weight: 650;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1.1rem;
        border-radius: 10px 10px 0 0;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🩺 Breast Cancer Classification</h1>
        <p>Interactive machine learning dashboard for comparing classification models
        and evaluating breast cancer diagnosis predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD SAVED MODELS
# ============================================================

scaler = joblib.load("model/scaler.pkl")

logistic_model = joblib.load("model/logistic_regression.pkl")
decision_tree_model = joblib.load("model/decision_tree.pkl")
knn_model = joblib.load("model/knn.pkl")
naive_bayes_model = joblib.load("model/naive_bayes.pkl")
random_forest_model = joblib.load("model/random_forest.pkl")

model_options = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "K-Nearest Neighbors": knn_model,
    "Gaussian Naive Bayes": naive_bayes_model,
    "Random Forest": random_forest_model,
}

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Model Controls")
    st.caption("Choose a model and upload your test dataset.")

    selected_model_name = st.selectbox(
        "Machine Learning Model",
        list(model_options.keys()),
    )

    selected_model = model_options[selected_model_name]

    st.markdown("---")
    st.markdown("### 📁 Test Dataset")

    uploaded_file = st.file_uploader(
        "Upload Test Data (CSV)",
        type=["csv"],
        help="The CSV must contain a 'diagnosis' column.",
    )

    st.markdown("---")
    st.markdown("### 📌 Models Included")
    st.markdown(
        """
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Random Forest
        """
    )

# ============================================================
# LANDING STATE
# ============================================================

if uploaded_file is None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">📊 Compare Models</div>
                <div class="info-text">
                    Evaluate five classification algorithms using the same test data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">🎯 Evaluate Performance</div>
                <div class="info-text">
                    Review Accuracy, AUC, Precision, Recall, F1 Score and MCC.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">🔍 Inspect Predictions</div>
                <div class="info-text">
                    Explore the confusion matrix, classification report and predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("👈 Upload your test CSV from the sidebar to begin the analysis.")

    st.markdown(
        '<div class="footer">Breast Cancer Classification • Machine Learning Assignment</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ============================================================
# READ TEST DATA
# ============================================================

test_data = pd.read_csv(uploaded_file)
test_data.columns = test_data.columns.str.strip()

target_column = "diagnosis"

if target_column not in test_data.columns:
    st.error("The uploaded CSV must contain a 'diagnosis' column.")
    st.write("Columns detected:", test_data.columns.tolist())
    st.stop()

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric("Samples", f"{len(test_data):,}")

with d2:
    st.metric("Features", f"{X_test.shape[1]:,}")

with d3:
    st.metric("Class 0", f"{(y_test == 0).sum():,}")

with d4:
    st.metric("Class 1", f"{(y_test == 1).sum():,}")

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(test_data.head(10), use_container_width=True, hide_index=True)

# ============================================================
# MODEL COMPARISON
# ============================================================

comparison_results = []

for model_name, model in model_options.items():

    if model_name in [
        "Logistic Regression",
        "K-Nearest Neighbors",
    ]:
        X_input_model = scaler.transform(X_test)
    else:
        X_input_model = X_test

    model_pred = model.predict(X_input_model)
    model_prob = model.predict_proba(X_input_model)[:, 1]

    model_accuracy = accuracy_score(y_test, model_pred)
    model_auc = roc_auc_score(y_test, model_prob)
    model_precision = precision_score(
        y_test, model_pred, zero_division=0
    )
    model_recall = recall_score(
        y_test, model_pred, zero_division=0
    )
    model_f1 = f1_score(
        y_test, model_pred, zero_division=0
    )
    model_mcc = matthews_corrcoef(y_test, model_pred)

    comparison_results.append(
        {
            "Model": model_name,
            "Accuracy": model_accuracy,
            "AUC": model_auc,
            "Precision": model_precision,
            "Recall": model_recall,
            "F1 Score": model_f1,
            "MCC": model_mcc,
        }
    )

comparison_df = pd.DataFrame(comparison_results)

# ============================================================
# SELECTED MODEL PREDICTIONS
# ============================================================

if selected_model_name in [
    "Logistic Regression",
    "K-Nearest Neighbors",
]:
    X_input = scaler.transform(X_test)
else:
    X_input = X_test

y_pred = selected_model.predict(X_input)
y_prob = selected_model.predict_proba(X_input)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# ============================================================
# TOP-LEVEL MODEL STATUS
# ============================================================

best_model_row = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
best_model_name = best_model_row["Model"]
best_f1 = best_model_row["F1 Score"]

st.markdown('<div class="section-title">🏆 Model Comparison</div>', unsafe_allow_html=True)

st.caption(
    f"Selected model: **{selected_model_name}**  •  "
    f"Best F1 Score: **{best_model_name} ({best_f1:.4f})**"
)

# Highlight the best model in the comparison table.
display_df = comparison_df.copy()

for column in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC",
]:
    display_df[column] = display_df[column].map(lambda x: f"{x:.4f}")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# SELECTED MODEL PERFORMANCE
# ============================================================

st.markdown(
    f'<div class="section-title">🎯 {selected_model_name} Performance</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(6)

metrics = [
    ("Accuracy", accuracy),
    ("AUC", auc),
    ("Precision", precision),
    ("Recall", recall),
    ("F1 Score", f1),
    ("MCC", mcc),
]

for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# DETAILED ANALYSIS TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    ["📊 Confusion Matrix", "📑 Classification Report", "🔎 Predictions"]
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

with tab1:
    cm = confusion_matrix(y_test, y_pred)

    cm_col1, cm_col2 = st.columns([1.25, 1])

    with cm_col1:
        fig, ax = plt.subplots(figsize=(5.2, 4.1))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            annot_kws={"size": 14},
            cbar=False,
            linewidths=1,
            linecolor="white",
            ax=ax,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(
            f"{selected_model_name} — Confusion Matrix",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with cm_col2:
        tn, fp, fn, tp = cm.ravel()

        st.markdown("#### Interpretation")

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">True Negatives</div>
                <div class="info-text">{tn:,} samples correctly classified as Class 0.</div>
            </div>

            <div class="info-card">
                <div class="info-title">False Positives</div>
                <div class="info-text">{fp:,} Class 0 samples incorrectly predicted as Class 1.</div>
            </div>

            <div class="info-card">
                <div class="info-title">False Negatives</div>
                <div class="info-text">{fn:,} Class 1 samples incorrectly predicted as Class 0.</div>
            </div>

            <div class="info-card">
                <div class="info-title">True Positives</div>
                <div class="info-text">{tp:,} samples correctly classified as Class 1.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

with tab2:
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Class 0", "Class 1"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_table = report_df.drop(
        index="accuracy",
        errors="ignore",
    )

    st.dataframe(
        report_table.round(4),
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">Overall Accuracy</div>
            <div class="info-text">
                The selected model correctly classified
                <strong>{accuracy:.4f}</strong> of the test samples.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

with tab3:
    prediction_df = pd.DataFrame(
        {
            "Actual Diagnosis": y_test.values,
            "Predicted Diagnosis": y_pred,
            "Prediction Probability": y_prob,
        }
    )

    prediction_df.index.name = "Sample"

    # Add a simple correctness indicator for easier visual inspection.
    prediction_df["Result"] = prediction_df.apply(
        lambda row: "✓ Correct"
        if row["Actual Diagnosis"] == row["Predicted Diagnosis"]
        else "✗ Incorrect",
        axis=1,
    )

    st.dataframe(
        prediction_df.round(4),
        use_container_width=True,
    )

    st.caption(
        "Prediction Probability represents the model's probability for Class 1."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">🩺 Breast Cancer Classification • ML Assignment 2</div>',
    unsafe_allow_html=True,
)
