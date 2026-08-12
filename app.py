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


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide",
)


# -----------------------------
# Title
# -----------------------------

st.title("🩺 Breast Cancer Classification")

st.write(
    "Compare machine learning models for breast cancer diagnosis."
)


# -----------------------------
# Load Saved Models
# -----------------------------

scaler = joblib.load("model/scaler.pkl")

logistic_model = joblib.load(
    "model/logistic_regression.pkl"
)

decision_tree_model = joblib.load(
    "model/decision_tree.pkl"
)

knn_model = joblib.load(
    "model/knn.pkl"
)

naive_bayes_model = joblib.load(
    "model/naive_bayes.pkl"
)

random_forest_model = joblib.load(
    "model/random_forest.pkl"
)


# -----------------------------
# Model Selection
# -----------------------------

model_options = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "K-Nearest Neighbors": knn_model,
    "Gaussian Naive Bayes": naive_bayes_model,
    "Random Forest": random_forest_model,
}


selected_model_name = st.selectbox(
    "Select a Machine Learning Model",
    list(model_options.keys())
)

selected_model = model_options[selected_model_name]


# -----------------------------
# CSV Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)


if uploaded_file is not None:

    # -----------------------------
    # Read Test Data
    # -----------------------------

    test_data = pd.read_csv(uploaded_file)

    test_data.columns = test_data.columns.str.strip()

    st.subheader("Uploaded Test Data")

    st.dataframe(
        test_data.head(),
        use_container_width=True
    )


    # -----------------------------
    # Validate Target Column
    # -----------------------------

    target_column = "diagnosis"

    if target_column not in test_data.columns:

        st.error(
            "The uploaded CSV must contain a 'diagnosis' column."
        )

        st.write(
            "Columns detected:",
            test_data.columns.tolist()
        )

        st.stop()


    # -----------------------------
    # Separate Features and Target
    # -----------------------------

    X_test = test_data.drop(
        columns=[target_column]
    )

    y_test = test_data[target_column]


    # -----------------------------
    # Model Comparison
    # -----------------------------

    st.subheader("Model Comparison")

    comparison_results = []


    for model_name, model in model_options.items():

        # Apply scaling only to models
        # that were trained using scaled data

        if model_name in [
            "Logistic Regression",
            "K-Nearest Neighbors"
        ]:

            X_input_model = scaler.transform(X_test)

        else:

            X_input_model = X_test


        # Predictions

        model_pred = model.predict(
            X_input_model
        )

        model_prob = model.predict_proba(
            X_input_model
        )[:, 1]


        # Metrics

        model_accuracy = accuracy_score(
            y_test,
            model_pred
        )

        model_auc = roc_auc_score(
            y_test,
            model_prob
        )

        model_precision = precision_score(
            y_test,
            model_pred,
            zero_division=0
        )

        model_recall = recall_score(
            y_test,
            model_pred,
            zero_division=0
        )

        model_f1 = f1_score(
            y_test,
            model_pred,
            zero_division=0
        )

        model_mcc = matthews_corrcoef(
            y_test,
            model_pred
        )


        # Store results

        comparison_results.append(
            {
                "Model": model_name,
                "Accuracy": model_accuracy,
                "AUC": model_auc,
                "Precision": model_precision,
                "Recall": model_recall,
                "F1 Score": model_f1,
                "MCC": model_mcc
            }
        )


    # Create comparison DataFrame

    comparison_df = pd.DataFrame(
        comparison_results
    )


    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------
    # Selected Model Preprocessing
    # -----------------------------

    if selected_model_name in [
        "Logistic Regression",
        "K-Nearest Neighbors"
    ]:

        X_input = scaler.transform(
            X_test
        )

    else:

        X_input = X_test


    # -----------------------------
    # Predictions for Selected Model
    # -----------------------------

    y_pred = selected_model.predict(
        X_input
    )

    y_prob = selected_model.predict_proba(
        X_input
    )[:, 1]


    # -----------------------------
    # Evaluation Metrics
    # -----------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    # -----------------------------
    # Selected Model Performance
    # -----------------------------

    st.subheader(
        f"{selected_model_name} - Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    col4, col5, col6 = st.columns(3)


    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    fig, ax = plt.subplots(
        figsize=(2.8, 2.4)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Predicted 0",
            "Predicted 1"
        ],
        yticklabels=[
            "Actual 0",
            "Actual 1"
        ],
        annot_kws={
            "size": 12
        },
        cbar=False,
        ax=ax
    )


    ax.set_xlabel(
        "Predicted Label",
        fontsize=10
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=10
    )

    ax.set_title(
        f"{selected_model_name} - Confusion Matrix",
        fontsize=12
    )


    plt.tight_layout()

    st.pyplot(
        fig,
        width="content"
    )

    plt.close(fig)


    # -----------------------------
    # Classification Report
    # -----------------------------

    st.subheader("Classification Report")


    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Class 0",
            "Class 1"
        ],
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    # Accuracy is a single value,
    # so display it separately.

    report_table = report_df.drop(
        index="accuracy",
        errors="ignore"
    )


    st.dataframe(
        report_table.round(4),
        use_container_width=True
    )


    st.write(
        f"**Accuracy:** {accuracy:.4f}"
    )


    # -----------------------------
    # Prediction Results
    # -----------------------------

    st.subheader("Prediction Results")


    prediction_df = pd.DataFrame(
        {
            "Actual Diagnosis": y_test.values,
            "Predicted Diagnosis": y_pred,
            "Prediction Probability": y_prob
        }
    )


    prediction_df.index.name = "Sample"


    st.dataframe(
        prediction_df.round(4),
        use_container_width=True
    )