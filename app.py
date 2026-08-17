import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Dry Bean Classification")

st.write(
    "This application evaluates five machine learning "
    "classification models on the Dry Bean test dataset."
)

st.write(
    "**Models:** Logistic Regression, Decision Tree, "
    "K-Nearest Neighbors, Naive Bayes and Random Forest"
)


# ============================================================
# MODEL FILE PATHS
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


# ============================================================
# FUNCTION TO LOAD PICKLE FILE
# ============================================================

@st.cache_resource
def load_pickle(file_path):

    with open(file_path, "rb") as file:
        return pickle.load(file)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

try:

    label_encoder = load_pickle(
        "model/label_encoder.pkl"
    )

except Exception as e:

    st.error(
        "Could not load label_encoder.pkl. "
        "Make sure the file exists inside the model folder."
    )

    st.stop()


# ============================================================
# LOAD SCALER
# ============================================================

try:

    scaler = load_pickle(
        "model/scaler.pkl"
    )

except Exception as e:

    st.error(
        "Could not load scaler.pkl. "
        "Make sure the StandardScaler used during training "
        "has been saved inside the model folder."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select a classification model:",
    list(MODEL_PATHS.keys())
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file to evaluate "
        "the classification models."
    )

    st.stop()


# ============================================================
# READ TEST DATA
# ============================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read the uploaded CSV file: {e}")

    st.stop()


st.success("Test dataset uploaded successfully!")


# ============================================================
# DISPLAY DATASET
# ============================================================

st.subheader("Test Dataset")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Number of Samples",
        test_data.shape[0]
    )

with col2:

    st.metric(
        "Number of Features",
        test_data.shape[1] - 1
    )

st.dataframe(
    test_data.head(10),
    use_container_width=True
)


# ============================================================
# CHECK CLASS COLUMN
# ============================================================

if "Class" not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a 'Class' column."
    )

    st.stop()


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = test_data.drop(
    "Class",
    axis=1
)

y_test_original = test_data["Class"]


# ============================================================
# CONVERT CLASS LABELS
# ============================================================

try:

    y_test = label_encoder.transform(
        y_test_original
    )

except Exception as e:

    st.error(
        "The Class values in the uploaded test data "
        "do not match the classes used during model training."
    )

    st.stop()


# ============================================================
# CHECK FEATURE ORDER
# ============================================================

expected_features = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

missing_features = [
    feature
    for feature in expected_features
    if feature not in X_test.columns
]

if missing_features:

    st.error(
        "The following required features are missing from "
        "the uploaded dataset:"
    )

    st.write(missing_features)

    st.stop()


# Keep exactly the same feature order used during training
X_test = X_test[expected_features]


# ============================================================
# FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    X,
    y
):

    # Logistic Regression and KNN
    # were trained using StandardScaler

    if model_name in [
        "Logistic Regression",
        "K-Nearest Neighbors"
    ]:

        X_input = scaler.transform(X)

    else:

        X_input = X


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(X_input)


    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    y_probability = None

    if hasattr(model, "predict_proba"):

        try:

            y_probability = model.predict_proba(
                X_input
            )

        except Exception:

            y_probability = None


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        y_pred
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = recall_score(
        y,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    f1 = f1_score(
        y,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # MCC
    # --------------------------------------------------------

    mcc = matthews_corrcoef(
        y,
        y_pred
    )


    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    auc = None

    if y_probability is not None:

        try:

            auc = roc_auc_score(
                y,
                y_probability,
                multi_class="ovr",
                average="weighted"
            )

        except Exception:

            auc = None


    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "Predictions": y_pred
    }


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

try:

    model = load_pickle(
        MODEL_PATHS[selected_model]
    )

except Exception as e:

    st.error(
        f"Unable to load {selected_model}: {e}"
    )

    st.stop()


# ============================================================
# EVALUATE SELECTED MODEL
# ============================================================

try:

    results = evaluate_model(
        model,
        selected_model,
        X_test,
        y_test
    )

except Exception as e:

    st.error(
        f"Error while making predictions: {e}"
    )

    st.stop()


# ============================================================
# DISPLAY EVALUATION METRICS
# ============================================================

st.header("2. Evaluation Metrics")

st.subheader(
    f"Results for {selected_model}"
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Accuracy",
        f"{results['Accuracy']:.4f}"
    )

with col2:

    if results["AUC"] is not None:

        st.metric(
            "AUC Score",
            f"{results['AUC']:.4f}"
        )

    else:

        st.metric(
            "AUC Score",
            "N/A"
        )

with col3:

    st.metric(
        "Precision",
        f"{results['Precision']:.4f}"
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Recall",
        f"{results['Recall']:.4f}"
    )

with col5:

    st.metric(
        "F1 Score",
        f"{results['F1']:.4f}"
    )

with col6:

    st.metric(
        "MCC",
        f"{results['MCC']:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("3. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    results["Predictions"]
)


class_names = label_encoder.classes_


fig, ax = plt.subplots(
    figsize=(10, 8)
)

image = ax.imshow(cm)


ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)


ax.set_xticks(
    np.arange(len(class_names))
)

ax.set_yticks(
    np.arange(len(class_names))
)

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    class_names
)


# Display values inside the matrix

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


fig.colorbar(
    image,
    ax=ax
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("4. Classification Report")

report = classification_report(
    y_test,
    results["Predictions"],
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# ALL MODEL COMPARISON
# ============================================================

st.header("5. Comparison of All Models")

st.write(
    "The table below shows the performance of all five "
    "classifiers on the uploaded test data."
)


comparison_results = []


for model_name, model_path in MODEL_PATHS.items():

    try:

        current_model = load_pickle(
            model_path
        )

        current_results = evaluate_model(
            current_model,
            model_name,
            X_test,
            y_test
        )


        comparison_results.append({

            "ML Model": model_name,

            "Accuracy":
                current_results["Accuracy"],

            "AUC":
                current_results["AUC"],

            "Precision":
                current_results["Precision"],

            "Recall":
                current_results["Recall"],

            "F1":
                current_results["F1"],

            "MCC":
                current_results["MCC"]

        })


    except Exception as e:

        st.warning(
            f"Could not evaluate {model_name}: {e}"
        )


comparison_df = pd.DataFrame(
    comparison_results
)


# ============================================================
# DISPLAY COMPARISON TABLE
# ============================================================

if not comparison_df.empty:

    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}"
            }
        ),
        use_container_width=True
    )


# ============================================================
# BEST MODEL
# ============================================================

if not comparison_df.empty:

    best_index = comparison_df[
        "Accuracy"
    ].idxmax()

    best_model = comparison_df.loc[
        best_index
    ]


    st.success(
        f"🏆 Overall best model based on Accuracy: "
        f"**{best_model['ML Model']}** "
        f"with an Accuracy of "
        f"**{best_model['Accuracy']:.4f}**"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dry Bean Classification | Machine Learning Assignment - 2"
)