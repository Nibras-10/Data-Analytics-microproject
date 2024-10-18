import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS to set background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/4d0sht5Y/black-elegant-background-with-copy-space.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);  /* Make header transparent */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# App Title with Custom Styling
st.markdown('<h1 style="font-family: serif; color: white; text-align: center;">✨ Bank Marketing Model Interface </h1>', unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display DataFrame
    st.write("🗂 **Dataset Preview**:")
    st.dataframe(data.head())

    # Data Pre-processing
    st.subheader("🛠 Data Pre-processing")

    # Check for missing values
    missing_values = data.isnull().sum()
    st.write("❌ **Missing Values**:", missing_values)

    # Apply Min-Max Normalization
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # One-Hot Encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Feature Selection
    X = data_encoded.drop("y_yes", axis=1)
    y = data_encoded["y_yes"]
    selector = SelectKBest(score_func=chi2, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    st.write("📊 **Selected Features:**", selected_features)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Model Selection
    model_name = st.selectbox("📈 Choose a Model", ["Random Forest", "Gradient Boosting", "SVM", "k-NN"])

    if st.button("🚀 Train and Evaluate"):
        # Initialize Model
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif model_name == "SVM":
            model = SVC(probability=True, random_state=42)
        elif model_name == "k-NN":
            model = KNeighborsClassifier(n_neighbors=5)

        # Train the Model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        st.write(f"✅ **Accuracy:** {accuracy * 100:.2f}%")
        st.write(f"📏 **MCC:** {mcc:.2f}")

        # Classification Report
        st.text("📝 **Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix for {model_name}')
        st.pyplot(fig)

        # ROC Curve
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_probs):.2f})")
        ax.plot([0, 1], [0, 1], 'k--')  # Random Guess Line
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='best')
        st.pyplot(fig)
