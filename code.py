import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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

    # List of models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }

    # Dropdown to select individual model
    selected_model_name = st.selectbox("🔍 Choose a model to display results individually", list(models.keys()))

    if selected_model_name:
        model = models[selected_model_name]
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Engagement Distribution Visualization
        engaged_count = np.sum(y_pred)
        not_engaged_count = len(y_pred) - engaged_count
        engaged_percent = (engaged_count / len(y_pred)) * 100
        not_engaged_percent = (not_engaged_count / len(y_pred)) * 100

        st.subheader("📊 Engagement Distribution")
        fig, ax = plt.subplots()
        ax.pie([engaged_percent, not_engaged_percent], labels=['Engaged', 'Not Engaged'],
               autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        ax.set_title('Engagement Distribution')
        st.pyplot(fig)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.write(f"### {selected_model_name} Results")
        st.write(f"✅ **Accuracy:** {accuracy * 100:.2f}%")
        st.write(f"📏 **MCC:** {mcc:.2f}")

        st.text("📝 **Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix for {selected_model_name}')
        st.pyplot(fig)

        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"{selected_model_name} (AUC = {roc_auc_score(y_test, y_probs):.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='best')
        st.pyplot(fig)

    # New Customer Prediction Section
    st.subheader("🧑‍💼 New Customer Prediction")

    # Input form for new customer data
    with st.form("new_customer_form"):
        age = st.number_input("Age", min_value=18, max_value=100)
        balance = st.number_input("Balance", min_value=0.0)
        duration = st.number_input("Call Duration", min_value=0)
        campaign = st.number_input("Campaign Contacts", min_value=0)
        pdays = st.number_input("Days Since Last Contact (-1 if never)", min_value=-1)
        previous = st.number_input("Previous Contacts", min_value=0)
        submit = st.form_submit_button("Predict Subscription")

    if submit:
        new_customer = np.array([[age, balance, duration, campaign, pdays, previous]])
        new_customer = scaler.transform(new_customer)  # Normalize input
        prediction = model.predict(new_customer)[0]

        if prediction == 1:
            st.success("✅ The customer is predicted to **subscribe**.")
        else:
            st.error("❌ The customer is predicted **not to subscribe**.")
