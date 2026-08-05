import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Predictor")
st.markdown("**ML-powered prediction of customer churn using Random Forest**")
st.markdown("---")

# ─── Load & Prepare Data ──────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telecom_Churn.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Synthetic fallback dataset
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "tenure": np.random.randint(1, 72, n),
            "MonthlyCharges": np.random.uniform(20, 120, n),
            "TotalCharges": np.random.uniform(100, 8000, n),
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "PhoneService": np.random.choice(["Yes", "No"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
            "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73])
        })

    # Encode
    df_model = df.copy()

    # FIX #1: keep a separate, dedicated encoder per categorical column
    # (previously a single `le` was refit in a loop, so only the LAST
    # column's encoder survived — every other column's mapping was lost
    # and unavailable later for live prediction)
    encoders = {}

    target_le = LabelEncoder()
    if "Churn" in df_model.columns:
        df_model["Churn"] = target_le.fit_transform(df_model["Churn"].astype(str))

    cat_cols = df_model.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        col_le = LabelEncoder()
        df_model[col] = col_le.fit_transform(df_model[col].astype(str))
        encoders[col] = col_le  # store this column's own encoder for reuse later

    df_model = df_model.apply(pd.to_numeric, errors="coerce").dropna()

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # return encoders + the original df's category options so the live
    # prediction tab can build correct, valid inputs
    return model, X, y, X_test, y_test, y_pred, y_prob, df, encoders

model, X, y, X_test, y_test, y_pred, y_prob, df, encoders = load_and_train()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔍 Data Insights", "🎯 Live Prediction"])

# ─── Tab 1: Model Performance ─────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Model Performance Metrics")

    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_prob)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Accuracy", f"{acc:.1%}")
    col2.metric("📈 ROC-AUC Score", f"{auc:.3f}")
    col3.metric("🌲 Trees in Forest", "100")
    col4.metric("📋 Test Samples", len(y_test))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#FF6B6B", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    st.markdown("**Top 10 Feature Importances**")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    top_features.sort_values().plot(kind="barh", ax=ax, color="#4ECDC4")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── Tab 2: Data Insights ─────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Dataset Insights")

    if "Churn" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Churn Distribution**")
            churn_counts = df["Churn"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(churn_counts.values,
                   labels=churn_counts.index,
                   autopct='%1.1f%%',
                   colors=["#4ECDC4", "#FF6B6B"])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            if "tenure" in df.columns:
                st.markdown("**Tenure Distribution by Churn**")
                fig, ax = plt.subplots(figsize=(5, 4))
                for churn_val in df["Churn"].unique():
                    subset = df[df["Churn"] == churn_val]["tenure"]
                    ax.hist(subset, alpha=0.6, label=str(churn_val), bins=20)
                ax.set_xlabel("Tenure (months)")
                ax.set_ylabel("Count")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    with st.expander("📋 Raw Data Sample"):
        st.dataframe(df.head(20), width="stretch")

# ─── Tab 3: Live Prediction ───────────────────────────────────────────────────
with tab3:
    st.subheader("🎯 Predict Churn for a New Customer")
    st.markdown("Adjust the values below and click **Predict**")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.slider("Monthly Charges (₹)", 20, 120, 65)

    with col2:
        default_total_charges = min(8000.0, max(100.0, float(tenure * monthly_charges)))
        total_charges = st.number_input("Total Charges (₹)", 100.0, 8000.0, default_total_charges)
        senior = st.selectbox("Senior Citizen", [0, 1])

    with col3:
        # FIX #2: pull dropdown options straight from the ORIGINAL (unencoded)
        # data so the labels shown to the user always match what the
        # stored encoder actually knows how to transform
        contract_options = sorted(df["Contract"].unique().tolist()) if "Contract" in df.columns else ["Month-to-month", "One year", "Two year"]
        internet_options = sorted(df["InternetService"].unique().tolist()) if "InternetService" in df.columns else ["DSL", "Fiber optic", "No"]
        contract = st.selectbox("Contract Type", contract_options)
        internet = st.selectbox("Internet Service", internet_options)

    if st.button("🔮 Predict Churn", type="primary"):
        # Build input using model's feature set
        input_row = {col: 0 for col in X.columns}

        # Map known numeric values
        if "tenure" in input_row: input_row["tenure"] = tenure
        if "MonthlyCharges" in input_row: input_row["MonthlyCharges"] = monthly_charges
        if "TotalCharges" in input_row: input_row["TotalCharges"] = total_charges
        if "SeniorCitizen" in input_row: input_row["SeniorCitizen"] = senior

        # FIX #3 (the core bug): actually encode the user's Contract and
        # InternetService selections using the SAME encoders fit during
        # training, instead of silently leaving them at 0. This was the
        # train-serve skew — the model was trained on these columns as
        # real signal, but the live tab never passed real values for them.
        if "Contract" in input_row and "Contract" in encoders:
            input_row["Contract"] = encoders["Contract"].transform([contract])[0]
        if "InternetService" in input_row and "InternetService" in encoders:
            input_row["InternetService"] = encoders["InternetService"].transform([internet])[0]

        input_df = pd.DataFrame([input_row])[X.columns]  # enforce correct column order
        prob = model.predict_proba(input_df)[0][1]
        prediction = "🚨 Likely to Churn" if prob > 0.5 else "✅ Likely to Stay"

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Churn Probability", f"{prob:.1%}")

        if prob > 0.7:
            st.error("⚠️ High risk customer! Consider offering a retention discount.")
        elif prob > 0.4:
            st.warning("🟡 Medium risk. Monitor this customer closely.")
        else:
            st.success("🟢 Low churn risk. Customer appears satisfied.")

st.markdown("---")
st.markdown("*Built by **Jatin Chauhan** | Engineer @ Samsung R&D | [LinkedIn](https://www.linkedin.com/in/jatin-chauhan-a07153171/)*")
