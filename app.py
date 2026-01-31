import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("This is a Machine Learning project: **model is trained separately**, and this app uses the trained model.")

# ---------------------------
# Load trained model
# ---------------------------
try:
    clf = joblib.load("loan_model.pkl")
except:
    st.error("❌ Model file not found! Please run `python train_model.py` first to generate loan_model.pkl")
    st.stop()

# Load dataset ONLY for input options (dropdown values)
df = pd.read_csv("loan_prediction.csv")
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# Separate features
X = df.drop(columns=["Loan_Status"])

# Categorical and Numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

st.subheader("📌 Enter Applicant Details")

# ---------------------------
# Build HTML-like Form
# ---------------------------
with st.form("loan_form"):

    input_data = {}

    # Split form into two columns
    col1, col2 = st.columns(2)

    with col1:
        for col in cat_cols[:len(cat_cols)//2]:
            options = sorted(X[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(f"{col}", options)

    with col2:
        for col in cat_cols[len(cat_cols)//2:]:
            options = sorted(X[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(f"{col}", options)

    st.markdown("### 🔢 Numerical Inputs")
    for col in num_cols:
        default_val = float(X[col].median()) if X[col].notna().sum() > 0 else 0.0
        input_data[col] = st.number_input(f"{col}", value=default_val)

    submit = st.form_submit_button("🔍 Predict Loan Approval")

# ---------------------------
# Predict
# ---------------------------
if submit:
    user_df = pd.DataFrame([input_data])

    prediction = clf.predict(user_df)[0]
    probability = clf.predict_proba(user_df)[0][1]  # probability of approval

    st.markdown("---")
    st.subheader("✅ Prediction Result")

    if prediction == 1:
        st.success(f"🎉 Loan Approved (Y)\n\nApproval Probability: **{probability*100:.2f}%**")
    else:
        st.error(f"❌ Loan Rejected (N)\n\nApproval Probability: **{probability*100:.2f}%**")

    st.write("### 🔎 Applicant Input Summary")
    st.dataframe(user_df, use_container_width=True)
