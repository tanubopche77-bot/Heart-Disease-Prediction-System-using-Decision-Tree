import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align: center;
    color: #e63946;
    font-size: 40px;
    font-weight: bold;
}
.result-success {
    padding: 15px;
    border-radius: 10px;
    background-color: #d4edda;
    color: #155724;
    font-size: 22px;
    font-weight: bold;
}
.result-danger {
    padding: 15px;
    border-radius: 10px;
    background-color: #f8d7da;
    color: #721c24;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("best_decision_tree_model.pkl")

st.markdown('<p class="title">❤️ Heart Disease Prediction System</p>',
            unsafe_allow_html=True)

st.write("Fill in the patient details below:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

with col3:
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

st.markdown("---")

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    sex_value = 1 if sex == "Male" else 0

    data = pd.DataFrame({
        'age': [age],
        'sex': [sex_value],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(
            '<div class="result-danger">⚠️ High Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-success">✅ Low Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )

    st.progress(float(prob[1]))
    st.write(f"**Disease Probability:** {prob[1]*100:.2f}%")