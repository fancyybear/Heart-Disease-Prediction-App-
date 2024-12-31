# Import necessary libraries
import streamlit as st
import numpy as np
import pickle
import base64

# Load the trained model and scaler
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function to set a background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
add_bg_from_local("887008c710cea243a539749f4cd59925.jpg")  # Replace with your image filename

# App title
st.title("Heart Disease Prediction App ❤️")
st.markdown("This app predicts the likelihood of heart disease based on patient information.")

# Sidebar for options
st.sidebar.title("Options")
st.sidebar.markdown("Customize your experience.")
background_option = st.sidebar.radio(
    "Background Options",
    ["Default", "Custom"],
    index=0
)
if background_option == "Custom":
    custom_image = st.sidebar.file_uploader("Upload a background image", type=["jpg", "jpeg", "png"])
    if custom_image:
        add_bg_from_local(custom_image.name)

st.sidebar.markdown("### Created by: Sahid Ahamad")
# st.sidebar.markdown("**Dataset:** Heart Disease UCI")

# Input fields
st.write("Enter the patient's details below:")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina?", options=[0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST Segment (0-2)", options=[0, 1, 2])
ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.selectbox("Thalassemia (0-3)", options=[0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    prob = model.predict_proba(scaled_features)

    if prediction[0] == 1:
        st.error("🚨 High Risk: Heart Disease Detected")
        st.write(f"Prediction Confidence: {prob[0][1] * 100:.2f}%")
    else:
        st.success("✅ Low Risk: No Heart Disease Detected")
        st.write(f"Prediction Confidence: {prob[0][0] * 100:.2f}%")

# Footer
st.sidebar.markdown("#### Tips for Heart Health:")
st.sidebar.markdown("- Regular exercise 🏃")
st.sidebar.markdown("- Healthy diet 🍎")
st.sidebar.markdown("- Regular check-ups 🩺")
