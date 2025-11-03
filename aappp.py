import streamlit as st
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler

import joblib
scaler = joblib.load("scaler.joblib")

model = joblib.load("diabetess_model.joblib")

if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
else:
    scaler = None

# --- Title ---
st.title("🩸 Diabetes Detection App")

# --- Sidebar for inputs ---
st.sidebar.header("🧍‍♀️ Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, 100)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

# --- Predict Button ---
predict = st.sidebar.button("Predict Diabetes Status")

# --- Main section ---
if predict:
    # Prepare data
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)
    prediction_prob = model.predict_proba(user_input_scaled)[0][1]

    # Show result
    st.subheader("🔍 Prediction Result:")
    if prediction_prob > 0.5:
        st.error(f"⚠️ Likely Diabetic (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ Likely Not Diabetic (Probability: {prediction_prob:.2f})")

else:
    # --- Show image before prediction ---
    img = Image.open("np_file_194287.jpeg")
    st.markdown("#### 💡 Upload patient details in the sidebar to predict diabetes.")
    st.image(img, caption="Diabetes Detection", use_container_width=True)

# --- Footer ---
st.sidebar.markdown("---")
# st.sidebar.info("Developed by Anshika Goel 💻\nData Source: Diabetes Dataset")







