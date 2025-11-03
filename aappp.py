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

know_insulin = st.sidebar.checkbox("I know my insulin level", False)

if know_insulin:
    insulin = st.sidebar.number_input("Insulin Level (µU/mL)", 0, 900, 80)
else:
    insulin = 80  # average normal value
    st.sidebar.write("Using average insulin level: 80 µU/mL")


height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 165.0)
weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 65.0)

# Calculate BMI
bmi = weight / ((height / 100) ** 2)
st.sidebar.write(f"Calculated BMI: {bmi:.1f}")
family_history = st.sidebar.selectbox(
    "Family History of Diabetes",
    ("No one", "Grandfather", "Father"),
    help="Select if any close family member has diabetes."
)

# Convert dropdown choice into a numeric value for the model
if family_history == "No one":
    dpf = 0.1
elif family_history == "Grandfather":
    dpf = 0.5
else:  # Father
    dpf = 0.8
age = st.sidebar.number_input("Age", 1, 120, 30)

# --- Predict Button ---
predict = st.sidebar.button("Predict Diabetes Status")

# --- Main section ---
if predict:
    # Prepare data
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    # user_input_scaled = scaler.transform(user_input)
    prediction_prob = model.predict_proba(user_input)[0][1]

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
    st.image(img, caption="Diabetes Detection", use_column_width=True)

# --- Footer ---
st.sidebar.markdown("---")
# st.sidebar.info("Developed by Anshika Goel 💻\nData Source: Diabetes Dataset")










