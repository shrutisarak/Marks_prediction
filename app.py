import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load the Trained Model
@st.cache_resource
def load_model():
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pickle`. Make sure the file exists in the directory. Details: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.header("📥 Input Parameters")
st.sidebar.write("Adjust the features below to predict outcome:")

number_courses = st.sidebar.number_input(
    label="Number of Courses (`number_courses`)",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
    help="Select total number of enrolled courses."
)

time_study = st.sidebar.number_input(
    label="Daily Study Time in Hours (`time_study`)",
    min_value=0.0,
    max_value=24.0,
    value=4.5,
    step=0.5,
    help="Enter daily hours spent studying."
)

# --- Main Layout ---
st.markdown('<div class="main-header">🎓 Student Score Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by KNN Regression Machine Learning Model</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Input Summary")
    
    # Display Inputs in a clean Table
    input_df = pd.DataFrame({
        'Feature': ['Number of Courses', 'Study Time (Hours)'],
        'Value': [number_courses, time_study]
    })
    st.table(input_df)

    predict_btn = st.button("🔮 Predict Result", use_container_width=True, type="primary")

with col2:
    st.subheader("🎯 Prediction Output")
    
    if predict_btn:
        # Prepare inputs matching model feature names
        features = np.array([[number_courses, time_study]])
        
        # Perform Prediction
        prediction = model.predict(features)[0]
        
        # Output Card
        st.markdown(
            f"""
            <div class="prediction-card">
                <h3>Estimated Score / Result</h3>
                <h1 style="color: #1E88E5; font-size: 3rem; margin: 10px 0;">{prediction:.2f}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.info("👈 Click the **Predict Result** button to see the outcome.")

# --- Footer ---
st.markdown("---")
st.caption("🤖 Model: K-Neighbors Regressor | Deployed via Streamlit")
