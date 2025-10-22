import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Thyroid Cancer Recurrence Prediction")
st.markdown("""
This app predicts the likelihood of thyroid cancer recurrence based on clinical and pathological features.
Upload your patient data or fill out the form below to get predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["Single Prediction", "Batch Prediction", "Model Info"])

def load_model():
    """Load the trained Random Forest model"""
    try:
        # Try different possible model file formats/locations
        try:
            model = joblib.load('rf_model.pkl')
        except:
            model = pickle.load(open('rf_model.pkl', 'rb'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_recurrence(input_data):
    """Make prediction using the loaded model"""
    model = load_model()
    if model is not None:
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    return None, None

if app_mode == "Single Prediction":
    st.header("Single Patient Prediction")
    
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Female", "Male"])
        smoking = st.selectbox("Current Smoking", ["No", "Yes"])
        hx_smoking = st.selectbox("History of Smoking", ["No", "Yes"])
    
    with col2:
        hx_radiotherapy = st.selectbox("History of Radiotherapy", ["No", "Yes"])
        thyroid_function = st.selectbox("Thyroid Function", 
            ["Euthyroid", "Clinical Hypothyroidism", "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"])
        physical_exam = st.selectbox("Physical Examination",
            ["Normal", "Multinodular goiter", "Single nodular goiter-left", "Single nodular goiter-right"])
    with col3:
        adenopathy = st.selectbox("Adenopathy",
            ["No", "Left", "Right", "Posterior", "Extensive"])
        pathology = st.selectbox("Pathology",
            ["Papillary", "Micropapillary", "Hurthel cell"])
        focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
    
    # Additional clinical features
    st.subheader("Clinical Staging")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        t_stage = st.selectbox("T Stage", ["T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
        n_stage = st.selectbox("N Stage", ["N0", "N1a", "N1b"])
    with col5:
        m_stage = st.selectbox("M Stage", ["M0", "M1"])
        risk = st.selectbox("Risk Category", ["Low", "Intermediate", "High"])
    with col6:
        stage = st.selectbox("Overall Stage", ["I", "II", "III", "IVA", "IVB"])
        response = st.selectbox("Treatment Response", 
            ["Excellent", "Indeterminate", "Structural Incomplete"])
    
    # Prediction button
    if st.button("Predict Recurrence Risk"):
        # Convert inputs to model format (one-hot encoding)
        input_dict = {
            'Age': age,
            'Gender_M': 1 if gender == "Male" else 0,
            'Smoking_Yes': 1 if smoking == "Yes" else 0,
            'Hx Smoking_Yes': 1 if hx_smoking == "Yes" else 0,
            'Hx Radiothreapy_Yes': 1 if hx_radiotherapy == "Yes" else 0,
            # Thyroid Function (one-hot)
            'Thyroid Function_Clinical Hypothyroidism': 1 if thyroid_function == "Clinical Hypothyroidism" else 0,
            'Thyroid Function_Euthyroid': 1 if thyroid_function == "Euthyroid" else 0,
            'Thyroid Function_Subclinical Hyperthyroidism': 1 if thyroid_function == "Subclinical Hyperthyroidism" else 0,
            'Thyroid Function_Subclinical Hypothyroidism': 1 if thyroid_function == "Subclinical Hypothyroidism" else 0,
            # Physical Examination (one-hot)
            'Physical Examination_Multinodular goiter': 1 if physical_exam == "Multinodular goiter" else 0,
            'Physical Examination_Normal': 1 if physical_exam == "Normal" else 0,
            'Physical Examination_Single nodular goiter-left': 1 if physical_exam == "Single nodular goiter-left" else 0,
            'Physical Examination_Single nodular goiter-right': 1 if physical_exam == "Single nodular goiter-right" else 0,
            # Adenopathy (one-hot)
            'Adenopathy_Extensive': 1 if adenopathy == "Extensive" else 0,
            'Adenopathy_Left': 1 if adenopathy == "Left" else 0,
            'Adenopathy_No': 1 if adenopathy == "No" else 0,
            'Adenopathy_Posterior': 1 if adenopathy == "Posterior" else 0,
            'Adenopathy_Right': 1 if adenopathy == "Right" else 0,
            # Pathology (one-hot)
            'Pathology_Hurthel cell': 1 if pathology == "Hurthel cell" else 0,
            'Pathology_Micropapillary': 1 if pathology == "Micropapillary" else 0,
            'Pathology_Papillary': 1 if pathology == "Papillary" else 0,
            'Focality_Uni-Focal': 1 if focality == "Uni-Focal" else 0,
            # Risk (one-hot)
            'Risk_Intermediate': 1 if risk == "Intermediate" else 0,
            'Risk_Low': 1 if risk == "Low" else 0,
            # T Stage (one-hot)
            'T_T1b': 1 if t_stage == "T1b" else 0,
            'T_T2': 1 if t_stage == "T2" else 0,
            'T_T3a': 1 if t_stage == "T3a" else 0,
            'T_T3b': 1 if t_stage == "T3b" else 0,
            'T_T4a': 1 if t_stage == "T4a" else 0,
            'T_T4b': 1 if t_stage == "T4b" else 0,
            # N Stage (one-hot)
            'N_N1a': 1 if n_stage == "N1a" else 0,
            'N_N1b': 1 if n_stage == "N1b" else 0,
            'M_M1': 1 if m_stage == "M1" else 0,
            # Stage (one-hot)
            'Stage_II': 1 if stage == "II" else 0,
            'Stage_III': 1 if stage == "III" else 0,
            'Stage_IVA': 1 if stage == "IVA" else 0,
            'Stage_IVB': 1 if stage == "IVB" else 0,
            # Response (one-hot)
            'Response_Excellent': 1 if response == "Excellent" else 0,
            'Response_Indeterminate': 1 if response == "Indeterminate" else 0,
            'Response_Structural Incomplete': 1 if response == "Structural Incomplete" else 0
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure column order matches training data
        # You might need to reorder columns based on your model's expected input
        try:
            model = load_model()
            if hasattr(model, 'feature_names_in_'):
                input_df = input_df[model.feature_names_in_]
        except:
            pass
        
        # Make prediction
        prediction, probability = predict_recurrence(input_df)
        
        if prediction is not None:
            recurrence_prob = probability[0][1]  # Probability of recurrence (class 1)
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction[0] == 1:
                st.error(f"🚨 High Risk of Recurrence: {recurrence_prob:.1%}")
            else:
                st.success(f"✅ Low Risk of Recurrence: {1-recurrence_prob:.1%}")
            
            # Probability gauge
            st.subheader("Recurrence Probability")
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh([0], [recurrence_prob], color='red' if recurrence_prob > 0.3 else 'orange' if recurrence_prob > 0.1 else 'green')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability of Recurrence')
            ax.set_yticks([])
            ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7)
            ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.7)
            st.pyplot(fig)

elif app_mode == "Batch Prediction":
    st.header("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with patient data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_data.head())
            
            # Make batch predictions
            if st.button("Predict Batch"):
                predictions, probabilities = predict_recurrence(batch_data)
                if predictions is not None:
                    results_df = batch_data.copy()
                    results_df['Predicted_Recurrence'] = predictions
                    results_df['Recurrence_Probability'] = probabilities[:, 1]
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results_df[['Predicted_Recurrence', 'Recurrence_Probability']])
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="thyroid_recurrence_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif app_mode == "Model Info":
    st.header("Model Information")
    
    st.subheader("Random Forest Model Details")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier
    - **Target**: Thyroid Cancer Recurrence (Binary Classification)
    - **Features**: 40 clinical and pathological variables
    - **Dataset**: 614 patient records
    """)
    
    st.subheader("Feature Categories")
    st.markdown("""
    1. **Demographic**: Age, Gender, Smoking history
    2. **Clinical**: Thyroid function, Physical examination findings
    3. **Pathological**: Cancer type, Focality, Risk stratification
    4. **Staging**: TNM staging, Overall stage
    5. **Treatment Response**: Post-treatment assessment
    """)

# Footer
st.markdown("---")
st.markdown("*For clinical use only. Predictions should be interpreted by qualified healthcare professionals.*")
