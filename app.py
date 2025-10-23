import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

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
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["Single Prediction", "Model Info"])

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        model_paths = [
            'rf_model.pkl',
            'model/rf_model.pkl',
            './rf_model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                st.sidebar.success(f"✅ Model loaded successfully")
                return model
        
        st.error("❌ Model file not found. Please ensure 'rf_model.pkl' is in your repository.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def standardize_age(age):
    """Apply standardization for Age feature"""
    age_mean = 40.0  # Approximate - adjust based on your training data
    age_std = 14.0   # Approximate - adjust based on your training data
    return (age - age_mean) / age_std

def create_feature_template():
    """Create template with all features"""
    feature_template = {
        'Age': 0,
        'Gender_M': 0,
        'Smoking_Yes': 0,
        'Hx Smoking_Yes': 0,
        'Hx Radiothreapy_Yes': 0,
        'Thyroid Function_Clinical Hypothyroidism': 0,
        'Thyroid Function_Euthyroid': 0,
        'Thyroid Function_Subclinical Hyperthyroidism': 0,
        'Thyroid Function_Subclinical Hypothyroidism': 0,
        'Physical Examination_Multinodular goiter': 0,
        'Physical Examination_Normal': 0,
        'Physical Examination_Single nodular goiter-left': 0,
        'Physical Examination_Single nodular goiter-right': 0,
        'Adenopathy_Extensive': 0,
        'Adenopathy_Left': 0,
        'Adenopathy_No': 0,
        'Adenopathy_Posterior': 0,
        'Adenopathy_Right': 0,
        'Pathology_Hurthel cell': 0,
        'Pathology_Micropapillary': 0,
        'Pathology_Papillary': 0,
        'Focality_Uni-Focal': 0,
        'Risk_Intermediate': 0,
        'Risk_Low': 0,
        'T_T1b': 0,
        'T_T2': 0,
        'T_T3a': 0,
        'T_T3b': 0,
        'T_T4a': 0,
        'T_T4b': 0,
        'N_N1a': 0,
        'N_N1b': 0,
        'M_M1': 0,
        'Stage_II': 0,
        'Stage_III': 0,
        'Stage_IVA': 0,
        'Stage_IVB': 0,
        'Response_Excellent': 0,
        'Response_Indeterminate': 0,
        'Response_Structural Incomplete': 0
    }
    return feature_template

def prepare_input_data(user_inputs):
    """Convert user inputs to model format with error handling"""
    try:
        features = create_feature_template()
        
        # Set Age (standardized)
        features['Age'] = standardize_age(user_inputs['age'])
        
        # Binary features
        features['Gender_M'] = 1 if user_inputs['gender'] == "Male" else 0
        features['Smoking_Yes'] = 1 if user_inputs['smoking'] == "Yes" else 0
        features['Hx Smoking_Yes'] = 1 if user_inputs['hx_smoking'] == "Yes" else 0
        features['Hx Radiothreapy_Yes'] = 1 if user_inputs['hx_radiotherapy'] == "Yes" else 0
        features['Focality_Uni-Focal'] = 1 if user_inputs['focality'] == "Uni-Focal" else 0
        features['M_M1'] = 1 if user_inputs['m_stage'] == "M1" else 0
        
        # One-hot encoded features with reference categories
        # Thyroid Function (Clinical Hyperthyroidism is reference)
        if user_inputs['thyroid_function'] != "Clinical Hyperthyroidism":
            thyroid_mapping = {
                "Clinical Hypothyroidism": "Thyroid Function_Clinical Hypothyroidism",
                "Euthyroid": "Thyroid Function_Euthyroid",
                "Subclinical Hyperthyroidism": "Thyroid Function_Subclinical Hyperthyroidism",
                "Subclinical Hypothyroidism": "Thyroid Function_Subclinical Hypothyroidism"
            }
            if user_inputs['thyroid_function'] in thyroid_mapping:
                features[thyroid_mapping[user_inputs['thyroid_function']]] = 1
        
        # Physical Examination (Diffuse goiter is reference)
        if user_inputs['physical_exam'] != "Diffuse goiter":
            physical_mapping = {
                "Multinodular goiter": "Physical Examination_Multinodular goiter",
                "Normal": "Physical Examination_Normal",
                "Single nodular goiter-left": "Physical Examination_Single nodular goiter-left", 
                "Single nodular goiter-right": "Physical Examination_Single nodular goiter-right"
            }
            if user_inputs['physical_exam'] in physical_mapping:
                features[physical_mapping[user_inputs['physical_exam']]] = 1
        
        # Adenopathy (Bilateral is reference)
        if user_inputs['adenopathy'] != "Bilateral":
            adenopathy_mapping = {
                "Extensive": "Adenopathy_Extensive",
                "Left": "Adenopathy_Left",
                "No": "Adenopathy_No",
                "Posterior": "Adenopathy_Posterior", 
                "Right": "Adenopathy_Right"
            }
            if user_inputs['adenopathy'] in adenopathy_mapping:
                features[adenopathy_mapping[user_inputs['adenopathy']]] = 1
        
        # Pathology (Follicular is reference)
        if user_inputs['pathology'] != "Follicular":
            pathology_mapping = {
                "Hurthel cell": "Pathology_Hurthel cell",
                "Micropapillary": "Pathology_Micropapillary", 
                "Papillary": "Pathology_Papillary"
            }
            if user_inputs['pathology'] in pathology_mapping:
                features[pathology_mapping[user_inputs['pathology']]] = 1
        
        # Risk (High is reference)
        if user_inputs['risk'] != "High":
            risk_mapping = {
                "Intermediate": "Risk_Intermediate",
                "Low": "Risk_Low"
            }
            if user_inputs['risk'] in risk_mapping:
                features[risk_mapping[user_inputs['risk']]] = 1
        
        # T Stage (T1a is reference)
        if user_inputs['t_stage'] != "T1a":
            t_mapping = {
                "T1b": "T_T1b", "T2": "T_T2", "T3a": "T_T3a", 
                "T3b": "T_T3b", "T4a": "T_T4a", "T4b": "T_T4b"
            }
            if user_inputs['t_stage'] in t_mapping:
                features[t_mapping[user_inputs['t_stage']]] = 1
        
        # N Stage (N0 is reference)
        if user_inputs['n_stage'] != "N0":
            n_mapping = {"N1a": "N_N1a", "N1b": "N_N1b"}
            if user_inputs['n_stage'] in n_mapping:
                features[n_mapping[user_inputs['n_stage']]] = 1
        
        # Stage (I is reference)
        if user_inputs['stage'] != "I":
            stage_mapping = {
                "II": "Stage_II", "III": "Stage_III", 
                "IVA": "Stage_IVA", "IVB": "Stage_IVB"
            }
            if user_inputs['stage'] in stage_mapping:
                features[stage_mapping[user_inputs['stage']]] = 1
        
        # Response (Biochemical Incomplete is reference)
        if user_inputs['response'] != "Biochemical Incomplete":
            response_mapping = {
                "Excellent": "Response_Excellent",
                "Indeterminate": "Response_Indeterminate", 
                "Structural Incomplete": "Response_Structural Incomplete"
            }
            if user_inputs['response'] in response_mapping:
                features[response_mapping[user_inputs['response']]] = 1
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure correct column order
        expected_columns = list(create_feature_template().keys())
        feature_df = feature_df[expected_columns]
        
        return feature_df
        
    except Exception as e:
        st.error(f"❌ Error preparing input data: {str(e)}")
        st.stop()
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
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None
    return None, None

if app_mode == "Single Prediction":
    st.header("Single Patient Prediction")
    
    # Patient Information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Female", "Male"])
        smoking = st.selectbox("Current Smoking", ["No", "Yes"])
        hx_smoking = st.selectbox("History of Smoking", ["No", "Yes"])
        hx_radiotherapy = st.selectbox("History of Radiotherapy", ["No", "Yes"])
    
    with col2:
        thyroid_function = st.selectbox("Thyroid Function",
            ["Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Euthyroid", 
             "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"])
        
        physical_exam = st.selectbox("Physical Examination",
            ["Diffuse goiter", "Multinodular goiter", "Normal", 
             "Single nodular goiter-left", "Single nodular goiter-right"])
        
        adenopathy = st.selectbox("Adenopathy",
            ["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"])
    
    with col3:
        pathology = st.selectbox("Pathology", 
            ["Follicular", "Hurthel cell", "Micropapillary", "Papillary"])
        
        focality = st.selectbox("Focality", ["Multi-Focal", "Uni-Focal"])
        risk = st.selectbox("Risk Category", ["High", "Intermediate", "Low"])
    
    # Clinical Staging
    st.subheader("Clinical Staging")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        t_stage = st.selectbox("T Stage", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
        n_stage = st.selectbox("N Stage", ["N0", "N1a", "N1b"])
    
    with col5:
        m_stage = st.selectbox("M Stage", ["M0", "M1"])
        stage = st.selectbox("Overall Stage", ["I", "II", "III", "IVA", "IVB"])
    
    with col6:
        response = st.selectbox("Treatment Response", 
            ["Biochemical Incomplete", "Excellent", "Indeterminate", "Structural Incomplete"])
    
    # Prediction button
    if st.button("Predict Recurrence Risk", type="primary"):
        with st.spinner('Analyzing patient data...'):
            user_inputs = {
                'age': age,
                'gender': gender,
                'smoking': smoking,
                'hx_smoking': hx_smoking,
                'hx_radiotherapy': hx_radiotherapy,
                'thyroid_function': thyroid_function,
                'physical_exam': physical_exam,
                'adenopathy': adenopathy,
                'pathology': pathology,
                'focality': focality,
                'risk': risk,
                't_stage': t_stage,
                'n_stage': n_stage,
                'm_stage': m_stage,
                'stage': stage,
                'response': response
            }
            
            input_df = prepare_input_data(user_inputs)
            prediction, probability = predict_recurrence(input_df)
            
            if prediction is not None:
                recurrence_prob = probability[0][1]
                
                # Display results
                st.subheader("Prediction Results")
                
                if prediction[0] == 1:
                    st.error(f"🚨 High Risk of Recurrence: {recurrence_prob:.1%}")
                else:
                    st.success(f"✅ Low Risk of Recurrence: {recurrence_prob:.1%}")
                
                # Probability visualization
                st.subheader("Recurrence Probability")
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [recurrence_prob], 
                       color='red' if recurrence_prob > 0.5 else 'orange' if recurrence_prob > 0.3 else 'lightgreen')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability of Recurrence')
                ax.set_yticks([])
                ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7)
                ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7)
                ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
                st.pyplot(fig)

elif app_mode == "Model Info":
    st.header("Model Information")
    
    st.subheader("Random Forest Model Details")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier
    - **Target**: Thyroid Cancer Recurrence (Binary Classification)
    - **Features**: 41 clinical and pathological variables
    - **Dataset**: Thyroid cancer patient records
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

# Load model at startup
if 'model_loaded' not in st.session_state:
    load_model()
    st.session_state.model_loaded = True
