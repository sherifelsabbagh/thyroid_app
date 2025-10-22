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
Fill out the form below to get predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["Single Prediction", "Model Info"])

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Try different possible model file locations
        model_paths = [
            'rf_model.pkl',
            'model/rf_model.pkl',
            './rf_model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                st.sidebar.success(f"Model loaded successfully from {path}")
                return model
        
        st.error("Model file not found. Please ensure 'rf_model.pkl' is in your repository.")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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

def standardize_age(age):
    """Apply the same standardization you used during training"""
    age_mean = 40.0  # Replace with actual mean from your training
    age_std = 14.0   # Replace with actual std from your training
    standardized_age = (age - age_mean) / age_std
    return standardized_age

def create_feature_template():
    """Create a template with all features in the exact order used during training"""
    feature_template = {
        'Age': 0,  # This will be standardized
        
        # Binary features
        'Gender_M': 0,
        'Smoking_Yes': 0,
        'Hx Smoking_Yes': 0,
        'Hx Radiothreapy_Yes': 0,
        'Focality_Uni-Focal': 0,
        'M_M1': 0,
        
        # Thyroid Function (one-hot) - Clinical Hyperthyroidism is reference (dropped)
        'Thyroid Function_Clinical Hypothyroidism': 0,
        'Thyroid Function_Euthyroid': 0,
        'Thyroid Function_Subclinical Hyperthyroidism': 0,
        'Thyroid Function_Subclinical Hypothyroidism': 0,
        
        # Physical Examination (one-hot) - Diffuse goiter is reference (dropped)
        'Physical Examination_Multinodular goiter': 0,
        'Physical Examination_Normal': 0,
        'Physical Examination_Single nodular goiter-left': 0,
        'Physical Examination_Single nodular goiter-right': 0,
        
        # Adenopathy (one-hot) - Bilateral is reference (dropped)
        'Adenopathy_Extensive': 0,
        'Adenopathy_Left': 0,
        'Adenopathy_No': 0,
        'Adenopathy_Posterior': 0,
        'Adenopathy_Right': 0,
        
        # Pathology (one-hot) - Follicular is reference (dropped)
        'Pathology_Hurthel cell': 0,
        'Pathology_Micropapillary': 0,
        'Pathology_Papillary': 0,
        
        # Risk (one-hot) - High is reference (dropped)
        'Risk_Intermediate': 0,
        'Risk_Low': 0,
        
        # T Stage (one-hot) - T1a is reference (dropped)
        'T_T1b': 0,
        'T_T2': 0,
        'T_T3a': 0,
        'T_T3b': 0,
        'T_T4a': 0,
        'T_T4b': 0,
        
        # N Stage (one-hot) - N0 is reference (dropped)
        'N_N1a': 0,
        'N_N1b': 0,
        
        # Stage (one-hot) - I is reference (dropped)
        'Stage_II': 0,
        'Stage_III': 0,
        'Stage_IVA': 0,
        'Stage_IVB': 0,
        
        # Response (one-hot) - Biochemical Incomplete is reference (dropped)
        'Response_Excellent': 0,
        'Response_Indeterminate': 0,
        'Response_Structural Incomplete': 0
    }
    return feature_template

def prepare_input_data(user_inputs):
    """Convert user inputs to the exact format expected by the model"""
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
    
    # CRITICAL: One-hot encoding with reference categories (first alphabetical - dropped during training)
    
    # Thyroid Function (Clinical Hyperthyroidism is reference - dropped)
    if user_inputs['thyroid_function'] != "Clinical Hyperthyroidism":
        thyroid_mapping = {
            "Clinical Hypothyroidism": "Thyroid Function_Clinical Hypothyroidism",
            "Euthyroid": "Thyroid Function_Euthyroid",
            "Subclinical Hyperthyroidism": "Thyroid Function_Subclinical Hyperthyroidism",
            "Subclinical Hypothyroidism": "Thyroid Function_Subclinical Hypothyroidism"
        }
        features[thyroid_mapping[user_inputs['thyroid_function']]] = 1
    
    # Physical Examination (Diffuse goiter is reference - dropped)
    if user_inputs['physical_exam'] != "Diffuse goiter":
        physical_mapping = {
            "Multinodular goiter": "Physical Examination_Multinodular goiter",
            "Normal": "Physical Examination_Normal",
            "Single nodular goiter-left": "Physical Examination_Single nodular goiter-left", 
            "Single nodular goiter-right": "Physical Examination_Single nodular goiter-right"
        }
        features[physical_mapping[user_inputs['physical_exam']]] = 1
    
    # Adenopathy (Bilateral is reference - dropped)
    if user_inputs['adenopathy'] != "Bilateral":
        adenopathy_mapping = {
            "Extensive": "Adenopathy_Extensive",
            "Left": "Adenopathy_Left",
            "No": "Adenopathy_No",
            "Posterior": "Adenopathy_Posterior", 
            "Right": "Adenopathy_Right"
        }
        features[adenopathy_mapping[user_inputs['adenopathy']]] = 1
    
    # Pathology (Follicular is reference - dropped)
    if user_inputs['pathology'] != "Follicular":
        pathology_mapping = {
            "Hurthel cell": "Pathology_Hurthel cell",
            "Micropapillary": "Pathology_Micropapillary", 
            "Papillary": "Pathology_Papillary"
        }
        features[pathology_mapping[user_inputs['pathology']]] = 1
    
    # Risk (High is reference - dropped)
    if user_inputs['risk'] != "High":
        risk_mapping = {
            "Intermediate": "Risk_Intermediate",
            "Low": "Risk_Low"
        }
        features[risk_mapping[user_inputs['risk']]] = 1
    
    # T Stage (T1a is reference - dropped)
    if user_inputs['t_stage'] != "T1a":
        t_mapping = {
            "T1b": "T_T1b", "T2": "T_T2", "T3a": "T_T3a", 
            "T3b": "T_T3b", "T4a": "T_T4a", "T4b": "T_T4b"
        }
        features[t_mapping[user_inputs['t_stage']]] = 1
    
    # N Stage (N0 is reference - dropped)
    if user_inputs['n_stage'] != "N0":
        n_mapping = {"N1a": "N_N1a", "N1b": "N_N1b"}
        features[n_mapping[user_inputs['n_stage']]] = 1
    
    # Stage (I is reference - dropped)
    if user_inputs['stage'] != "I":
        stage_mapping = {
            "II": "Stage_II", "III": "Stage_III", 
            "IVA": "Stage_IVA", "IVB": "Stage_IVB"
        }
        features[stage_mapping[user_inputs['stage']]] = 1
    
    # Response (Biochemical Incomplete is reference - dropped)
    if user_inputs['response'] != "Biochemical Incomplete":
        response_mapping = {
            "Excellent": "Response_Excellent",
            "Indeterminate": "Response_Indeterminate", 
            "Structural Incomplete": "Response_Structural Incomplete"
        }
        features[response_mapping[user_inputs['response']]] = 1
    
    # Convert to DataFrame with correct column order
    feature_df = pd.DataFrame([features])
    expected_columns = list(create_feature_template().keys())
    feature_df = feature_df[expected_columns]
    
    return feature_df

if app_mode == "Single Prediction":
    st.header("Single Patient Prediction")
    
    st.subheader("Patient Information")
    
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
    
    # Additional clinical features
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
    
    # Debug information
    with st.expander("Debug Information (For Testing)"):
        st.write("This section shows how your inputs are being processed:")
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
        st.json(user_inputs)
        
        # Show feature transformation
        input_df = prepare_input_data(user_inputs)
        st.write("Features sent to model (first 10 columns):")
        st.dataframe(input_df.iloc[:, :10])
    
    # Prediction button
    if st.button("Predict Recurrence Risk", type="primary"):
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
        
        # Prepare input data
        input_df = prepare_input_data(user_inputs)
        
        # Make prediction
        prediction, probability = predict_recurrence(input_df)
        
        if prediction is not None:
            recurrence_prob = probability[0][1]  # Probability of recurrence (class 1)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            col7, col8 = st.columns(2)
            
            with col7:
                if prediction[0] == 1:
                    st.error(f"🚨 **High Risk of Recurrence**")
                    st.metric("Recurrence Probability", f"{recurrence_prob:.1%}")
                else:
                    st.success(f"✅ **Low Risk of Recurrence**")
                    st.metric("Recurrence Probability", f"{recurrence_prob:.1%}")
            
            with col8:
                # Risk level interpretation
                if recurrence_prob < 0.1:
                    risk_level = "Very Low"
                    color = "green"
                elif recurrence_prob < 0.3:
                    risk_level = "Low"
                    color = "lightgreen"
                elif recurrence_prob < 0.5:
                    risk_level = "Moderate"
                    color = "orange"
                else:
                    risk_level = "High"
                    color = "red"
                
                st.info(f"**Risk Level:** :{color}[{risk_level}]")
            
            # Probability gauge
            st.subheader("📊 Recurrence Probability Gauge")
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh([0], [recurrence_prob], 
                   color='red' if recurrence_prob > 0.5 else 'orange' if recurrence_prob > 0.3 else 'lightgreen' if recurrence_prob > 0.1 else 'green')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability of Recurrence')
            ax.set_yticks([])
            ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.7, label='Low Risk')
            ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='High Risk')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
            st.pyplot(fig)

elif app_mode == "Model Info":
    st.header("Model Information")
    
    st.subheader("Random Forest Model Details")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier
    - **Target**: Thyroid Cancer Recurrence (Binary Classification)
    - **Features**: 41 clinical and pathological variables
    - **Dataset**: Thyroid cancer patient records
    - **Preprocessing**: StandardScaler for Age, One-Hot Encoding for categorical variables
    """)
    
    st.subheader("Feature Categories")
    st.markdown("""
    1. **Demographic**: Age, Gender, Smoking history
    2. **Clinical**: Thyroid function, Physical examination findings
    3. **Pathological**: Cancer type, Focality, Risk stratification
    4. **Staging**: TNM staging, Overall stage
    5. **Treatment Response**: Post-treatment assessment
    """)
    
    st.subheader("Reference Categories (Dropped during encoding)")
    st.markdown("""
    The following categories were used as reference (baseline) and dropped during one-hot encoding:
    - **Pathology**: Follicular
    - **Thyroid Function**: Clinical Hyperthyroidism  
    - **Physical Examination**: Diffuse goiter
    - **Adenopathy**: Bilateral
    - **Risk**: High
    - **T Stage**: T1a
    - **N Stage**: N0
    - **Stage**: I
    - **Response**: Biochemical Incomplete
    """)

# Footer
st.markdown("---")
st.markdown("*For clinical use only. Predictions should be interpreted by qualified healthcare professionals.*")

# Load model at startup
if 'model_loaded' not in st.session_state:
    load_model()
    st.session_state.model_loaded = True
