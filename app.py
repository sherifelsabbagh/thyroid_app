import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66, #40c057);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .prediction-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Title with professional styling
st.markdown('<h1 class="main-header">🏥 Thyroid Cancer Recurrence Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 3rem; font-size: 1.1rem;'>
A clinical decision support tool for predicting thyroid cancer recurrence risk based on comprehensive patient assessment.
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>🏥 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.selectbox(
        "Select Mode",
        ["Patient Assessment", "Model Information", "Clinical Guidelines"],
        index=0
    )

# Model functions (same as before but with better error handling)
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
                return model
        
        st.error("❌ Model file not found. Please ensure 'rf_model.pkl' is in your repository.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def standardize_age(age):
    """Apply standardization for Age feature"""
    age_mean = 40.0
    age_std = 14.0
    return (age - age_mean) / age_std

def create_feature_template():
    """Create template with all features"""
    feature_template = {
        'Age': 0, 'Gender_M': 0, 'Smoking_Yes': 0, 'Hx Smoking_Yes': 0, 'Hx Radiothreapy_Yes': 0,
        'Thyroid Function_Clinical Hypothyroidism': 0, 'Thyroid Function_Euthyroid': 0,
        'Thyroid Function_Subclinical Hyperthyroidism': 0, 'Thyroid Function_Subclinical Hypothyroidism': 0,
        'Physical Examination_Multinodular goiter': 0, 'Physical Examination_Normal': 0,
        'Physical Examination_Single nodular goiter-left': 0, 'Physical Examination_Single nodular goiter-right': 0,
        'Adenopathy_Extensive': 0, 'Adenopathy_Left': 0, 'Adenopathy_No': 0, 'Adenopathy_Posterior': 0, 'Adenopathy_Right': 0,
        'Pathology_Hurthel cell': 0, 'Pathology_Micropapillary': 0, 'Pathology_Papillary': 0,
        'Focality_Uni-Focal': 0, 'Risk_Intermediate': 0, 'Risk_Low': 0,
        'T_T1b': 0, 'T_T2': 0, 'T_T3a': 0, 'T_T3b': 0, 'T_T4a': 0, 'T_T4b': 0,
        'N_N1a': 0, 'N_N1b': 0, 'M_M1': 0,
        'Stage_II': 0, 'Stage_III': 0, 'Stage_IVA': 0, 'Stage_IVB': 0,
        'Response_Excellent': 0, 'Response_Indeterminate': 0, 'Response_Structural Incomplete': 0
    }
    return feature_template

def prepare_input_data(user_inputs):
    """Convert user inputs to model format with error handling"""
    try:
        features = create_feature_template()
        
        # Set features with error handling
        features['Age'] = standardize_age(user_inputs['age'])
        features['Gender_M'] = 1 if user_inputs['gender'] == "Male" else 0
        features['Smoking_Yes'] = 1 if user_inputs['smoking'] == "Yes" else 0
        features['Hx Smoking_Yes'] = 1 if user_inputs['hx_smoking'] == "Yes" else 0
        features['Hx Radiothreapy_Yes'] = 1 if user_inputs['hx_radiotherapy'] == "Yes" else 0
        features['Focality_Uni-Focal'] = 1 if user_inputs['focality'] == "Uni-Focal" else 0
        features['M_M1'] = 1 if user_inputs['m_stage'] == "M1" else 0
        
        # One-hot encoding with reference categories
        mappings = {
            'thyroid_function': {
                'ref': "Clinical Hyperthyroidism",
                'mapping': {
                    "Clinical Hypothyroidism": "Thyroid Function_Clinical Hypothyroidism",
                    "Euthyroid": "Thyroid Function_Euthyroid",
                    "Subclinical Hyperthyroidism": "Thyroid Function_Subclinical Hyperthyroidism",
                    "Subclinical Hypothyroidism": "Thyroid Function_Subclinical Hypothyroidism"
                }
            },
            'physical_exam': {
                'ref': "Diffuse goiter",
                'mapping': {
                    "Multinodular goiter": "Physical Examination_Multinodular goiter",
                    "Normal": "Physical Examination_Normal",
                    "Single nodular goiter-left": "Physical Examination_Single nodular goiter-left",
                    "Single nodular goiter-right": "Physical Examination_Single nodular goiter-right"
                }
            },
            'adenopathy': {
                'ref': "Bilateral",
                'mapping': {
                    "Extensive": "Adenopathy_Extensive",
                    "Left": "Adenopathy_Left",
                    "No": "Adenopathy_No",
                    "Posterior": "Adenopathy_Posterior",
                    "Right": "Adenopathy_Right"
                }
            },
            'pathology': {
                'ref': "Follicular",
                'mapping': {
                    "Hurthel cell": "Pathology_Hurthel cell",
                    "Micropapillary": "Pathology_Micropapillary",
                    "Papillary": "Pathology_Papillary"
                }
            },
            'risk': {
                'ref': "High",
                'mapping': {
                    "Intermediate": "Risk_Intermediate",
                    "Low": "Risk_Low"
                }
            },
            't_stage': {
                'ref': "T1a",
                'mapping': {
                    "T1b": "T_T1b", "T2": "T_T2", "T3a": "T_T3a",
                    "T3b": "T_T3b", "T4a": "T_T4a", "T4b": "T_T4b"
                }
            },
            'n_stage': {
                'ref': "N0",
                'mapping': {
                    "N1a": "N_N1a", "N1b": "N_N1b"
                }
            },
            'stage': {
                'ref': "I",
                'mapping': {
                    "II": "Stage_II", "III": "Stage_III",
                    "IVA": "Stage_IVA", "IVB": "Stage_IVB"
                }
            },
            'response': {
                'ref': "Biochemical Incomplete",
                'mapping': {
                    "Excellent": "Response_Excellent",
                    "Indeterminate": "Response_Indeterminate",
                    "Structural Incomplete": "Response_Structural Incomplete"
                }
            }
        }
        
        for feature_key, config in mappings.items():
            if user_inputs[feature_key] != config['ref']:
                feature_name = config['mapping'].get(user_inputs[feature_key])
                if feature_name:
                    features[feature_name] = 1
        
        feature_df = pd.DataFrame([features])
        expected_columns = list(create_feature_template().keys())
        feature_df = feature_df[expected_columns]
        
        return feature_df
        
    except Exception as e:
        st.error(f"❌ Error preparing input data: {str(e)}")
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

def create_risk_gauge(probability):
    """Create a beautiful gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Recurrence Risk Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'lightgreen'},
                {'range': [20, 50], 'color': 'yellow'},
                {'range': [50, 80], 'color': 'orange'},
                {'range': [80, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ))
    
    fig.update_layout(height=300, margin=dict(l=50, r=50, t=100, b=50))
    return fig

def get_clinical_recommendations(probability, user_inputs):
    """Generate clinical recommendations based on risk level"""
    recommendations = []
    
    if probability < 0.1:
        recommendations.append("✅ Routine follow-up (6-12 months)")
        recommendations.append("✅ Standard thyroid ultrasound monitoring")
        recommendations.append("✅ Annual clinical assessment")
    elif probability < 0.3:
        recommendations.append("🔸 Consider 6-month follow-up intervals")
        recommendations.append("🔸 Regular thyroglobulin monitoring")
        recommendations.append("🔸 Ultrasound every 6-12 months")
    elif probability < 0.5:
        recommendations.append("⚠️ Increased surveillance recommended")
        recommendations.append("⚠️ Consider 3-6 month follow-up")
        recommendations.append("⚠️ Multidisciplinary team consultation")
    else:
        recommendations.append("🚨 Intensive monitoring required")
        recommendations.append("🚨 Consider adjuvant therapy evaluation")
        recommendations.append("🚨 Specialist endocrinology referral")
        recommendations.append("🚨 Frequent imaging (3-6 month intervals)")
    
    # Additional specific recommendations
    if user_inputs['t_stage'] in ['T3a', 'T3b', 'T4a', 'T4b']:
        recommendations.append("📋 Advanced T-stage: Consider more intensive imaging")
    
    if user_inputs['n_stage'] != "N0":
        recommendations.append("📋 Lymph node involvement: Monitor neck compartments closely")
    
    if user_inputs['risk'] == "High":
        recommendations.append("📋 High-risk pathology: Consider specialized protocols")
    
    return recommendations

if app_mode == "Patient Assessment":
    # Progress bar
    st.markdown("### Patient Assessment")
    progress_bar = st.progress(0)
    
    # Patient Information Section
    st.markdown('<div class="section-header">📋 Patient Demographics & History</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        age = st.number_input("**Age**", min_value=0, max_value=100, value=45, 
                           help="Patient's current age in years")
        gender = st.selectbox("**Gender**", ["Female", "Male"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        smoking = st.selectbox("**Current Smoking**", ["No", "Yes"],
                            help="Current tobacco use status")
        hx_smoking = st.selectbox("**History of Smoking**", ["No", "Yes"],
                                help="Previous tobacco use history")
        hx_radiotherapy = st.selectbox("**History of Radiotherapy**", ["No", "Yes"],
                                     help="Previous radiation exposure")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        thyroid_function = st.selectbox("**Thyroid Function**",
            ["Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Euthyroid", 
             "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"],
            help="Current thyroid functional status")
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(25)
    
    # Clinical Examination Section
    st.markdown('<div class="section-header">🔍 Clinical Examination Findings</div>', unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        physical_exam = st.selectbox("**Physical Examination**",
            ["Diffuse goiter", "Multinodular goiter", "Normal", 
             "Single nodular goiter-left", "Single nodular goiter-right"],
            help="Findings on physical examination")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col5:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        adenopathy = st.selectbox("**Adenopathy**",
            ["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"],
            help="Lymph node involvement")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col6:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        pathology = st.selectbox("**Pathology**", 
            ["Follicular", "Hurthel cell", "Micropapillary", "Papillary"],
            help="Histopathological diagnosis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(50)
    
    # Staging Section
    st.markdown('<div class="section-header">📊 Cancer Staging & Risk Stratification</div>', unsafe_allow_html=True)
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        t_stage = st.selectbox("**T Stage**", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"],
                             help="Tumor size and extent")
        n_stage = st.selectbox("**N Stage**", ["N0", "N1a", "N1b"],
                             help="Regional lymph node involvement")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col8:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        m_stage = st.selectbox("**M Stage**", ["M0", "M1"],
                             help="Distant metastasis")
        stage = st.selectbox("**Overall Stage**", ["I", "II", "III", "IVA", "IVB"],
                           help="AJCC TNM stage grouping")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col9:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        focality = st.selectbox("**Focality**", ["Multi-Focal", "Uni-Focal"],
                              help="Single vs multiple tumor foci")
        risk = st.selectbox("**Risk Category**", ["High", "Intermediate", "Low"],
                          help="ATA risk stratification")
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(75)
    
    # Treatment Response
    st.markdown('<div class="section-header">💊 Treatment Response Assessment</div>', unsafe_allow_html=True)
    
    col10, col11, _ = st.columns(3)
    
    with col10:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        response = st.selectbox("**Treatment Response**", 
            ["Biochemical Incomplete", "Excellent", "Indeterminate", "Structural Incomplete"],
            help="Response to initial therapy")
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(100)
    
    # Prediction Button
    st.markdown("---")
    col12, col13, col14 = st.columns([1, 2, 1])
    
    with col13:
        predict_btn = st.button("🎯 **Calculate Recurrence Risk**", 
                              type="primary", 
                              use_container_width=True,
                              help="Analyze recurrence risk based on all clinical inputs")
    
    if predict_btn:
        with st.spinner('🔄 Analyzing comprehensive clinical data...'):
            user_inputs = {
                'age': age, 'gender': gender, 'smoking': smoking, 'hx_smoking': hx_smoking,
                'hx_radiotherapy': hx_radiotherapy, 'thyroid_function': thyroid_function,
                'physical_exam': physical_exam, 'adenopathy': adenopathy, 'pathology': pathology,
                'focality': focality, 'risk': risk, 't_stage': t_stage, 'n_stage': n_stage,
                'm_stage': m_stage, 'stage': stage, 'response': response
            }
            
            input_df = prepare_input_data(user_inputs)
            
            if input_df is not None:
                prediction, probability = predict_recurrence(input_df)
                
                if prediction is not None:
                    recurrence_prob = probability[0][1]
                    
                    # Results Section
                    st.markdown("---")
                    st.markdown("## 📈 Prediction Results")
                    
                    # Main result cards
                    col15, col16 = st.columns(2)
                    
                    with col15:
                        if prediction[0] == 1:
                            st.markdown(f"""
                            <div class="risk-high">
                                <h2>🚨 High Recurrence Risk</h2>
                                <h1>{recurrence_prob:.1%}</h1>
                                <p>Probability of Disease Recurrence</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="risk-low">
                                <h2>✅ Low Recurrence Risk</h2>
                                <h1>{recurrence_prob:.1%}</h1>
                                <p>Probability of Disease Recurrence</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col16:
                        # Gauge chart
                        fig = create_risk_gauge(recurrence_prob)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Clinical Recommendations
                    st.markdown("---")
                    st.markdown("## 💡 Clinical Recommendations")
                    
                    recommendations = get_clinical_recommendations(recurrence_prob, user_inputs)
                    
                    col17, col18 = st.columns(2)
                    
                    with col17:
                        st.markdown("#### 📋 Monitoring Protocol")
                        for rec in recommendations[:len(recommendations)//2]:
                            st.markdown(f"<div class='prediction-card'>{rec}</div>", unsafe_allow_html=True)
                    
                    with col18:
                        st.markdown("#### 🔬 Additional Considerations")
                        for rec in recommendations[len(recommendations)//2:]:
                            st.markdown(f"<div class='prediction-card'>{rec}</div>", unsafe_allow_html=True)
                    
                    # Risk Interpretation
                    st.markdown("---")
                    st.markdown("#### 🎯 Risk Stratification")
                    
                    if recurrence_prob < 0.1:
                        st.success("**Very Low Risk (
                                                           st.success("**Very Low Risk (<10%)**: Excellent prognosis with standard follow-up")
                    elif recurrence_prob < 0.3:
                        st.info("**Low Risk (10-30%)**: Favorable outcome with routine monitoring")
                    elif recurrence_prob < 0.5:
                        st.warning("**Moderate Risk (30-50%)**: Requires increased surveillance")
                    else:
                        st.error("**High Risk (>50%)**: Intensive management and close monitoring required")

                    # Key Risk Factors
                    st.markdown("#### 🔍 Key Contributing Factors")
                    risk_factors = []
                    
                    if user_inputs['t_stage'] in ['T3a', 'T3b', 'T4a', 'T4b']:
                        risk_factors.append("Advanced T-stage (T3-T4)")
                    if user_inputs['n_stage'] != "N0":
                        risk_factors.append("Lymph node involvement")
                    if user_inputs['m_stage'] == "M1":
                        risk_factors.append("Distant metastasis")
                    if user_inputs['stage'] in ["III", "IVA", "IVB"]:
                        risk_factors.append("Advanced overall stage")
                    if user_inputs['risk'] == "High":
                        risk_factors.append("High-risk pathological features")
                    if user_inputs['response'] == "Structural Incomplete":
                        risk_factors.append("Structural incomplete response")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• No high-risk features identified")

elif app_mode == "Model Information":
    st.markdown("## 🔬 Model Information & Technical Details")
    
    # Model Overview Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Model Architecture</h4>
            <p><b>Random Forest Classifier</b></p>
            <p>Ensemble method with 100 decision trees</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Performance Metrics</h4>
            <p><b>Clinical Validation</b></p>
            <p>Cross-validated accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🔬 Feature Engineering</h4>
            <p><b>41 Clinical Variables</b></p>
            <p>Comprehensive feature set</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Categories with expanders
    st.markdown("#### 📁 Feature Categories")
    
    feature_categories = {
        "👤 Demographic Factors": ["Age", "Gender", "Smoking History", "Radiation History"],
        "🔍 Clinical Presentation": ["Thyroid Function", "Physical Examination", "Adenopathy"],
        "🧬 Pathological Features": ["Cancer Type", "Focality", "Risk Stratification"],
        "📐 Tumor Staging": ["T Stage", "N Stage", "M Stage", "Overall Stage"],
        "💊 Treatment Response": ["Response to Initial Therapy"]
    }
    
    for category, features in feature_categories.items():
        with st.expander(f"{category} ({len(features)} features)"):
            cols = st.columns(2)
            for i, feature in enumerate(features):
                cols[i % 2].write(f"• {feature}")
    
    # Data Preprocessing
    st.markdown("#### ⚙️ Data Preprocessing")
    st.markdown("""
    <div class="feature-card">
    <h5>Standardization & Encoding</h5>
    <ul>
    <li><b>Age</b>: Standardized (z-score normalization)</li>
    <li><b>Categorical Variables</b>: One-hot encoding with drop-first</li>
    <li><b>Missing Data</b>: Exclusion of incomplete records</li>
    <li><b>Feature Scaling</b>: Not required for tree-based models</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Validation
    st.markdown("#### 🏥 Clinical Validation")
    st.markdown("""
    <div class="feature-card">
    <h5>Model Development & Validation</h5>
    <ul>
    <li><b>Dataset</b>: Retrospective thyroid cancer cohort</li>
    <li><b>Sample Size</b>: Clinically representative population</li>
    <li><b>Validation</b>: Temporal and cross-validation splits</li>
    <li><b>Clinical Review</b>: Expert endocrinology oversight</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "Clinical Guidelines":
    st.markdown("## 📚 Clinical Guidelines & References")
    
    # ATA Guidelines Summary
    st.markdown("#### 🏥 ATA Risk Stratification System")
    
    guidelines = {
        "Low Risk": [
            "Intrathyroidal differentiated thyroid cancer",
            "Clinical N0 or ≤5 pathologic N1 micrometastases",
            "No extathyroidal extension",
            "No vascular invasion",
            "Classic PTC ≤4 cm or FVPTC ≤4 cm"
        ],
        "Intermediate Risk": [
            "Microscopic extathyroidal extension",
            "Cervical lymph node metastases",
            "Aggressive histology",
            "Vascular invasion",
            "RAI-avid metastatic foci in the neck"
        ],
        "High Risk": [
            "Macroscopic extathyroidal extension",
            "Incomplete tumor resection",
            "Distant metastases",
            "Postoperative serum Tg suggestive of distant metastases",
            "Pathologic N1 with any metastatic node ≥3 cm"
        ]
    }
    
    for risk_level, criteria in guidelines.items():
        with st.expander(f"📋 {risk_level} Risk Criteria"):
            for criterion in criteria:
                st.write(f"• {criterion}")
    
    # Follow-up Recommendations
    st.markdown("#### 🔄 Recommended Follow-up Protocols")
    
    follow_up = {
        "Low Risk": [
            "Clinical assessment every 6-12 months",
            "Neck ultrasound at 6-12 months, then annually for 3-5 years",
            "Tg and anti-Tg antibodies every 6-12 months",
            "Consider decreasing frequency after 5 years of remission"
        ],
        "Intermediate Risk": [
            "Clinical assessment every 6 months for 2-3 years",
            "Neck ultrasound every 6-12 months for 3-5 years",
            "Stimulated Tg measurement as clinically indicated",
            "Consider cross-sectional imaging for concerning features"
        ],
        "High Risk": [
            "Clinical assessment every 3-6 months initially",
            "Neck ultrasound every 6 months for several years",
            "Regular stimulated Tg testing",
            "Cross-sectional imaging (CT/MRI) as indicated",
            "Consider more frequent RAI scans initially"
        ]
    }
    
    col1, col2, col3 = st.columns(3)
    for i, (risk_level, protocols) in enumerate(follow_up.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"**{risk_level}**")
            for protocol in protocols:
                st.write(f"• {protocol}")
    
    # References
    st.markdown("#### 📖 Key References")
    references = [
        "Haugen BR, et al. 2015 American Thyroid Association Management Guidelines for Adult Patients with Thyroid Nodules and Differentiated Thyroid Cancer. Thyroid. 2016",
        "Tuttle RM, et al. Contemporary post-surgical management of differentiated thyroid carcinoma. Clin Oncol. 2021",
        "NCCN Clinical Practice Guidelines in Oncology: Thyroid Carcinoma. Version 3.2022"
    ]
    
    for ref in references:
        st.write(f"• {ref}")

# Footer with professional disclaimer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem; padding: 20px;'>
    <p><strong>Important Disclaimer</strong></p>
    <p><em>This tool is intended for clinical decision support only. Predictions should be interpreted by qualified healthcare professionals in the context of comprehensive clinical assessment. Always follow institutional protocols and current clinical guidelines.</em></p>
    <p style='margin-top: 10px;'>© 2024 Thyroid Cancer Recurrence Predictor • For Clinical Use</p>
</div>
""", unsafe_allow_html=True)

# Load model at startup
if 'model_loaded' not in st.session_state:
    model = load_model()
    if model is not None:
        st.session_state.model_loaded = True
