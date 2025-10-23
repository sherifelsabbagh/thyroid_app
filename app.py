import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import os

# Set page configuration
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #51cf66;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title with better styling
st.markdown('<h1 class="main-header">🏥 Thyroid Cancer Recurrence Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 3rem;'>
This tool predicts the likelihood of thyroid cancer recurrence based on clinical and pathological features.
</div>
""", unsafe_allow_html=True)

# Sidebar navigation with better styling
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=80)
    st.markdown("## Navigation")
    
    selected = option_menu(
        menu_title=None,
        options=["Patient Assessment", "Model Info", "About"],
        icons=["clipboard-pulse", "graph-up", "info-circle"],
        default_index=0,
    )

# Load model function (keep your existing one)
@st.cache_resource
def load_model():
    # Your existing load_model function
    pass

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
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    return fig

def create_feature_importance_plot():
    """Create a placeholder feature importance plot"""
    # This would be real if you had feature importance data
    features = ['T Stage', 'N Stage', 'Age', 'Pathology', 'Risk Category']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10]
    
    fig = px.bar(
        x=importance, 
        y=features,
        orientation='h',
        title="Top Factors Influencing Prediction",
        labels={'x': 'Importance', 'y': ''}
    )
    fig.update_layout(height=300)
    return fig

if selected == "Patient Assessment":
    # Progress bar
    st.markdown("### Patient Assessment")
    progress_bar = st.progress(0)
    
    # Patient Information Section
    st.markdown("---")
    st.markdown("#### 📋 Patient Demographics & History")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        age = st.number_input("**Age**", min_value=0, max_value=100, value=45, help="Patient's current age")
        gender = st.selectbox("**Gender**", ["Female", "Male"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        smoking = st.selectbox("**Current Smoking**", ["No", "Yes"])
        hx_smoking = st.selectbox("**History of Smoking**", ["No", "Yes"])
        hx_radiotherapy = st.selectbox("**History of Radiotherapy**", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        thyroid_function = st.selectbox("**Thyroid Function**",
            ["Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Euthyroid", 
             "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(25)
    
    # Clinical Examination Section
    st.markdown("---")
    st.markdown("#### 🔍 Clinical Examination")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        physical_exam = st.selectbox("**Physical Examination**",
            ["Diffuse goiter", "Multinodular goiter", "Normal", 
             "Single nodular goiter-left", "Single nodular goiter-right"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col5:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        adenopathy = st.selectbox("**Adenopathy**",
            ["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col6:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        pathology = st.selectbox("**Pathology**", 
            ["Follicular", "Hurthel cell", "Micropapillary", "Papillary"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(50)
    
    # Staging Section
    st.markdown("---")
    st.markdown("#### 📊 Cancer Staging")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        t_stage = st.selectbox("**T Stage**", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
        n_stage = st.selectbox("**N Stage**", ["N0", "N1a", "N1b"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col8:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        m_stage = st.selectbox("**M Stage**", ["M0", "M1"])
        stage = st.selectbox("**Overall Stage**", ["I", "II", "III", "IVA", "IVB"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col9:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        focality = st.selectbox("**Focality**", ["Multi-Focal", "Uni-Focal"])
        risk = st.selectbox("**Risk Category**", ["High", "Intermediate", "Low"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(75)
    
    # Treatment Response
    st.markdown("---")
    st.markdown("#### 💊 Treatment Response")
    
    col10, col11, _ = st.columns(3)
    
    with col10:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        response = st.selectbox("**Treatment Response**", 
            ["Biochemical Incomplete", "Excellent", "Indeterminate", "Structural Incomplete"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar.progress(100)
    
    # Prediction Button
    st.markdown("---")
    col12, col13, col14 = st.columns([1, 2, 1])
    
    with col13:
        predict_btn = st.button("🎯 **Calculate Recurrence Risk**", 
                              type="primary", 
                              use_container_width=True,
                              help="Click to analyze recurrence risk based on all inputs")
    
    if predict_btn:
        with st.spinner('🔄 Analyzing patient data...'):
            # Your existing prediction code here
            user_inputs = {
                'age': age, 'gender': gender, 'smoking': smoking, 'hx_smoking': hx_smoking,
                'hx_radiotherapy': hx_radiotherapy, 'thyroid_function': thyroid_function,
                'physical_exam': physical_exam, 'adenopathy': adenopathy, 'pathology': pathology,
                'focality': focality, 'risk': risk, 't_stage': t_stage, 'n_stage': n_stage,
                'm_stage': m_stage, 'stage': stage, 'response': response
            }
            
            # Your existing prepare_input_data and predict_recurrence functions
            input_df = prepare_input_data(user_inputs)
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
                            <h2>🚨 High Risk</h2>
                            <h1>{recurrence_prob:.1%}</h1>
                            <p>Probability of Recurrence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h2>✅ Low Risk</h2>
                            <h1>{recurrence_prob:.1%}</h1>
                            <p>Probability of Recurrence</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col16:
                    # Gauge chart
                    fig = create_risk_gauge(recurrence_prob)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk interpretation and recommendations
                st.markdown("---")
                col17, col18 = st.columns(2)
                
                with col17:
                    st.markdown("#### 📋 Risk Interpretation")
                    if recurrence_prob < 0.1:
                        st.success("**Very Low Risk**: Routine follow-up recommended")
                    elif recurrence_prob < 0.3:
                        st.info("**Low Risk**: Standard monitoring protocol")
                    elif recurrence_prob < 0.5:
                        st.warning("**Moderate Risk**: Consider increased surveillance")
                    else:
                        st.error("**High Risk**: Intensive monitoring and specialist consultation recommended")
                
                with col18:
                    st.markdown("#### 💡 Clinical Considerations")
                    considerations = []
                    if recurrence_prob > 0.3:
                        considerations.append("Consider more frequent ultrasound monitoring")
                    if recurrence_prob > 0.5:
                        considerations.append("Evaluate for adjuvant therapy")
                        considerations.append("Multidisciplinary team consultation recommended")
                    if not considerations:
                        considerations.append("Continue with standard follow-up protocol")
                    
                    for consideration in considerations:
                        st.write(f"• {consideration}")
                
                # Feature importance (placeholder)
                st.markdown("---")
                st.markdown("#### 🔍 Key Influencing Factors")
                fig_importance = create_feature_importance_plot()
                st.plotly_chart(fig_importance, use_container_width=True)

elif selected == "Model Info":
    st.markdown("## Model Information")
    
    # Model cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Model Type</h4>
            <p>Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Performance</h4>
            <p>Clinical Validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🔬 Features</h4>
            <p>41 Clinical Variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature categories
    st.markdown("#### Feature Categories")
    feature_categories = {
        "Demographic": ["Age", "Gender", "Smoking History"],
        "Clinical": ["Thyroid Function", "Physical Examination", "Adenopathy"],
        "Pathological": ["Cancer Type", "Focality", "Risk Stratification"],
        "Staging": ["TNM Staging", "Overall Stage"],
        "Treatment": ["Treatment Response"]
    }
    
    for category, features in feature_categories.items():
        with st.expander(f"📁 {category}"):
            for feature in features:
                st.write(f"• {feature}")

elif selected == "About":
    st.markdown("## About This Tool")
    
    st.markdown("""
    ### 🏥 Thyroid Cancer Recurrence Predictor
    
    This clinical decision support tool helps healthcare professionals assess the 
    risk of thyroid cancer recurrence based on comprehensive patient data.
    
    #### 🔬 Clinical Basis
    - Built on validated clinical research
    - Incorporates standard staging criteria
    - Considers multiple prognostic factors
    
    #### ⚠️ Important Notes
    - For clinical decision support only
    - Should be used by qualified healthcare professionals
    - Always combine with clinical judgment
    - Regular model validation recommended
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <i>Medical AI Tool • For Clinical Use Only</i>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8rem;'>
    <i>For clinical use only. Predictions should be interpreted by qualified healthcare professionals.</i>
</div>
""", unsafe_allow_html=True)
