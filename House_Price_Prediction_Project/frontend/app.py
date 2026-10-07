import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Page configuration
st.set_page_config(
    page_title="ValuHome AI | Indian House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Authentication & User Accounts
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {
        'admin': 'admin123',
        'nikhil': 'pass123',
        'demo': 'demo123'
    }

# Custom CSS for Clean Nordic Minimalist Light Theme with Auth Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
        color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header Banner Card */
    .header-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(203, 213, 225, 0.8);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        text-align: center;
    }
    
    .header-title {
        background: linear-gradient(135deg, #312e81 0%, #4338ca 40%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #475569;
        font-size: 1.15rem;
        font-weight: 500;
    }

    /* Authentication Form Card */
    .auth-card {
        max-width: 450px;
        margin: 40px auto;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(20px);
        border: 1px solid #cbd5e1;
        border-radius: 24px;
        padding: 35px;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.08);
    }
    
    .auth-title {
        color: #1e1b4b;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .auth-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.04);
        color: #0f172a;
    }
    
    .glass-card h4 {
        color: #1e1b4b;
        font-weight: 700;
        margin-bottom: 16px;
    }
    
    .price-display-card {
        background: linear-gradient(135deg, #e0e7ff 0%, #eff6ff 50%, #ccfbf1 100%);
        border: 2px solid #a5b4fc;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 35px rgba(67, 56, 202, 0.12);
    }

    .price-value-primary {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #312e81 0%, #0f766e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 6px 0;
        letter-spacing: -1px;
    }

    .price-value-sub {
        font-size: 1.25rem;
        color: #4338ca;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .price-range {
        color: #334155;
        font-size: 1.05rem;
        font-weight: 600;
    }
    
    .metric-badge {
        display: inline-block;
        background: #e0e7ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.6);
        padding: 8px 12px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 10px;
        color: #475569;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4338ca !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

def format_inr(val):
    if val >= 10000000:
        return f"₹{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val/100000:.2f} Lakhs"
    else:
        return f"₹{val:,.0f}"

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(base_dir, '..', 'models'))
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))

@st.cache_resource
def load_model_bundle():
    model_file = os.path.join(models_dir, 'house_price_model.pkl')
    if not os.path.exists(model_file):
        sys.path.append(models_dir)
        from train_model import train_and_evaluate
        train_and_evaluate()
    return joblib.load(model_file)

@st.cache_data
def load_dataset():
    data_path = os.path.join(data_dir, 'housing_data.csv')
    if not os.path.exists(data_path):
        sys.path.append(data_dir)
        from generate_data import generate_housing_dataset
        generate_housing_dataset()
    return pd.read_csv(data_path)

model_bundle = load_model_bundle()
df = load_dataset()

pipeline = model_bundle['model_pipeline']
best_model_name = model_bundle['best_model_name']
metrics = model_bundle['metrics']
best_rmse = metrics[best_model_name]['RMSE']

# Sidebar User Info & Logout Control
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/home.png", width=80)
    st.markdown("### ValuHome AI Portal")
    
    if st.session_state['authenticated']:
        st.success(f"👤 Logged in as **{st.session_state['user']}**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.session_state['user'] = None
            st.rerun()
    else:
        st.warning("🔒 Not Logged In")
        st.info("Log in using the form to access AI valuation tools.")
        
    st.markdown("---")
    st.markdown("**Platform Status:** `Online 🟢`")
    st.markdown(f"**ML Model:** `{best_model_name}`")
    st.markdown(f"**Model R² Score:** `{metrics[best_model_name]['R2']}`")

# Header Banner
st.markdown("""
<div class="header-card">
    <div class="header-title">🏡 ValuHome AI Real Estate Valuation</div>
    <div class="header-subtitle">Machine Learning Powered Valuation Engine (Prices in Indian Rupees ₹)</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# AUTHENTICATION SCREEN (LOGIN / REGISTER)
# ---------------------------------------------------------
if not st.session_state['authenticated']:
    col_center1, col_center2, col_center3 = st.columns([1, 1.8, 1])
    
    with col_center2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        auth_mode = st.radio("Select Portal Option", ["🔑 Login", "📝 Create Account"], horizontal=True)
        
        if auth_mode == "🔑 Login":
            st.markdown("### Account Login")
            username_input = st.text_input("Username", key="login_user", placeholder="Enter username")
            password_input = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("Log In 🚀", type="primary", use_container_width=True):
                    db = st.session_state['user_db']
                    if username_input in db and db[username_input] == password_input:
                        st.session_state['authenticated'] = True
                        st.session_state['user'] = username_input
                        st.toast("🎉 Login successful!", icon="✅")
                        st.rerun()
                    else:
                        st.error("❌ Invalid Username or Password")
            
            with col_b2:
                if st.button("⚡ Quick Demo Login", use_container_width=True):
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = "Demo User"
                    st.toast("🎉 Logged in as Demo User!", icon="🚀")
                    st.rerun()
                    
            st.caption("Default Demo Credentials: `admin` / `admin123` or click **Quick Demo Login**")

        else:
            st.markdown("### Create New Account")
            new_username = st.text_input("Choose Username", key="reg_user")
            new_password = st.text_input("Choose Password", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_pass_conf")
            
            if st.button("Register Account ✨", type="primary", use_container_width=True):
                if not new_username or not new_password:
                    st.warning("Please fill in all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif new_username in st.session_state['user_db']:
                    st.error("Username already exists. Please pick another.")
                else:
                    st.session_state['user_db'][new_username] = new_password
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = new_username
                    st.success("Account created successfully!")
                    st.rerun()
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.stop()  # Halt execution until authenticated

# ---------------------------------------------------------
# MAIN DASHBOARD (AUTHENTICATED ACCESS ONLY)
# ---------------------------------------------------------

# Main Navigation Tabs
tab_predict, tab_models, tab_eda, tab_data = st.tabs([
    "🎯 Price Predictor (₹)", 
    "📊 ML Models & Performance", 
    "📈 Market Data Insights", 
    "📂 Dataset Explorer"
])

# ---------------------------------------------------------
# TAB 1: PRICE PREDICTOR
# ---------------------------------------------------------
with tab_predict:
    st.subheader(f"Welcome, {st.session_state['user']}! Configure Property Details")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📐 Property Size & Layout")
        sqft = st.slider("Square Feet (Living Area)", min_value=600, max_value=5500, value=2000, step=50, help="Total built-up interior living space")
        bedrooms = st.selectbox("Bedrooms (BHK)", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=2)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_input2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📍 Location & Condition")
        location = st.selectbox("Neighborhood Location", options=['Suburbs', 'Downtown', 'Rural', 'Prime Coastal'], index=0)
        overall_condition = st.slider("Overall Property Condition (1 = Poor, 10 = Excellent)", min_value=1, max_value=10, value=7, step=1)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_input3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🚘 Age & Amenities")
        age_years = st.slider("House Age (Years)", min_value=0, max_value=45, value=8, step=1)
        garage_cars = st.selectbox("Parking Capacity (Cars)", options=[0, 1, 2, 3], index=2)
        st.markdown("</div>", unsafe_allow_html=True)

    input_data = pd.DataFrame({
        'Square_Feet': [sqft],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age_Years': [age_years],
        'Garage_Cars': [garage_cars],
        'Location': [location],
        'Overall_Condition': [overall_condition]
    })
    
    predicted_price = pipeline.predict(input_data)[0]
    min_range = max(0, predicted_price - best_rmse)
    max_range = predicted_price + best_rmse
    
    st.markdown("---")
    
    col_res1, col_res2 = st.columns([1.4, 1])
    
    with col_res1:
        st.markdown(f"""
        <div class="price-display-card">
            <span class="metric-badge">Model: {best_model_name}</span>
            <span class="metric-badge">R² Accuracy: {metrics[best_model_name]['R2']}</span>
            <div style="color: #475569; margin-top: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">ESTIMATED MARKET VALUE</div>
            <div class="price-value-primary">{format_inr(predicted_price)}</div>
            <div class="price-value-sub">(₹{predicted_price:,.0f})</div>
            <div class="price-range">Confidence Interval: <strong>{format_inr(min_range)}</strong> – <strong>{format_inr(max_range)}</strong> (±{format_inr(best_rmse)} RMSE)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 Value Driver Summary")
        
        sqft_rate = predicted_price / sqft if sqft > 0 else 0
        st.write(f"• **Estimated Value per Sq. Ft:** `₹{sqft_rate:,.2f}/sqft`")
        st.write(f"• **Location Tier:** `{location}`")
        st.write(f"• **Condition Rating:** `{overall_condition} / 10`")
        st.write(f"• **Property Age:** `{age_years} years` ({'Modern/New' if age_years < 10 else 'Established'})")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: ML MODELS & PERFORMANCE
# ---------------------------------------------------------
with tab_models:
    st.subheader("Model Benchmark & Evaluation Metrics (in ₹)")
    
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df.sort_values(by='R2', ascending=False)
    
    metrics_display = metrics_df.copy()
    metrics_display['MAE (₹)'] = metrics_display['MAE'].apply(lambda x: format_inr(x))
    metrics_display['RMSE (₹)'] = metrics_display['RMSE'].apply(lambda x: format_inr(x))
    
    col_m1, col_m2 = st.columns([1.1, 1.1])
    
    with col_m1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🏆 Model Performance Comparison")
        st.dataframe(
            metrics_display[['R2', 'MAE (₹)', 'RMSE (₹)']].style.highlight_max(axis=0, subset=['R2'], color='#c7d2fe'),
            use_container_width=True
        )
        st.markdown(f"**Selected Deployment Model:** `{best_model_name}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_m2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 R² Score Comparison")
        
        fig_r2 = px.bar(
            metrics_df.reset_index(),
            x='index',
            y='R2',
            color='R2',
            labels={'index': 'Model Architecture', 'R2': 'R² Score'},
            color_continuous_scale='Tealgrn',
            text='R2'
        )
        fig_r2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#0f172a',
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if model_bundle.get('feature_importance'):
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔑 Feature Importance Drivers")
        imp_df = pd.DataFrame(model_bundle['feature_importance'])
        
        fig_imp = px.bar(
            imp_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#0f172a',
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: MARKET DATA INSIGHTS
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA in ₹)")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📐 Price (₹) vs. Square Footage by Location")
        fig_scatter = px.scatter(
            df,
            x='Square_Feet',
            y='Price',
            color='Location',
            size='Bedrooms',
            hover_data=['Overall_Condition', 'Age_Years'],
            opacity=0.85,
            labels={'Price': 'Price in Rupees (₹)'},
            color_discrete_sequence=['#4338ca', '#0d9488', '#d97706', '#e11d48']
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#0f172a',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_e2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🏷️ Housing Price Distribution (₹)")
        fig_hist = px.histogram(
            df,
            x='Price',
            color='Location',
            nbins=40,
            marginal='box',
            labels={'Price': 'Price in Rupees (₹)'},
            color_discrete_sequence=['#4338ca', '#0d9488', '#d97706', '#e11d48']
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#0f172a',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔗 Feature Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale='Blues',
        aspect="auto"
    )
    fig_corr.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#0f172a'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: DATASET EXPLORER
# ---------------------------------------------------------
with tab_data:
    st.subheader("Raw Housing Dataset (Prices in ₹)")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    loc_filter = st.multiselect("Filter by Location", options=df['Location'].unique(), default=list(df['Location'].unique()))
    filtered_df = df[df['Location'].isin(loc_filter)].copy()
    filtered_df['Price_Formatted'] = filtered_df['Price'].apply(format_inr)
    
    st.dataframe(filtered_df[['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age_Years', 'Garage_Cars', 'Location', 'Overall_Condition', 'Price_Formatted', 'Price']], use_container_width=True)
    st.write(f"Showing **{len(filtered_df)}** records out of **{len(df)}** total properties.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>ValuHome AI Engine • Built with Python, Scikit-Learn & Streamlit</div>", unsafe_allow_html=True)
