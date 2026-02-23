import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix
from scipy import stats
import warnings
import time
from io import BytesIO
warnings.filterwarnings('ignore')

# Page config with better defaults
st.set_page_config(
    page_title="Data Detective 🔍",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Data Detective - Your AI-powered data analysis companion"
    }
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Global styles */
    .main {
        padding-top: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        border-left: 6px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    
    .section-header:hover {
        transform: translateX(5px);
    }
    
    .analyzer-header {
        border-left: 6px solid #3498db;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .visualizer-header {
        border-left: 6px solid #e74c3c;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }
    
    /* Cards and boxes */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #ff9800;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #2196f3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 12px 24px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 2px solid #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Data frame styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Progress bar custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f5f7fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #667eea;
        font-weight: 600;
    }
    
    /* Stats card */
    .stat-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #f0f0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with better structure
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'df': None,
        'analysis_done': False,
        'analysis_results': None,
        'problem_type': None,
        'target_column': None,
        'last_upload_time': None,
        'analysis_cache': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== HELPER FUNCTIONS ====================

@st.cache_data(show_spinner=False, ttl=3600, max_entries=5)
def load_data(uploaded_file):
    """Load data with better error handling and progress indication"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Basic validation
        if df.empty:
            raise ValueError("The uploaded file is empty")
        
        if len(df.columns) < 2:
            raise ValueError("Dataset must have at least 2 columns")
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def detect_problem_type(target_series):
    """Detect if problem is regression or classification with better logic"""
    # Handle object types
    if target_series.dtype == 'object':
        return 'classification'
    
    # Check for boolean
    if target_series.dtype == 'bool':
        return 'classification'
    
    # Check unique values ratio
    unique_ratio = target_series.nunique() / len(target_series)
    
    # If very few unique values relative to size, likely classification
    if target_series.nunique() <= 20 and unique_ratio < 0.05:
        return 'classification'
    
    return 'regression'

def calculate_vif(df, numeric_cols):
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Limit to max 15 features for performance
        if len(numeric_cols) > 15:
            st.info("🔔 Too many features for VIF calculation. Showing top 15 by variance.")
            variances = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(15).index.tolist()
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        
        # Calculate VIF with timeout protection
        vif_values = []
        for i in range(len(numeric_cols)):
            try:
                vif = variance_inflation_factor(df[numeric_cols].values, i)
                vif_values.append(vif)
            except:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        return vif_data
    except Exception as e:
        st.warning(f"Could not calculate VIF: {str(e)}")
        return None

@st.cache_data(show_spinner=False, max_entries=5)
def perform_regression_analysis(X, y):
    """Perform comprehensive regression analysis with optimizations"""
    results = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    results['r2_score'] = r2_score(y_test, y_pred)
    results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    results['mae'] = np.mean(np.abs(y_test - y_pred))
    results['y_test'] = y_test
    results['y_pred'] = y_pred
    
    # Store train/test split info
    results['train_size'] = len(X_train)
    results['test_size'] = len(X_test)
    
    results['coefficients'] = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr.coef_,
        'Abs_Coefficient': np.abs(lr.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Random Forest with optimized parameters
    n_estimators = min(50, max(10, len(X_train) // 10))  # Adaptive
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=min(10, len(X.columns) * 2),
        min_samples_split=max(2, len(X_train) // 100),
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['rf_r2'] = r2_score(y_test, rf_pred)
    results['rf_rmse'] = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    results['feature_importance'] = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Calculate correlations
    def safe_pearson(col, y):
        try:
            return stats.pearsonr(col, y)
        except:
            return (np.nan, np.nan)

    corr_matrix = X.apply(lambda col: safe_pearson(col, y))

    results['correlations'] = pd.DataFrame({
        'Feature': X.columns,
        'Correlation': pd.to_numeric([c[0] for c in corr_matrix], errors='coerce'),
        'P_Value': pd.to_numeric([c[1] for c in corr_matrix], errors='coerce')
    })
    
    results['correlations']['Significant'] = (
        results['correlations']['P_Value'] < 0.05
    ).fillna(False).map({True: '✓', False: '✗'})
    
    results['correlations'] = results['correlations'].sort_values(
        'Correlation', key=abs, ascending=False
    )
    
    return results

@st.cache_data(show_spinner=False, max_entries=5)
def perform_classification_analysis(X, y):
    """Perform comprehensive classification analysis with optimizations"""
    results = {}
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Store split info
    results['train_size'] = len(X_train)
    results['test_size'] = len(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)
    
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['y_test'] = y_test
    results['y_pred'] = y_pred
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Try to calculate ROC AUC (for binary classification)
    try:
        if len(np.unique(y)) == 2:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            results['roc_auc'] = roc_auc_score(
                y_test, y_pred_proba, multi_class='ovr', average='weighted'
            )
    except:
        results['roc_auc'] = None
    
    # Coefficients (for binary or first class in multi-class)
    if len(lr.coef_) == 1:
        coef = lr.coef_[0]
    else:
        coef = lr.coef_[0]  # Use first class
    
    results['coefficients'] = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Random Forest with adaptive parameters
    n_estimators = min(50, max(10, len(X_train) // 10))
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=min(10, len(X.columns) * 2),
        min_samples_split=max(2, len(X_train) // 100),
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['rf_accuracy'] = accuracy_score(y_test, rf_pred)
    
    results['feature_importance'] = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Calculate correlations with target (encoded)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    def safe_pearson(col, y):
        try:
            return stats.pearsonr(col, y)
        except:
            return (np.nan, np.nan)

    corr_matrix = X.apply(lambda col: safe_pearson(col, y_encoded))

    results['correlations'] = pd.DataFrame({
        'Feature': X.columns,
        'Correlation': pd.to_numeric([c[0] for c in corr_matrix], errors='coerce'),
        'P_Value': pd.to_numeric([c[1] for c in corr_matrix], errors='coerce')
    })
    
    results['correlations']['Significant'] = (
        results['correlations']['P_Value'] < 0.05
    ).fillna(False).map({True: '✓', False: '✗'})
    
    results['correlations'] = results['correlations'].sort_values(
        'Correlation', key=abs, ascending=False
    )
    
    return results

def create_download_button(df, filename, label="📥 Download Data"):
    """Create a download button for dataframes"""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

def display_metric_card(col, title, value, subtitle=""):
    """Display a beautiful metric card"""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">{title}</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem;">{value}</div>
            <div style="font-size: 0.85rem; opacity: 0.8;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

def show_progress_message(message, progress_bar):
    """Show animated progress message"""
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success(f"✓ {message}")

# ==================== MAIN APP ====================

# Sidebar with improved UX
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with your data"
    )
    
    if uploaded_file:
        # Check if file has changed
        file_changed = (
            st.session_state.last_upload_time != uploaded_file.name + 
            str(uploaded_file.size)
        )
        
        if file_changed:
            with st.spinner("🔄 Loading your data..."):
                st.session_state.df = load_data(uploaded_file)
                st.session_state.last_upload_time = (
                    uploaded_file.name + str(uploaded_file.size)
                )
                st.session_state.analysis_done = False
                st.session_state.analysis_results = None
    
    # Show dataset info and target selection if data is loaded (from either upload or session)
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Quick stats in sidebar
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing data warning
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 0:
            st.warning(f"⚠️ {missing_pct:.1f}% missing data")
        
        st.markdown("---")
        st.markdown("### 🎯 Target Selection")
        
        target_col = st.selectbox(
            "Select target variable",
            options=df.columns.tolist(),
            help="Choose the column you want to predict"
        )
        
        if target_col:
            st.session_state.target_column = target_col
            problem_type = detect_problem_type(df[target_col])
            st.session_state.problem_type = problem_type
            
            # Show problem type
            if problem_type == 'regression':
                st.info("📈 Regression Problem")
            else:
                st.info("🎯 Classification Problem")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [Documentation](#)
    - [Examples](#)
    - [GitHub](#)
    """)

# Main content area
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Header with animation
    st.markdown('<div class="main-header">🔍 Data Detective</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Discover insights in your data with AI-powered analysis</div>',
        unsafe_allow_html=True
    )
    
    # Main navigation tabs
    main_tab = st.radio(
        "Choose your tool:",
        ["📊 Overview", "🔬 Analyzer", "📈 Visualizer"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Overview Tab
    if main_tab == "📊 Overview":
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        display_metric_card(col1, "Total Rows", f"{len(df):,}", "Data points")
        display_metric_card(col2, "Total Columns", len(df.columns), "Features")
        display_metric_card(
            col3, "Numeric Features", 
            len(df.select_dtypes(include=[np.number]).columns), 
            "Quantitative"
        )
        display_metric_card(
            col4, "Categorical Features",
            len(df.select_dtypes(include=['object']).columns),
            "Qualitative"
        )
        
        # Data preview with better styling
        st.markdown("### 👀 Data Preview")
        st.markdown("*First 10 rows of your dataset*")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=400
        )
        
        create_download_button(df.head(100), "preview_data.csv", "📥 Download Preview (100 rows)")
        
        # Column information
        st.markdown("### 📋 Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).values,
            'Unique': df.nunique().values
        })
        
        st.dataframe(col_info, use_container_width=True, height=400)
        create_download_button(col_info, "column_info.csv", "📥 Download Column Info")
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        stat_type = st.radio(
            "Choose statistics type:",
            ["Numeric Only", "All Columns"],
            horizontal=True
        )
        
        if stat_type == "Numeric Only":
            stats_df = df.describe()
        else:
            stats_df = df.describe(include='all')
        
        st.dataframe(stats_df, use_container_width=True)
        create_download_button(stats_df.T, "summary_stats.csv", "📥 Download Statistics")
    
    # Analyzer Tab
    elif main_tab == "🔬 Analyzer":
        st.markdown(
            '<div class="section-header analyzer-header">🔬 Advanced Data Analyzer</div>',
            unsafe_allow_html=True
        )
        
        if st.session_state.target_column is None:
            st.warning("⚠️ Please select a target variable in the sidebar first!")
        else:
            target_col = st.session_state.target_column
            problem_type = st.session_state.problem_type
            
            # Create tabs for different analyses
            analysis_tabs = st.tabs([
                "🎯 Auto Analysis",
                "🔍 Feature Deep Dive",
                "🔗 Relationship Explorer",
                "🧪 Interaction Tester"
            ])
            
            # Tab 1: Auto Analysis
            with analysis_tabs[0]:
                st.markdown("### 🎯 One-Click Comprehensive Analysis")
                
                st.markdown("""
                <div class="info-box">
                    <strong>What this does:</strong><br>
                    Like a full medical checkup, this analyzes your entire dataset and shows:
                    <ul>
                        <li>📊 Feature importance rankings</li>
                        <li>📈 Model performance metrics</li>
                        <li>🔍 Statistical correlations</li>
                        <li>⚠️ Potential issues (multicollinearity, missing data)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    run_analysis = st.button(
                        "🚀 Run Analysis",
                        type="primary",
                        use_container_width=True
                    )
                
                if run_analysis or st.session_state.analysis_done:
                    if not st.session_state.analysis_done:
                        # Prepare data
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔄 Preparing data...")
                        progress_bar.progress(10)
                        
                        # Drop target and non-numeric columns
                        X = df.drop(columns=[target_col])
                        y = df[target_col]
                        
                        # Handle missing values
                        numeric_cols = X.select_dtypes(include=[np.number]).columns
                        X_numeric = X[numeric_cols].fillna(X[numeric_cols].mean())
                        
                        if len(X_numeric.columns) == 0:
                            st.error("❌ No numeric features found for analysis!")
                            st.stop()
                        
                        status_text.text("🧮 Training models...")
                        progress_bar.progress(40)
                        
                        # Perform analysis
                        if problem_type == 'regression':
                            results = perform_regression_analysis(X_numeric, y)
                        else:
                            results = perform_classification_analysis(X_numeric, y)
                        
                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        
                        st.session_state.analysis_results = results
                        st.session_state.analysis_done = True
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success("✓ Analysis completed successfully!")
                        st.balloons()
                    
                    # Display results
                    results = st.session_state.analysis_results
                    
                    # Performance Metrics
                    st.markdown("### 🎯 Model Performance")
                    
                    if problem_type == 'regression':
                        col1, col2, col3, col4 = st.columns(4)
                        display_metric_card(
                            col1, "R² Score (Linear)", 
                            f"{results['r2_score']:.4f}",
                            "Variance explained"
                        )
                        display_metric_card(
                            col2, "RMSE", 
                            f"{results['rmse']:.2f}",
                            "Prediction error"
                        )
                        display_metric_card(
                            col3, "R² Score (RF)", 
                            f"{results['rf_r2']:.4f}",
                            "Random Forest"
                        )
                        display_metric_card(
                            col4, "MAE", 
                            f"{results['mae']:.2f}",
                            "Mean Absolute Error"
                        )
                        
                        st.markdown("""
                        <div class="info-box">
                            <strong>📘 Understanding the metrics:</strong><br>
                            • <strong>R² Score</strong>: Like a grade (0-1). Higher is better. 0.7+ is good!<br>
                            • <strong>RMSE</strong>: Average prediction error in your target's units<br>
                            • <strong>MAE</strong>: Average absolute error (easier to interpret than RMSE)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        display_metric_card(
                            col1, "Accuracy (Logistic)",
                            f"{results['accuracy']:.2%}",
                            "Correct predictions"
                        )
                        display_metric_card(
                            col2, "Accuracy (RF)",
                            f"{results['rf_accuracy']:.2%}",
                            "Random Forest"
                        )
                        if results['roc_auc']:
                            display_metric_card(
                                col3, "ROC AUC",
                                f"{results['roc_auc']:.4f}",
                                "Classification quality"
                            )
                        display_metric_card(
                            col4, "Test Size",
                            results['test_size'],
                            "samples"
                        )
                    
                    # Feature Importance
                    st.markdown("### 🏆 Top Important Features")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Linear Model Coefficients")
                        top_coef = results['coefficients'].head(10)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_coef)))
                        ax.barh(top_coef['Feature'], top_coef['Abs_Coefficient'], color=colors)
                        ax.set_xlabel('Coefficient Magnitude', fontsize=11, fontweight='bold')
                        ax.set_title('Linear Model Feature Impact', fontsize=13, fontweight='bold')
                        ax.grid(True, alpha=0.2, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        create_download_button(
                            results['coefficients'],
                            "coefficients.csv",
                            "📥 Download Coefficients"
                        )
                    
                    with col2:
                        st.markdown("#### Random Forest Importance")
                        top_imp = results['feature_importance'].head(10)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_imp)))
                        ax.barh(top_imp['Feature'], top_imp['Importance'], color=colors)
                        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                        ax.set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
                        ax.grid(True, alpha=0.2, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        create_download_button(
                            results['feature_importance'],
                            "feature_importance.csv",
                            "📥 Download Importance"
                        )
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>🎯 How to interpret:</strong><br>
                        • <strong>Coefficients</strong>: Direct impact strength. Like: "For every $1 increase in price, sales drop by 2 units"<br>
                        • <strong>Importance</strong>: Overall predictive power. Like: "Temperature is the #1 factor in ice cream sales"
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Correlations
                    st.markdown("### 🔗 Feature Correlations with Target")
                    
                    corr_df = results['correlations'].copy()
                    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
                    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(15)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in corr_df['Correlation']]
                    ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                    ax.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
                    ax.set_title('Feature Correlations (Green=Positive, Red=Negative)', 
                               fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.2, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show correlation table
                    st.dataframe(
                        results['correlations'],
                        use_container_width=True,
                        height=300
                    )
                    
                    create_download_button(
                        results['correlations'],
                        "correlations.csv",
                        "📥 Download Correlations"
                    )
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>📊 Correlation Guide:</strong><br>
                        • <strong>+1.0</strong>: Perfect positive (both go up together)<br>
                        • <strong>0.0</strong>: No linear relationship<br>
                        • <strong>-1.0</strong>: Perfect negative (one up, other down)<br>
                        • <strong>|r| > 0.7</strong>: Strong correlation<br>
                        • <strong>✓</strong> = Statistically significant (p < 0.05)
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tab 2: Feature Deep Dive
            with analysis_tabs[1]:
                st.markdown("### 🔍 Deep Dive into Individual Features")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in numeric_cols:
                    numeric_cols.remove(target_col)
                
                if len(numeric_cols) == 0:
                    st.warning("No numeric features available for analysis.")
                else:
                    selected_feature = st.selectbox(
                        "Select a feature to analyze:",
                        options=numeric_cols,
                        help="Choose a feature for detailed statistical analysis"
                    )
                    
                    if selected_feature:
                        feature_data = df[selected_feature].dropna()
                        
                        # Statistics cards
                        col1, col2, col3, col4, col5 = st.columns(5)
                        display_metric_card(col1, "Mean", f"{feature_data.mean():.2f}", "Average")
                        display_metric_card(col2, "Median", f"{feature_data.median():.2f}", "Middle value")
                        display_metric_card(col3, "Std Dev", f"{feature_data.std():.2f}", "Spread")
                        display_metric_card(col4, "Min", f"{feature_data.min():.2f}", "Lowest")
                        display_metric_card(col5, "Max", f"{feature_data.max():.2f}", "Highest")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Distribution")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.hist(feature_data, bins=30, color='#667eea', 
                                   alpha=0.7, edgecolor='black')
                            ax.axvline(feature_data.mean(), color='red', 
                                      linestyle='--', linewidth=2, label='Mean')
                            ax.axvline(feature_data.median(), color='green', 
                                      linestyle='--', linewidth=2, label='Median')
                            ax.set_xlabel(selected_feature, fontsize=11)
                            ax.set_ylabel('Frequency', fontsize=11)
                            ax.set_title(f'Distribution of {selected_feature}', 
                                       fontsize=12, fontweight='bold')
                            ax.legend()
                            ax.grid(True, alpha=0.2)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.markdown("#### Box Plot")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            bp = ax.boxplot([feature_data], vert=True, patch_artist=True,
                                          boxprops=dict(facecolor='#667eea', alpha=0.7),
                                          medianprops=dict(color='red', linewidth=2),
                                          whiskerprops=dict(linewidth=1.5),
                                          capprops=dict(linewidth=1.5))
                            ax.set_ylabel(selected_feature, fontsize=11)
                            ax.set_title(f'Box Plot of {selected_feature}', 
                                       fontsize=12, fontweight='bold')
                            ax.grid(True, alpha=0.2, axis='y')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # Outlier detection
                        Q1 = feature_data.quantile(0.25)
                        Q3 = feature_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = feature_data[
                            (feature_data < Q1 - 1.5 * IQR) | 
                            (feature_data > Q3 + 1.5 * IQR)
                        ]
                        
                        if len(outliers) > 0:
                            st.warning(f"""
                            ⚠️ **{len(outliers)} outliers detected ({len(outliers)/len(feature_data)*100:.1f}%)**
                            
                            **Analogy**: Like having a few extremely tall people in a height survey.
                            They're real data, but might affect your average!
                            
                            **Recommendation**: Consider if these are errors or genuine extreme values.
                            """)
                        
                        # Statistical tests
                        st.markdown("#### 📊 Statistical Tests")
                        
                        # Normality test
                        _, p_value = stats.normaltest(feature_data)
                        
                        if p_value > 0.05:
                            st.success(f"""
                            ✅ **Data appears normally distributed** (p={p_value:.4f})
                            
                            **Analogy**: Like heights in a population - bell curve shape!
                            **Good for**: Most statistical methods work well
                            """)
                        else:
                            st.info(f"""
                            ℹ️ **Data is not normally distributed** (p={p_value:.4f})
                            
                            **Analogy**: Like income - a few very high earners skew things.
                            **Consider**: Log transformation or non-parametric methods
                            """)
            
            # Tab 3: Relationship Explorer
            with analysis_tabs[2]:
                st.markdown("### 🔗 Explore Feature Relationships")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in numeric_cols:
                    numeric_cols.remove(target_col)
                
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric features to explore relationships.")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        feature_x = st.selectbox(
                            "Select X-axis feature:",
                            options=numeric_cols,
                            key="feat_x"
                        )
                    
                    with col2:
                        remaining_cols = [c for c in numeric_cols if c != feature_x]
                        feature_y = st.selectbox(
                            "Select Y-axis feature:",
                            options=remaining_cols,
                            key="feat_y"
                        )
                    
                    if feature_x and feature_y:
                        # Calculate correlation
                        corr, p_val = stats.pearsonr(
                            df[feature_x].dropna(),
                            df[feature_y].dropna()
                        )
                        
                        # Display correlation
                        col1, col2, col3 = st.columns(3)
                        display_metric_card(
                            col1, "Correlation", 
                            f"{corr:.4f}",
                            "Pearson's r"
                        )
                        display_metric_card(
                            col2, "P-Value",
                            f"{p_val:.4f}",
                            "Significance"
                        )
                        
                        if abs(corr) > 0.7:
                            strength = "Strong"
                            color = "#2ecc71"
                        elif abs(corr) > 0.4:
                            strength = "Moderate"
                            color = "#f39c12"
                        else:
                            strength = "Weak"
                            color = "#e74c3c"
                        
                        display_metric_card(
                            col3, "Relationship",
                            strength,
                            f"<span style='color:{color}'>●</span> Correlation strength"
                        )
                        
                        # Scatter plot
                        st.markdown("#### Scatter Plot with Trend Line")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Clean data
                        plot_data = df[[feature_x, feature_y]].dropna()
                        
                        # Scatter
                        ax.scatter(
                            plot_data[feature_x],
                            plot_data[feature_y],
                            alpha=0.6,
                            s=50,
                            c='#667eea',
                            edgecolors='white',
                            linewidth=0.5
                        )
                        
                        # Trend line
                        z = np.polyfit(plot_data[feature_x], plot_data[feature_y], 1)
                        p = np.poly1d(z)
                        ax.plot(
                            plot_data[feature_x],
                            p(plot_data[feature_x]),
                            "r--",
                            linewidth=2,
                            label=f'Trend (r={corr:.3f})'
                        )
                        
                        ax.set_xlabel(feature_x, fontsize=12, fontweight='bold')
                        ax.set_ylabel(feature_y, fontsize=12, fontweight='bold')
                        ax.set_title(
                            f'{feature_x} vs {feature_y}',
                            fontsize=14,
                            fontweight='bold'
                        )
                        ax.legend()
                        ax.grid(True, alpha=0.2)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Interpretation
                        if p_val < 0.05:
                            sig_text = "statistically significant ✓"
                        else:
                            sig_text = "not statistically significant ✗"
                        
                        if corr > 0:
                            direction = "positive (both increase together)"
                        else:
                            direction = "negative (one increases, other decreases)"
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>📘 Interpretation:</strong><br>
                            • The relationship is <strong>{direction}</strong><br>
                            • Strength: <strong>{strength.lower()}</strong> (r = {corr:.4f})<br>
                            • This relationship is <strong>{sig_text}</strong> (p = {p_val:.4f})<br><br>
                            <strong>Analogy:</strong> Think of it like...
                            {"☕ Coffee and alertness - more coffee → more alert!" if corr > 0.5 else
                             "📉 Price and demand - higher price → lower demand!" if corr < -0.5 else
                             "🤷 Temperature and lottery numbers - pretty much random!"}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tab 4: Interaction Tester
            with analysis_tabs[3]:
                st.markdown("### 🧪 Test Feature Interactions")
                
                st.markdown("""
                <div class="info-box">
                    <strong>What are interactions?</strong><br>
                    When the effect of one feature depends on another feature!<br><br>
                    <strong>Analogy:</strong> Coffee + sugar = tasty. But coffee × temperature = crucial!
                    Cold coffee is bad regardless of sugar. The effect of sugar depends on temperature!
                </div>
                """, unsafe_allow_html=True)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in numeric_cols:
                    numeric_cols.remove(target_col)
                
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric features to test interactions.")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        feat1 = st.selectbox(
                            "Select first feature:",
                            options=numeric_cols,
                            key="inter_feat1"
                        )
                    
                    with col2:
                        remaining = [c for c in numeric_cols if c != feat1]
                        feat2 = st.selectbox(
                            "Select second feature:",
                            options=remaining,
                            key="inter_feat2"
                        )
                    
                    if st.button("🧪 Test Interaction", type="primary"):
                        with st.spinner("Testing interaction..."):
                            # Prepare data
                            test_df = df[[feat1, feat2, target_col]].dropna()
                            X_no_inter = test_df[[feat1, feat2]]
                            X_with_inter = X_no_inter.copy()
                            X_with_inter['interaction'] = (
                                X_no_inter[feat1] * X_no_inter[feat2]
                            )
                            y = test_df[target_col]
                            
                            # Train models
                            X_train_no, X_test_no, y_train, y_test = train_test_split(
                                X_no_inter, y, test_size=0.2, random_state=42
                            )
                            X_train_yes, X_test_yes, _, _ = train_test_split(
                                X_with_inter, y, test_size=0.2, random_state=42
                            )
                            
                            if problem_type == 'regression':
                                # Without interaction
                                model_no = LinearRegression()
                                model_no.fit(X_train_no, y_train)
                                score_no = model_no.score(X_test_no, y_test)
                                
                                # With interaction
                                model_yes = LinearRegression()
                                model_yes.fit(X_train_yes, y_train)
                                score_yes = model_yes.score(X_test_yes, y_test)
                                
                                metric_name = "R² Score"
                            else:
                                # Without interaction
                                model_no = LogisticRegression(max_iter=1000)
                                model_no.fit(X_train_no, y_train)
                                score_no = model_no.score(X_test_no, y_test)
                                
                                # With interaction
                                model_yes = LogisticRegression(max_iter=1000)
                                model_yes.fit(X_train_yes, y_train)
                                score_yes = model_yes.score(X_test_yes, y_test)
                                
                                metric_name = "Accuracy"
                            
                            # Results
                            improvement = ((score_yes - score_no) / score_no) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            display_metric_card(
                                col1, f"{metric_name} (No Interaction)",
                                f"{score_no:.4f}",
                                "Baseline"
                            )
                            display_metric_card(
                                col2, f"{metric_name} (With Interaction)",
                                f"{score_yes:.4f}",
                                "Enhanced"
                            )
                            display_metric_card(
                                col3, "Improvement",
                                f"{improvement:+.2f}%",
                                "Performance gain"
                            )
                            
                            # Interpretation
                            if improvement > 5:
                                st.success(f"""
                                🎉 **Strong interaction detected!**
                                
                                Adding the interaction term improved performance by {improvement:.1f}%.
                                
                                **Analogy**: Like discovering that exercise + good diet works way better 
                                than just adding their individual effects!
                                
                                **Recommendation**: Include this interaction in your final model.
                                """)
                            elif improvement > 1:
                                st.info(f"""
                                ℹ️ **Moderate interaction present**
                                
                                Performance improved by {improvement:.1f}%.
                                
                                **Consider**: The interaction might be useful, but test on more data.
                                """)
                            else:
                                st.warning(f"""
                                📉 **Weak or no interaction**
                                
                                Only {improvement:.1f}% improvement (might be random noise).
                                
                                **Analogy**: Like expecting coffee + orange juice to be amazing - 
                                sometimes things just don't interact meaningfully!
                                """)
    
    # Visualizer Tab
    else:  # Visualizer
        st.markdown(
            '<div class="section-header visualizer-header">📈 Data Visualizer</div>',
            unsafe_allow_html=True
        )
        
        viz_tabs = st.tabs([
            "📊 Distributions",
            "🔥 Correlation Heatmap",
            "⚖️ Feature Comparison",
            "🎯 Target Analysis",
            "📉 Model Performance"
        ])
        
        # Tab 1: Distributions
        with viz_tabs[0]:
            st.markdown("### 📊 Feature Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.warning("No numeric columns to visualize.")
            else:
                # Multi-select for features
                selected_features = st.multiselect(
                    "Select features to visualize:",
                    options=numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    help="Choose up to 6 features"
                )
                
                if selected_features:
                    # Limit to 6 for performance
                    selected_features = selected_features[:6]
                    
                    # Calculate grid
                    n_features = len(selected_features)
                    n_cols = 2
                    n_rows = (n_features + 1) // 2
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
                    if n_features == 1:
                        axes = np.array([axes])
                    axes = axes.flatten()
                    
                    for idx, col in enumerate(selected_features):
                        ax = axes[idx]
                        data = df[col].dropna()
                        
                        # Histogram with KDE
                        ax.hist(
                            data,
                            bins=30,
                            alpha=0.7,
                            color='#667eea',
                            edgecolor='black',
                            density=True
                        )
                        
                        # KDE overlay
                        try:
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 100)
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
                        except:
                            pass
                        
                        ax.set_title(col, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Value', fontsize=10)
                        ax.set_ylabel('Density', fontsize=10)
                        ax.grid(True, alpha=0.2)
                        
                        # Add mean line
                        ax.axvline(
                            data.mean(),
                            color='green',
                            linestyle='--',
                            linewidth=2,
                            label=f'Mean: {data.mean():.2f}'
                        )
                        ax.legend(fontsize=8)
                    
                    # Hide extra subplots
                    for idx in range(n_features, len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>📘 Reading distributions:</strong><br>
                        • <strong>Bell curve</strong>: Normal distribution (most common in nature)<br>
                        • <strong>Skewed right</strong>: Long tail to the right (like income - few very rich)<br>
                        • <strong>Skewed left</strong>: Long tail to the left (like age at retirement - few very young)<br>
                        • <strong>Uniform</strong>: Flat - all values equally likely (like dice rolls)
                    </div>
                    """, unsafe_allow_html=True)
        
        # Tab 2: Correlation Heatmap
        with viz_tabs[1]:
            st.markdown("### 🔥 Correlation Heatmap")
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap.")
            else:
                # Option to filter features
                show_all = st.checkbox("Show all features", value=True)
                
                if not show_all:
                    max_features = st.slider(
                        "Number of features to show:",
                        min_value=5,
                        max_value=min(20, len(numeric_df.columns)),
                        value=min(10, len(numeric_df.columns))
                    )
                    
                    # Select top features by variance
                    variances = numeric_df.var().sort_values(ascending=False)
                    top_features = variances.head(max_features).index.tolist()
                    numeric_df = numeric_df[top_features]
                
                # Calculate correlation
                corr_matrix = numeric_df.corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Use a better colormap
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    ax=ax,
                    vmin=-1,
                    vmax=1
                )
                
                ax.set_title(
                    'Feature Correlation Matrix',
                    fontsize=16,
                    fontweight='bold',
                    pad=20
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Download correlation matrix
                create_download_button(
                    corr_matrix,
                    "correlation_matrix.csv",
                    "📥 Download Correlation Matrix"
                )
                
                st.markdown("""
                <div class="info-box">
                    <strong>🔥 Heatmap Guide:</strong><br>
                    • <strong>Red cells</strong>: Strong positive correlation (both go up)<br>
                    • <strong>Green cells</strong>: Strong negative correlation (one up, one down)<br>
                    • <strong>Yellow/white cells</strong>: Weak or no correlation<br><br>
                    <strong>Warning:</strong> Features with high correlation (|r| > 0.8) might cause multicollinearity!
                </div>
                """, unsafe_allow_html=True)
                
                # Find high correlations
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if high_corr:
                    st.warning("⚠️ **High correlations detected** (|r| > 0.8):")
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                    st.markdown("""
                    **Recommendation**: Consider removing one feature from each highly 
                    correlated pair to avoid multicollinearity in your model.
                    
                    **Analogy**: Like having "temperature in Celsius" and "temperature in 
                    Fahrenheit" - they tell you the same thing!
                    """)
        
        # Tab 3: Feature Comparison
        with viz_tabs[2]:
            st.markdown("### ⚖️ Side-by-Side Feature Comparison")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric features for comparison.")
            else:
                # Multi-select
                compare_features = st.multiselect(
                    "Select features to compare:",
                    options=numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))],
                    help="Choose 2-5 features"
                )
                
                if len(compare_features) >= 2:
                    compare_features = compare_features[:5]  # Limit
                    
                    # Box plot comparison
                    st.markdown("#### Box Plot Comparison")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Prepare data
                    plot_data = [df[col].dropna() for col in compare_features]
                    
                    bp = ax.boxplot(
                        plot_data,
                        labels=compare_features,
                        patch_artist=True,
                        notch=True
                    )
                    
                    # Color boxes
                    colors = plt.cm.Set3(np.linspace(0, 1, len(compare_features)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    
                    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
                    ax.set_title(
                        'Feature Comparison - Box Plots',
                        fontsize=14,
                        fontweight='bold'
                    )
                    ax.grid(True, alpha=0.2, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Statistics table
                    st.markdown("#### Summary Statistics")
                    
                    stats_df = df[compare_features].describe().T
                    stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                    stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']
                    
                    # Style the dataframe
                    st.dataframe(
                        stats_df.style.background_gradient(cmap='YlGnBu', axis=0),
                        use_container_width=True
                    )
                    
                    create_download_button(
                        stats_df,
                        "feature_comparison.csv",
                        "📥 Download Comparison"
                    )
        
        # Tab 4: Target Analysis  
        with viz_tabs[3]:
            st.markdown("### 🎯 Target Variable Analysis")
            
            if st.session_state.target_column:
                target_col = st.session_state.target_column
                target_data = df[target_col]
                problem_type = st.session_state.problem_type
                
                if problem_type == 'regression':
                    st.markdown("#### Target Distribution (Regression)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    display_metric_card(col1, "Mean", f"{target_data.mean():.2f}", "Average")
                    display_metric_card(col2, "Median", f"{target_data.median():.2f}", "Middle")
                    display_metric_card(col3, "Std Dev", f"{target_data.std():.2f}", "Spread")
                    display_metric_card(
                        col4, "Range",
                        f"{target_data.max() - target_data.min():.2f}",
                        "Max - Min"
                    )
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Histogram
                    ax1.hist(
                        target_data.dropna(),
                        bins=30,
                        color='#667eea',
                        alpha=0.7,
                        edgecolor='black'
                    )
                    ax1.axvline(
                        target_data.mean(),
                        color='red',
                        linestyle='--',
                        linewidth=2,
                        label='Mean'
                    )
                    ax1.axvline(
                        target_data.median(),
                        color='green',
                        linestyle='--',
                        linewidth=2,
                        label='Median'
                    )
                    ax1.set_xlabel(target_col, fontsize=11, fontweight='bold')
                    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    ax1.set_title('Distribution', fontsize=13, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.2)
                    
                    # Q-Q plot for normality check
                    stats.probplot(target_data.dropna(), dist="norm", plot=ax2)
                    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Normality test
                    _, p_value = stats.normaltest(target_data.dropna())
                    
                    if p_value > 0.05:
                        st.success(f"""
                        ✅ **Target appears normally distributed** (p={p_value:.4f})
                        
                        **Great!** Linear regression and similar models will work well.
                        """)
                    else:
                        st.info(f"""
                        ℹ️ **Target is not normally distributed** (p={p_value:.4f})
                        
                        **Consider**: Log transformation or tree-based models might work better.
                        """)
                
                else:  # Classification
                    st.markdown("#### Target Class Distribution")
                    
                    value_counts = target_data.value_counts()
                    
                    # Pie chart and bar chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Pie chart
                    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
                    ax1.pie(
                        value_counts.values,
                        labels=value_counts.index,
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90
                    )
                    ax1.set_title('Class Distribution (Pie)', fontsize=13, fontweight='bold')
                    
                    # Bar chart
                    ax2.bar(
                        value_counts.index.astype(str),
                        value_counts.values,
                        color=colors
                    )
                    ax2.set_xlabel(target_col, fontsize=11, fontweight='bold')
                    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
                    ax2.set_title('Class Distribution (Bar)', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.2, axis='y')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Distribution table
                    dist_df = pd.DataFrame({
                        'Class': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(dist_df, use_container_width=True)
                    
                    # Check for imbalance
                    if value_counts.max() / value_counts.min() > 3:
                        st.warning("""
                        ⚠️ **Class Imbalance Detected!**
                        
                        **Analogy**: Like a sports team where 90% are goalkeepers - 
                        the model might struggle to learn about the minority class.
                        
                        **Recommendation**: Consider using techniques like SMOTE, 
                        class weights, or stratified sampling.
                        """)
            else:
                st.info("👆 Select a target variable in the sidebar first!")
        
        # Tab 5: Model Performance
        with viz_tabs[4]:
            st.markdown("### 📉 Model Performance Visualization")
            
            if st.session_state.analysis_done and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                problem_type = st.session_state.problem_type
                
                if problem_type == 'regression':
                    st.markdown("#### 🎯 Actual vs Predicted")
                    
                    # Scatter plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(
                        results['y_test'],
                        results['y_pred'],
                        alpha=0.6,
                        s=60,
                        c='#667eea',
                        edgecolors='white',
                        linewidth=0.5
                    )
                    
                    # Perfect prediction line
                    min_val = min(results['y_test'].min(), results['y_pred'].min())
                    max_val = max(results['y_test'].max(), results['y_pred'].max())
                    ax.plot(
                        [min_val, max_val],
                        [min_val, max_val],
                        'r--',
                        linewidth=3,
                        label='Perfect Prediction',
                        alpha=0.7
                    )
                    
                    ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
                    ax.set_title(
                        f'Actual vs Predicted (R² = {results["r2_score"]:.4f})',
                        fontsize=14,
                        fontweight='bold'
                    )
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>📘 How to read:</strong><br>
                        • Points close to red line = good predictions<br>
                        • Points far from line = prediction errors<br>
                        • <strong>Analogy</strong>: Like darts on a dartboard - closer to center is better!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Residual plot
                    st.markdown("#### 📊 Residual Analysis")
                    residuals = results['y_test'] - results['y_pred']
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Residual plot
                    ax1.scatter(results['y_pred'], residuals, alpha=0.6, s=60, c='#764ba2')
                    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
                    ax1.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
                    ax1.set_ylabel('Residuals (Errors)', fontsize=11, fontweight='bold')
                    ax1.set_title('Residual Plot', fontsize=13, fontweight='bold')
                    ax1.grid(True, alpha=0.2)
                    
                    # Residual distribution
                    ax2.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
                    ax2.set_xlabel('Residuals', fontsize=11, fontweight='bold')
                    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                    ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold')
                    ax2.grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>📊 Residual Guide:</strong><br>
                        • Residuals = errors (actual - predicted)<br>
                        • Random scatter around zero = good model<br>
                        • Patterns = model missing something<br>
                        • <strong>Analogy</strong>: Like checking if a scale is calibrated - 
                          random errors are okay, systematic bias is not!
                    </div>
                    """, unsafe_allow_html=True)
                
                else:  # Classification
                    st.markdown("#### 🎯 Confusion Matrix")
                    
                    # Confusion matrix heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        results['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        square=True,
                        ax=ax,
                        xticklabels=np.unique(results['y_test']),
                        yticklabels=np.unique(results['y_test']),
                        cbar_kws={'label': 'Count'}
                    )
                    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
                    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>📊 Confusion Matrix Guide:</strong><br>
                        • Diagonal = correct predictions ✓<br>
                        • Off-diagonal = mistakes ✗<br>
                        • <strong>Analogy</strong>: Like a student's answer sheet - 
                          diagonal shows correct answers, off-diagonal shows where they got confused!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Feature importance comparison
                    st.markdown("#### 🎯 Model Comparison: Feature Importance")
                    
                    top_n = 10
                    coef_df = results['coefficients'].head(top_n).copy()
                    imp_df = results['feature_importance'].head(top_n).copy()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Coefficients
                    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(coef_df)))
                    coef_df.plot(
                        x='Feature',
                        y='Abs_Coefficient',
                        kind='barh',
                        ax=ax1,
                        legend=False,
                        color=colors1
                    )
                    ax1.set_xlabel('Coefficient Magnitude', fontsize=11, fontweight='bold')
                    ax1.set_ylabel('')
                    ax1.set_title('Logistic Regression Coefficients', 
                                fontsize=12, fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='x')
                    
                    # RF Importance
                    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(imp_df)))
                    imp_df.plot(
                        x='Feature',
                        y='Importance',
                        kind='barh',
                        ax=ax2,
                        legend=False,
                        color=colors2
                    )
                    ax2.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                    ax2.set_ylabel('')
                    ax2.set_title('Random Forest Importance', 
                                fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("""
                👆 Run 'Auto Analysis' in the Analyzer section first to see model 
                performance visualizations!
                
                **Tip**: Go to 🔬 Analyzer → 🎯 Auto Analysis
                """)

else:
    # Landing page with enhanced design
    st.markdown('<div class="main-header">🔍 Data Detective</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Discover hidden insights in your data with AI-powered analysis</div>',
        unsafe_allow_html=True
    )
    
    # Feature showcase
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🔬 Analyzer Department</h3>
            <p><strong>Analogy:</strong> Think of this as your data's diagnostic lab</p>
            <ul>
                <li><strong>Auto Analysis</strong>: Full health checkup in one click</li>
                <li><strong>Feature Deep Dive</strong>: Examine individual features under microscope</li>
                <li><strong>Relationship Explorer</strong>: See how features interact</li>
                <li><strong>Interaction Tester</strong>: Find hidden synergies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Visualizer Department</h3>
            <p><strong>Analogy:</strong> Your data's imaging center</p>
            <ul>
                <li><strong>Distributions</strong>: See how data spreads out</li>
                <li><strong>Correlation Heatmap</strong>: Network map of relationships</li>
                <li><strong>Comparisons</strong>: Side-by-side feature analysis</li>
                <li><strong>Target Analysis</strong>: Focus on your prediction goal</li>
                <li><strong>Model Performance</strong>: Visualize prediction accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    <div class="stat-card">
        <h4>📋 Quick Start Guide</h4>
        <ol>
            <li><strong>Upload your data</strong> - Use the sidebar to upload CSV or Excel</li>
            <li><strong>Select target</strong> - Choose what you want to predict</li>
            <li><strong>Explore</strong> - Use Analyzer for insights, Visualizer for charts</li>
            <li><strong>Download</strong> - Export your results and findings</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Example use cases
    st.markdown("### 💡 Example Use Cases")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h4>📈 Sales Prediction</h4>
            <p>Discover what drives revenue and visualize the patterns in beautiful charts!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h4>🎯 Customer Churn</h4>
            <p>Identify which factors matter most and see the relationships visually!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h4>💰 Price Optimization</h4>
            <p>Test feature interactions and optimize your pricing strategy!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data option
    st.markdown("---")
    st.markdown("### 🎲 Or Try with Sample Data")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🎲 Generate Sample Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating sample data..."):
                np.random.seed(42)
                n = 300
                
                sample_df = pd.DataFrame({
                    'price': np.random.uniform(10, 100, n),
                    'ad_spend': np.random.uniform(1000, 10000, n),
                    'discount_pct': np.random.uniform(0, 30, n),
                    'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n),
                    'competition': np.random.uniform(1, 10, n),
                    'quality_score': np.random.uniform(1, 5, n)
                })
                
                # Create target with realistic interactions
                sample_df['sales'] = (
                    -2.5 * sample_df['price'] +
                    0.8 * sample_df['ad_spend'] +
                    1.5 * sample_df['discount_pct'] +
                    0.002 * sample_df['ad_spend'] * sample_df['discount_pct'] +
                    300 * sample_df['quality_score'] +
                    -150 * sample_df['competition'] +
                    np.random.normal(0, 500, n)
                )
                
                st.session_state.df = sample_df
                st.session_state.last_upload_time = "sample_data"
                
                time.sleep(1)
                st.success("✓ Sample dataset loaded! Use the sidebar to get started.")
                st.balloons()
                time.sleep(1)
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;">
    Made with ❤️ by Data Detective | Powered by Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)