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
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Data Detective 🔍",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2ecc71;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f4f8 0%, #ffffff 100%);
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    .analyzer-header {
        border-left: 5px solid #3498db;
        background: linear-gradient(90deg, #e3f2fd 0%, #ffffff 100%);
    }
    .visualizer-header {
        border-left: 5px solid #e74c3c;
        background: linear-gradient(90deg, #ffebee 0%, #ffffff 100%);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# ==================== HELPER FUNCTIONS ====================
@st.cache_data(show_spinner=False, max_entries=3)
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

@st.cache_data(show_spinner=False, max_entries=3)
def detect_problem_type(target_series):
    """Detect if problem is regression or classification"""
    if target_series.dtype == 'object':
        return 'classification'
    if target_series.nunique() <= 10 and target_series.dtype in ['int64', 'int32']:
        return 'classification'
    return 'regression'

def calculate_vif(df, numeric_cols):
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i) 
                       for i in range(len(numeric_cols))]
    return vif_data

@st.cache_data(show_spinner=False, max_entries=3)
def perform_regression_analysis(X, y):
    """Perform comprehensive regression analysis"""
    results = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    results['r2_score'] = r2_score(y_test, y_pred)
    results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    results['y_test'] = y_test
    results['y_pred'] = y_pred
    results['coefficients'] = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr.coef_,
        'Abs_Coefficient': np.abs(lr.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Random Forest for feature importance
    rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['rf_r2'] = r2_score(y_test, rf_pred)
    
    results['feature_importance'] = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    def safe_pearson(col, y):
        try:
            return stats.pearsonr(col, y)
        except:
            return (np.nan, np.nan)

    corr_matrix = X.apply(lambda col: safe_pearson(col, y))

    results['correlations'] = pd.DataFrame({
        'Feature': X.columns,
        'Correlation': pd.to_numeric(
            [c[0] for c in corr_matrix],
            errors='coerce'
        ),
        'P_Value': pd.to_numeric(
            [c[1] for c in corr_matrix],
            errors='coerce'
        )
    })
    results['correlations']['P_Value'] = pd.to_numeric(
        results['correlations']['P_Value'],
        errors='coerce'
    )
    results['correlations']['Significant'] = (
      results['correlations']['P_Value'] < 0.05
    ).fillna(False).map({True: '✓', False: '✗'})
    results['correlations'] = results['correlations'].sort_values(
        'Correlation', key=abs, ascending=False
    )
    
    return results

@st.cache_data(show_spinner=False, max_entries=3)
def perform_classification_analysis(X, y):
    """Perform comprehensive classification analysis"""
    results = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)
    
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['y_test'] = y_test
    results['y_pred'] = y_pred
    results['y_pred_proba'] = y_pred_proba
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Handle binary vs multiclass
    if len(np.unique(y)) == 2:
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        results['roc_auc'] = roc_auc_score(y_test, y_pred_proba, 
                                           multi_class='ovr', average='weighted')
    
    results['coefficients'] = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr.coef_[0] if len(lr.coef_) == 1 else lr.coef_.mean(axis=0),
        'Abs_Coefficient': np.abs(lr.coef_[0] if len(lr.coef_) == 1 else lr.coef_.mean(axis=0))
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
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
    
    return results

def test_interaction(X, y, feature1, feature2, problem_type):
    """Test interaction effect between two features"""
    X_base = X.copy()
    X_interaction = X.copy()
    X_interaction[f'{feature1}_x_{feature2}'] = X[feature1] * X[feature2]
    
    X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
    )

    X_train_int = X_interaction.loc[X_train_base.index]
    X_test_int = X_interaction.loc[X_test_base.index]
    
    if problem_type == 'regression':
        model_base = LinearRegression()
        model_int = LinearRegression()
        model_base.fit(X_train_base, y_train)
        model_int.fit(X_train_int, y_train)
        
        score_base = r2_score(y_test, model_base.predict(X_test_base))
        score_int = r2_score(y_test, model_int.predict(X_test_int))
        metric_name = "R² Score"
    else:
        model_base = LogisticRegression(max_iter=1000)
        model_int = LogisticRegression(max_iter=1000)
        model_base.fit(X_train_base, y_train)
        model_int.fit(X_train_int, y_train)
        
        score_base = accuracy_score(y_test, model_base.predict(X_test_base))
        score_int = accuracy_score(y_test, model_int.predict(X_test_int))
        metric_name = "Accuracy"
    
    if score_base != 0:
        improvement = ((score_int - score_base) / score_base) * 100
    else:
        improvement = 0
    
    return {
        'base_score': score_base,
        'interaction_score': score_int,
        'improvement_pct': improvement,
        'metric_name': metric_name
    }

# ==================== MAIN APP ====================

st.markdown('<p class="main-header">🔍 Data Detective</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Uncover Hidden Patterns in Your Data</p>', unsafe_allow_html=True)

# Sidebar
# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    # If file uploaded, load it
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    # 👇 THIS IS THE KEY CHANGE
    # Show preview + target selection IF df exists (from upload OR sample)
    if st.session_state.df is not None:
        df = st.session_state.df

        st.success(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Data preview
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # Target selection
        st.subheader("🎯 Select Target Variable")
        target_col = st.selectbox(
            "What are you trying to predict?",
            options=df.columns.tolist(),
            help="Choose the column you want to analyze or predict"
        )
        
        if target_col:
            st.session_state.target_col = target_col
            
            # Basic info
            st.markdown("---")
            st.markdown("**Quick Stats:**")
            st.metric("Total Rows", len(df))
            st.metric("Total Features", len(df.columns) - 1)
            
            # Detect problem type
            problem_type = detect_problem_type(df[target_col])
            st.metric("Problem Type", 
                     "🎯 Classification" if problem_type == 'classification' else "📈 Regression")

# Main content
if st.session_state.df is not None and 'target_col' in st.session_state:
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    # Create main section tabs
    main_section = st.selectbox(
        "🎛️ Choose Section:",
        ["🏠 Overview", "🔬 Analyzer", "📊 Visualizer"],
        label_visibility="collapsed"
    )
    
    # ==================== OVERVIEW SECTION ====================
    if main_section == "🏠 Overview":
        st.markdown('<div class="section-header">🏠 Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Data Quality Check
        st.subheader("📋 Data Quality Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values by Column:**")
            missing_counts = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': missing_counts,
                'Percentage': (missing_counts / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✓ No missing values detected!")
        
        with col2:
            st.markdown("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Type': df.dtypes.value_counts().index.astype(str),
                'Count': df.dtypes.value_counts().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        # Statistical Summary
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    # ==================== ANALYZER SECTION ====================
    elif main_section == "🔬 Analyzer":
        st.markdown('<div class="section-header analyzer-header">🔬 Data Analyzer</div>', unsafe_allow_html=True)
        st.markdown("""
        **Think of the Analyzer as a medical lab:** It runs tests on your data to diagnose what's healthy, 
        what's problematic, and what relationships exist between features - just like blood tests reveal 
        hidden health indicators!
        """)
        
        # Create analyzer tabs
        analyzer_tabs = st.tabs([
            "🎯 Auto Analysis", 
            "🔍 Feature Deep Dive",
            "🔗 Relationship Explorer", 
            "⚡ Interaction Tester"
        ])
        
        # Tab 1: Auto Analysis
        with analyzer_tabs[0]:
            st.header("🎯 Automated Analysis")
            st.markdown("Click below to run comprehensive analysis on your data.")
            
            if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing your data..."):
                    # Prepare data
                    df_analysis = df.copy()
                    
                    # Handle categorical variables
                    categorical_cols = df_analysis.select_dtypes(include=['object']).columns.tolist()
                    if target_col in categorical_cols:
                        categorical_cols.remove(target_col)
                    
                    # One-hot encode categorical features
                    if categorical_cols:
                        df_analysis = pd.get_dummies(df_analysis, columns=categorical_cols, drop_first=True)
                    
                    # Separate features and target
                    X = df_analysis.drop(columns=[target_col])
                    y = df_analysis[target_col]

                    # 🔧 Handle missing values
                    # Drop rows where target is missing
                    valid_idx = y.notna()
                    X = X.loc[valid_idx]
                    y = y.loc[valid_idx]

                    # Fill missing numeric features with median
                    X = X.fillna(X.median(numeric_only=True))

                    # Fill remaining non-numeric (just in case) with mode
                    for col in X.columns:
                        if X[col].isna().any():
                            X[col].fillna(X[col].mode()[0], inplace=True)
                    
                    # Detect problem type
                    problem_type = detect_problem_type(y)
                    
                    # Store in session state
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.problem_type = problem_type
                    
                    # Run analysis
                    if problem_type == 'regression':
                        results = perform_regression_analysis(X, y)
                    else:
                        results = perform_classification_analysis(X, y)
                    
                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True
                
                st.success("✅ Analysis Complete!")
            
            # Display results if available
            if st.session_state.analysis_done and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                problem_type = st.session_state.problem_type
                
                st.markdown("---")
                
                # Performance Metrics
                st.subheader("📊 Model Performance")
                if problem_type == 'regression':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score (Linear)", f"{results['r2_score']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{results['rmse']:.2f}")
                    with col3:
                        st.metric("R² Score (Random Forest)", f"{results['rf_r2']:.4f}")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy (Logistic)", f"{results['accuracy']:.4f}")
                    with col2:
                        st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                    with col3:
                        st.metric("Accuracy (Random Forest)", f"{results['rf_accuracy']:.4f}")
                
                # Feature Importance
                st.subheader("🎯 Feature Importance Rankings")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🌲 Random Forest Importance**")
                    st.markdown("""
                    *Like a wise council voting on who's most influential - features that help make 
                    better predictions get higher votes!*
                    """)
                    st.dataframe(results['feature_importance'].head(10), use_container_width=True)
                
                with col2:
                    if problem_type == 'regression':
                        st.markdown("**📈 Linear Model Coefficients**")
                        st.markdown("""
                        *Direct impact: how much the target changes when this feature increases by 1 unit.*
                        """)
                        st.dataframe(results['coefficients'].head(10), use_container_width=True)
                    else:
                        st.markdown("**📊 Logistic Coefficients**")
                        st.markdown("""
                        *Log-odds impact: positive = increases probability, negative = decreases it.*
                        """)
                        st.dataframe(results['coefficients'].head(10), use_container_width=True)
                
                # Correlations (for regression)
                if problem_type == 'regression':
                    st.subheader("🔗 Feature Correlations with Target")
                    st.markdown("""
                    *Correlation shows linear relationships: close to 1 or -1 means strong relationship, 
                    close to 0 means weak.*
                    """)
                    st.dataframe(results['correlations'], use_container_width=True)
                
                # Insights
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.subheader("💡 Key Insights")
                
                top_feature = results['feature_importance'].iloc[0]
                st.markdown(f"""
                - **Most Important Feature**: {top_feature['Feature']} (Importance: {top_feature['Importance']:.4f})
                - **Analogy**: Think of this like the star player in a sports team - they have the biggest 
                  impact on the game outcome!
                """)
                
                if problem_type == 'regression':
                    top_corr = results['correlations'].iloc[0]
                    st.markdown(f"""
                    - **Strongest Correlation**: {top_corr['Feature']} (r={top_corr['Correlation']:.3f})
                    - **Interpretation**: This is like a weather pattern - when this feature changes, 
                      the target tends to {'rise' if top_corr['Correlation'] > 0 else 'fall'} predictably.
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Feature Deep Dive
        with analyzer_tabs[1]:
            st.header("🔍 Feature Deep Dive")
            st.markdown("Examine individual features and detect potential issues.")
            
            if st.session_state.analysis_done:
                X = st.session_state.X
                
                # Feature selection
                selected_feature = st.selectbox("Select a feature to analyze:", X.columns)
                
                if selected_feature:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Distribution")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.hist(X[selected_feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
                        ax.set_xlabel(selected_feature)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_feature}')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("📈 Statistics")
                        stats_df = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                            'Value': [
                                f"{X[selected_feature].mean():.2f}",
                                f"{X[selected_feature].median():.2f}",
                                f"{X[selected_feature].std():.2f}",
                                f"{X[selected_feature].min():.2f}",
                                f"{X[selected_feature].max():.2f}",
                                f"{X[selected_feature].skew():.2f}",
                                f"{X[selected_feature].kurtosis():.2f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                
                # Multicollinearity Check
                st.subheader("🔗 Multicollinearity Detection (VIF)")
                st.markdown("""
                **Analogy**: VIF is like checking if students are copying each other's homework. 
                If two features are too similar (VIF > 10), one is redundant!
                """)
                
                if st.button("Calculate VIF"):
                    with st.spinner("Calculating VIF scores..."):
                        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) >= 2:
                            vif_df = calculate_vif(X, numeric_cols)
                            vif_df = vif_df.sort_values('VIF', ascending=False)
                            
                            st.dataframe(vif_df, use_container_width=True)
                            
                            high_vif = vif_df[vif_df['VIF'] > 10]
                            if len(high_vif) > 0:
                                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                                st.warning(f"⚠️ High multicollinearity in {len(high_vif)} features (VIF > 10)")
                                st.markdown("**Recommendation**: Consider removing highly correlated features or using PCA.")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.success("✓ No significant multicollinearity detected!")
            else:
                st.info("👆 Run 'Auto Analysis' first to enable this feature")
        
        # Tab 3: Relationship Explorer
        with analyzer_tabs[2]:
            st.header("🔍 Relationship Explorer")
            st.markdown("""
            **Analogy**: Like examining how temperature affects ice cream sales - visualize how 
            two features relate to each other!
            """)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns to explore relationships")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Select X Variable", numeric_cols, key='x_var')
                with col2:
                    y_var = st.selectbox("Select Y Variable", 
                                         [col for col in numeric_cols if col != x_var], 
                                         key='y_var')
                
                if st.button("📊 Visualize Relationship"):
                    # Calculate correlation
                    temp_df = df[[x_var, y_var]].dropna()
                    corr, p_val = stats.pearsonr(temp_df[x_var], temp_df[y_var])
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[x_var], df[y_var], alpha=0.6, s=50)
                    
                    # Add trend line
                    z = np.polyfit(df[x_var].dropna(), df[y_var].dropna(), 1)
                    p = np.poly1d(z)
                    ax.plot(df[x_var].sort_values(), p(df[x_var].sort_values()), 
                           "r--", linewidth=2, label=f'Trend Line')
                    
                    ax.set_xlabel(x_var, fontsize=12)
                    ax.set_ylabel(y_var, fontsize=12)
                    ax.set_title(f'{x_var} vs {y_var}\nCorrelation: {corr:.3f} (p={p_val:.4f})', 
                               fontsize=14)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Interpretation
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.subheader("📊 Interpretation")
                    
                    if abs(corr) > 0.7:
                        strength = "strong"
                        analogy = "like the relationship between exercise and fitness"
                    elif abs(corr) > 0.4:
                        strength = "moderate"
                        analogy = "like the relationship between study time and grades"
                    else:
                        strength = "weak"
                        analogy = "like the relationship between shoe size and intelligence (barely any!)"
                    
                    direction = "positive" if corr > 0 else "negative"
                    significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
                    
                    st.markdown(f"""
                    - **Correlation Strength**: {strength.capitalize()} {direction} relationship (r={corr:.3f})
                    - **Analogy**: This is {analogy}
                    - **Statistical Significance**: {significance.capitalize()} (p={p_val:.4f})
                    - **Interpretation**: As {x_var} increases, {y_var} tends to {'increase' if corr > 0 else 'decrease'}
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 4: Interaction Tester
        with analyzer_tabs[3]:
            st.header("⚡ Interaction Effect Tester")
            st.markdown("""
            **Analogy**: Like testing if coffee + sugar = more energy than expected. 
            Does combining Feature A with Feature B create a synergistic boost?
            
            *In chemistry, this is like catalysis - two substances together create a bigger reaction 
            than their individual effects combined!*
            """)
            
            if st.session_state.analysis_done:
                X = st.session_state.X
                y = st.session_state.y
                problem_type = st.session_state.problem_type
                
                col1, col2 = st.columns(2)
                with col1:
                    feature1 = st.selectbox("Select Feature 1", X.columns, key='int_f1')
                with col2:
                    feature2 = st.selectbox("Select Feature 2", 
                                           [col for col in X.columns if col != feature1], 
                                           key='int_f2')
                
                if st.button("🧪 Test Interaction"):
                    with st.spinner("Testing interaction effect..."):
                        results = test_interaction(X, y, feature1, feature2, problem_type)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Model Score", f"{results['base_score']:.4f}")
                        with col2:
                            st.metric("With Interaction", f"{results['interaction_score']:.4f}")
                        with col3:
                            st.metric("Improvement", f"{results['improvement_pct']:.2f}%")
                        
                        # Interpretation
                        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                        st.subheader("🔬 Analysis")
                        
                        if results['improvement_pct'] > 2:
                            st.success(f"""
                            ✅ **Strong Interaction Detected!**
                            
                            **Chemistry Analogy**: Like mixing baking soda + vinegar creates a bigger reaction 
                            than either substance alone! These features work together synergistically.
                            
                            Adding **{feature1} × {feature2}** improves the {results['metric_name']} 
                            by **{results['improvement_pct']:.2f}%**.
                            
                            **Recommendation**: Include this interaction term in your final model.
                            """)
                        elif results['improvement_pct'] > 0:
                            st.info(f"""
                            ℹ️ **Weak Interaction**
                            
                            The interaction improves performance by {results['improvement_pct']:.2f}%, 
                            which is marginal. Like salt + pepper - they work together but not dramatically.
                            """)
                        else:
                            st.warning(f"""
                            ❌ **No Beneficial Interaction**
                            
                            Performance decreased by {abs(results['improvement_pct']):.2f}%. 
                            
                            **Analogy**: Like mixing oil + water - they don't blend well together.
                            These features work better independently.
                            """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👆 Run 'Auto Analysis' first to enable interaction testing")
    
    # ==================== VISUALIZER SECTION ====================
    elif main_section == "📊 Visualizer":
        st.markdown('<div class="section-header visualizer-header">📊 Data Visualizer</div>', unsafe_allow_html=True)
        st.markdown("""
        **Think of the Visualizer as an art gallery:** After the lab (Analyzer) diagnoses the data, 
        this is where we create beautiful, insightful visualizations - turning numbers into stories!
        """)
        
        # Create visualizer tabs
        viz_tabs = st.tabs([
            "📈 Distribution Charts",
            "🔥 Correlation Heatmap",
            "📊 Feature Comparisons",
            "🎯 Target Analysis",
            "📉 Model Performance"
        ])
        
        # Tab 1: Distribution Charts
        with viz_tabs[0]:
            st.header("📈 Distribution Visualizations")
            st.markdown("**Analogy**: Like a population pyramid - see the shape and spread of your data!")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            viz_type = st.radio("Choose visualization type:", 
                               ["Histogram", "Box Plot", "Violin Plot", "All Features Overview"])
            
            if viz_type == "All Features Overview":
                plt.close('all')
                st.subheader("📊 All Numeric Features Distribution")
                
                # Create subplots for all numeric features
                n_features = min(len(numeric_cols), 12)  # Limit to 12 for readability
                cols_to_plot = numeric_cols[:n_features]
                
                n_cols = 3
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                axes = axes.flatten() if n_features > 1 else [axes]
                
                for idx, col in enumerate(cols_to_plot):
                    axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
                    axes[idx].set_xlabel('')
                    axes[idx].grid(True, alpha=0.3)
                
                # Hide extra subplots
                for idx in range(n_features, len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                selected_col = st.selectbox("Select feature to visualize:", numeric_cols)
                
                if viz_type == "Histogram":
                    plt.close('all')
                    bins = st.slider("Number of bins:", 10, 100, 30)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df[selected_col].dropna(), bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                    ax.set_xlabel(selected_col, fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'Distribution of {selected_col}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add mean and median lines
                    mean_val = df[selected_col].mean()
                    median_val = df[selected_col].median()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                elif viz_type == "Box Plot":
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    box_plot = ax.boxplot(df[selected_col].dropna(), vert=True, patch_artist=True)
                    box_plot['boxes'][0].set_facecolor('lightblue')
                    box_plot['boxes'][0].set_edgecolor('black')
                    ax.set_ylabel(selected_col, fontsize=12)
                    ax.set_title(f'Box Plot of {selected_col}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **Box Plot Guide:**
                    - Box = middle 50% of data (25th to 75th percentile)
                    - Line in box = median
                    - Whiskers = typical data range
                    - Dots beyond whiskers = potential outliers
                    """)
                    
                elif viz_type == "Violin Plot":
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    parts = ax.violinplot([df[selected_col].dropna()], positions=[0], 
                                         showmeans=True, showmedians=True)
                    ax.set_ylabel(selected_col, fontsize=12)
                    ax.set_title(f'Violin Plot of {selected_col}', fontsize=14, fontweight='bold')
                    ax.set_xticks([])
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **Violin Plot Guide:**
                    - Width = density (how many data points at that value)
                    - Wide parts = many data points
                    - Narrow parts = few data points
                    - Combines box plot info with distribution shape!
                    """)
        
        # Tab 2: Correlation Heatmap
        with viz_tabs[1]:
            st.header("🔥 Correlation Heatmap")
            st.markdown("""
            **Analogy**: Like a friendship matrix in a social network - see which features are best buddies 
            (highly correlated) and which are strangers (uncorrelated)!
            """)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            plt.close('all')
            if len(numeric_cols) >= 2:
                # Correlation method selection
                method = st.selectbox("Correlation method:", ["Pearson", "Spearman"])
                
                # Calculate correlation
                corr_matrix = df[numeric_cols].corr(method=method.lower())
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, linewidths=1, 
                           cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title(f'{method} Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Insights
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.subheader("💡 Correlation Insights")
                
                # Find strongest correlations (excluding diagonal)
                corr_df = (
                    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    .stack()
                    .reset_index()
                )
                corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
                corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                
                st.markdown("**Strongest Correlations:**")
                st.dataframe(corr_df.head(10), use_container_width=True)
                
                st.markdown("""
                **Color Guide:**
                - 🔴 Red = Positive correlation (move together)
                - 🔵 Blue = Negative correlation (move opposite)
                - ⚪ White = No correlation (independent)
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Need at least 2 numeric features for correlation analysis")
        
        # Tab 3: Feature Comparisons
        with viz_tabs[2]:
            st.header("📊 Feature Comparison Charts")
            st.markdown("**Analogy**: Like comparing athletes' stats - see features side by side!")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            chart_type = st.selectbox("Select chart type:", 
                                     ["Bar Chart", "Scatter Plot", "Line Chart", "Pair Plot"])
            
            if chart_type == "Pair Plot":
                plt.close('all')
                st.subheader("🎨 Pair Plot (Feature Relationships)")
                
                # Select subset of features for pair plot
                max_features = min(5, len(numeric_cols))
                selected_features = st.multiselect(
                    "Select features for pair plot (max 5):",
                    numeric_cols,
                    default=numeric_cols[:max_features]
                )
                
                if len(selected_features) >= 2:
                    if st.button("Generate Pair Plot"):
                        with st.spinner("Creating pair plot..."):
                            fig = sns.pairplot(df[selected_features], diag_kind='hist', 
                                             plot_kws={'alpha': 0.6})
                            fig.fig.suptitle('Feature Pair Plot', y=1.02, fontsize=16, fontweight='bold')
                            st.pyplot(fig)
                            
                            st.info("""
                            **How to read:**
                            - Diagonal = distribution of each feature
                            - Off-diagonal = scatter plots showing relationships
                            - Look for patterns, clusters, or linear relationships!
                            """)
                else:
                    st.warning("Select at least 2 features for pair plot")
            
            elif chart_type == "Scatter Plot":
                plt.close('all')
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("X-axis feature:", numeric_cols)
                with col2:
                    y_feature = st.selectbox("Y-axis feature:", 
                                            [c for c in numeric_cols if c != x_feature])
                
                # Optional color by categorical
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                color_by = None
                if categorical_cols:
                    use_color = st.checkbox("Color by category?")
                    if use_color:
                        color_by = st.selectbox("Select category:", categorical_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if color_by:
                    for category in df[color_by].unique():
                        mask = df[color_by] == category
                        ax.scatter(df.loc[mask, x_feature], df.loc[mask, y_feature], 
                                 label=category, alpha=0.6, s=50)
                    ax.legend()
                else:
                    ax.scatter(df[x_feature], df[y_feature], alpha=0.6, s=50, color='steelblue')
                
                ax.set_xlabel(x_feature, fontsize=12)
                ax.set_ylabel(y_feature, fontsize=12)
                ax.set_title(f'{x_feature} vs {y_feature}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif chart_type == "Bar Chart":
                plt.close('all')
                # Group by categorical and show mean of numeric
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        cat_feature = st.selectbox("Categorical feature:", categorical_cols)
                    with col2:
                        num_feature = st.selectbox("Numeric feature (to aggregate):", numeric_cols)
                    
                    # Calculate means
                    means = df.groupby(cat_feature)[num_feature].mean().sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                    ax.set_xlabel(cat_feature, fontsize=12)
                    ax.set_ylabel(f'Mean {num_feature}', fontsize=12)
                    ax.set_title(f'Average {num_feature} by {cat_feature}', 
                               fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("No categorical features available for bar chart")
            
            elif chart_type == "Line Chart":
                plt.close('all')
                st.subheader("📈 Line Chart (Trends)")
                
                # Select features to plot
                features_to_plot = st.multiselect("Select features:", numeric_cols, 
                                                 default=numeric_cols[:3])
                
                if features_to_plot:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for feature in features_to_plot:
                        ax.plot(df.index, df[feature], marker='o', markersize=3, 
                               label=feature, alpha=0.7, linewidth=2)
                    
                    ax.set_xlabel('Index', fontsize=12)
                    ax.set_ylabel('Value', fontsize=12)
                    ax.set_title('Feature Trends', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # Tab 4: Target Analysis
        with viz_tabs[3]:
            st.header("🎯 Target Variable Analysis")
            st.markdown(f"**Analyzing: {target_col}**")
            
            problem_type = detect_problem_type(df[target_col])
            
            if problem_type == 'regression':
                st.markdown("**Type**: Continuous (Regression)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df[target_col].dropna(), bins=30, edgecolor='black', 
                           alpha=0.7, color='coral')
                    ax.set_xlabel(target_col, fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'Distribution of {target_col}', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Box plot
                    fig, ax = plt.subplots(figsize=(8, 5))
                    box_plot = ax.boxplot(df[target_col].dropna(), vert=True, patch_artist=True)
                    box_plot['boxes'][0].set_facecolor('lightcoral')
                    ax.set_ylabel(target_col, fontsize=12)
                    ax.set_title(f'Box Plot of {target_col}', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Statistics
                st.subheader("📊 Target Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', 'Skewness'],
                    'Value': [
                        f"{df[target_col].mean():.2f}",
                        f"{df[target_col].median():.2f}",
                        f"{df[target_col].std():.2f}",
                        f"{df[target_col].min():.2f}",
                        f"{df[target_col].max():.2f}",
                        f"{df[target_col].max() - df[target_col].min():.2f}",
                        f"{df[target_col].skew():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
                
            else:
                st.markdown("**Type**: Categorical (Classification)")
                
                # Value counts
                value_counts = df[target_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    value_counts.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
                    ax.set_xlabel(target_col, fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.set_title(f'Distribution of {target_col}', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
                          startangle=90, colors=sns.color_palette('pastel'))
                    ax.set_title(f'Proportion of {target_col}', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Class distribution
                st.subheader("📊 Class Distribution")
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
        
        # Tab 5: Model Performance
        with viz_tabs[4]:
            st.header("📉 Model Performance Visualization")
            
            if st.session_state.analysis_done and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                problem_type = st.session_state.problem_type
                
                if problem_type == 'regression':
                    st.subheader("🎯 Actual vs Predicted")
                    
                    # Scatter plot of predictions
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(results['y_test'], results['y_pred'], alpha=0.6, s=50)
                    
                    # Perfect prediction line
                    min_val = min(results['y_test'].min(), results['y_pred'].min())
                    max_val = max(results['y_test'].max(), results['y_pred'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 
                           'r--', linewidth=2, label='Perfect Prediction')
                    
                    ax.set_xlabel('Actual Values', fontsize=12)
                    ax.set_ylabel('Predicted Values', fontsize=12)
                    ax.set_title(f'Actual vs Predicted (R² = {results["r2_score"]:.4f})', 
                               fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **How to read:**
                    - Points close to red line = good predictions
                    - Points far from line = prediction errors
                    - **Analogy**: Like darts on a dartboard - closer to center is better!
                    """)
                    
                    # Residual plot
                    st.subheader("📊 Residual Plot")
                    residuals = results['y_test'] - results['y_pred']
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(results['y_pred'], residuals, alpha=0.6, s=50)
                    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                    ax.set_xlabel('Predicted Values', fontsize=12)
                    ax.set_ylabel('Residuals', fontsize=12)
                    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **Residual Plot Guide:**
                    - Residuals = errors (actual - predicted)
                    - Random scatter around zero = good model
                    - Patterns = model missing something
                    - **Analogy**: Like checking if a scale is calibrated - 
                      random errors are okay, systematic bias is not!
                    """)
                
                else:  # Classification
                    st.subheader("🎯 Confusion Matrix")
                    
                    # Confusion matrix heatmap
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                               cmap='Blues', square=True, ax=ax,
                               xticklabels=np.unique(results['y_test']),
                               yticklabels=np.unique(results['y_test']))
                    ax.set_ylabel('Actual', fontsize=12)
                    ax.set_xlabel('Predicted', fontsize=12)
                    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **Confusion Matrix Guide:**
                    - Diagonal = correct predictions
                    - Off-diagonal = mistakes
                    - **Analogy**: Like a student's answer sheet - 
                      diagonal shows correct answers, off-diagonal shows where they got confused!
                    """)
                    
                    # Feature importance comparison
                    st.subheader("🎯 Model Comparison: Feature Importance")
                    
                    # Combine coefficients and RF importance
                    top_n = 10
                    coef_df = results['coefficients'].head(top_n).copy()
                    imp_df = results['feature_importance'].head(top_n).copy()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Coefficients
                    coef_df.plot(x='Feature', y='Abs_Coefficient', kind='barh', 
                                ax=ax1, legend=False, color='steelblue')
                    ax1.set_xlabel('Coefficient Magnitude', fontsize=11)
                    ax1.set_ylabel('')
                    ax1.set_title('Logistic Regression Coefficients', fontsize=12, fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='x')
                    
                    # RF Importance
                    imp_df.plot(x='Feature', y='Importance', kind='barh', 
                               ax=ax2, legend=False, color='coral')
                    ax2.set_xlabel('Importance Score', fontsize=11)
                    ax2.set_ylabel('')
                    ax2.set_title('Random Forest Importance', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("👆 Run 'Auto Analysis' in the Analyzer section first to see model performance visualizations")

else:
    # Landing page
    st.markdown("""
    ## Welcome to Data Detective! 🔍
    
    ### What is this tool?
    **Analogy**: Think of this as your personal data doctor with two departments:
    - **🔬 Analyzer (Lab)**: Runs diagnostic tests on your data
    - **📊 Visualizer (Imaging Center)**: Creates visual scans to see patterns
    
    ### Features:
    
    #### 🔬 Analyzer Section:
    - **Auto Analysis**: One-click comprehensive analysis (like a full health checkup)
    - **Feature Deep Dive**: Examine individual features closely (like specialized tests)
    - **Relationship Explorer**: See how features relate (like checking organ interactions)
    - **Interaction Tester**: Find synergies between features (like drug interaction tests)
    
    #### 📊 Visualizer Section:
    - **Distribution Charts**: See how data is spread (like population demographics)
    - **Correlation Heatmap**: Feature relationship network (like a social network map)
    - **Feature Comparisons**: Side-by-side analysis (like comparing treatments)
    - **Target Analysis**: Focus on what you're predicting (like focusing on symptoms)
    - **Model Performance**: Evaluate prediction accuracy (like treatment success rates)
    
    ### Get Started:
    1. 📁 Upload your CSV or Excel file in the sidebar
    2. 🎯 Select your target variable (what you want to predict)
    3. 🔬 Use the Analyzer to understand relationships
    4. 📊 Use the Visualizer to see beautiful charts
    
    ---
    
    ### Example Use Cases:
    - **Sales Prediction**: What drives revenue? See it in charts!
    - **Customer Churn**: Which factors matter most? Visualize the patterns!
    - **Price Optimization**: How do features interact? Test and visualize!
    """)
    
    # Sample data option
    st.subheader("Or try with sample data:")
    if st.button("🎲 Generate Sample Dataset", type="primary"):
        np.random.seed(42)
        n = 200
        sample_df = pd.DataFrame({
            'price': np.random.uniform(10, 100, n),
            'ad_spend': np.random.uniform(1000, 10000, n),
            'discount_pct': np.random.uniform(0, 30, n),
            'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n),
            'competition': np.random.uniform(1, 10, n)
        })
        
        # Create target with interactions
        sample_df['sales'] = (
            -2.5 * sample_df['price'] +
            0.8 * sample_df['ad_spend'] +
            1.5 * sample_df['discount_pct'] +
            0.002 * sample_df['ad_spend'] * sample_df['discount_pct'] +  # Interaction!
            -150 * sample_df['competition'] +
            np.random.normal(0, 500, n)
        )
        
        st.session_state.df = sample_df
        st.success("✓ Sample dataset loaded! Select a section above to explore.")
        st.rerun()
