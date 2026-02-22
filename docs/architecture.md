# 🏗️ Data Detective – Architecture Overview

This document explains the internal architecture, data flow, modeling pipeline, and design decisions behind Data Detective.

---

## 1. High-Level Architecture

Data Detective is built as a single-page interactive Streamlit application with modular logical separation between:

- Data ingestion
- Preprocessing
- Model analysis
- Statistical testing
- Visualization
- Interaction testing
- Session state management

The system is organized into logical layers rather than physical modules.
```code
User Upload
↓
Data Loader
↓
Preprocessing Pipeline
↓
Problem Type Detection
↓
Model Training (Regression / Classification)
↓
Statistical Diagnostics
↓
Visualization Engine
↓
UI Rendering
```
---

---

## 2. Core Components

### 2.1 Data Layer

Responsible for:

- Uploading CSV / Excel files
- Loading data into pandas DataFrame
- Caching loaded data
- Storing dataset in Streamlit session state

Key responsibilities:
- File validation
- Basic preview
- Target variable selection

---

### 2.2 Preprocessing Layer

Handles:

- Automatic detection of categorical features
- One-hot encoding (drop_first=True)
- Missing value handling:
  - Target rows with NaN removed
  - Numeric features filled with median
  - Remaining missing values filled with mode
- Feature/target separation

Design decision:
Keep preprocessing lightweight and transparent for exploratory analysis rather than production-grade pipelines.

---

### 2.3 Problem Type Detection

Automatically determines:

- Regression → Continuous numeric target
- Classification → Categorical or low-cardinality integer target

Logic:
- Object dtype → Classification
- Integer with <= 10 unique values → Classification
- Otherwise → Regression

---

## 3. Modeling Layer

### 3.1 Regression Pipeline

Models:
- Linear Regression
- Random Forest Regressor

Outputs:
- R² score
- RMSE
- Coefficients
- Feature importance
- Pearson correlations (with p-values)

Additional diagnostics:
- Residuals
- Actual vs Predicted scatter plot

---

### 3.2 Classification Pipeline

Models:
- Logistic Regression
- Random Forest Classifier

Outputs:
- Accuracy
- ROC-AUC
- Confusion matrix
- Coefficients
- Feature importance

Supports:
- Binary classification
- Multiclass classification (OVR strategy)

---

## 4. Statistical Analysis Layer

### 4.1 Correlation Testing

For regression:
- Pearson correlation
- P-value computation
- Significance flag (p < 0.05)

### 4.2 Multicollinearity Detection

Variance Inflation Factor (VIF) computed using:

statsmodels.stats.outliers_influence.variance_inflation_factor

Thresholds:
- VIF > 5 → Moderate multicollinearity
- VIF > 10 → High multicollinearity warning

---

## 5. Interaction Testing Engine

Purpose:
Detect whether adding an interaction term improves model performance.

Process:
1. Train base model
2. Create interaction feature (Feature1 × Feature2)
3. Train model with interaction
4. Compare performance
5. Compute percentage improvement

Metric used:
- Regression → R²
- Classification → Accuracy

Improvement formula:

Improvement % = ((Score_with_interaction - Base_score) / Base_score) × 100

---

## 6. Visualization Engine

Built using:
- Matplotlib
- Seaborn

Visual modules include:

- Histograms
- Box plots
- Violin plots
- Correlation heatmaps
- Scatter plots with trend lines
- Pair plots
- Confusion matrix heatmap
- Residual plots
- Feature comparison bar charts

Design choice:
Keep visualizations static but clear for performance and stability.

---

## 7. Session State Management

Streamlit session state stores:

- df
- target_col
- X
- y
- problem_type
- analysis_results
- analysis_done

Purpose:
- Avoid recomputation
- Maintain state across tab changes
- Improve performance with caching

---

## 8. Caching Strategy

Uses:

@st.cache_data

Applied to:
- Data loading
- Problem type detection
- Regression analysis
- Classification analysis

Benefits:
- Faster repeated runs
- Reduced recomputation
- Improved UI responsiveness

---

## 9. Design Philosophy

Data Detective is designed for:

- Exploratory analysis
- Rapid insight generation
- Educational clarity
- Business diagnostics

It is NOT designed for:

- Production-grade ML pipelines
- Hyperparameter tuning
- Model deployment
- Automated feature engineering

---

## 10. Limitations

- No cross-validation
- No advanced feature scaling
- Limited preprocessing customization
- Not optimized for very large datasets (>100k rows)
- No persistent storage

---

## 11. Future Architectural Improvements

Planned enhancements:

- Modularization into:
  - analysis/
  - visualization/
  - utils/
- SHAP explainability integration
- Cross-validation framework
- Exportable model pipeline
- PDF report generation
- Time-series module

---

## 12. Summary

Data Detective follows a layered architecture:

Data → Preprocessing → Detection → Modeling → Diagnostics → Visualization

It prioritizes clarity, interpretability, and usability over complexity.

The system is intentionally structured to provide maximum analytical insight with minimal user effort.