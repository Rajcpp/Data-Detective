# 🔍 Data Detective

**A full-featured interactive data analysis and modeling lab built with Streamlit.**  
Upload a dataset. Detect patterns. Test interactions. Visualize relationships. Evaluate models — all in one place.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 What Is Data Detective?

Data Detective is an interactive web application that combines:

- 📊 Exploratory Data Analysis (EDA)
- 🤖 Automated model selection (Regression vs Classification)
- 🌲 Feature importance ranking
- 🔗 Correlation and relationship exploration
- ⚡ Interaction effect testing
- 📈 Model diagnostics and performance visualization
- 📉 Multicollinearity detection (VIF)

It functions as both a statistical lab and a visual analytics studio.

---

## 🧠 Core Capabilities

### 🔬 Analyzer (Statistical Lab)

- Automatic detection of problem type:
  - Regression (continuous target)
  - Classification (categorical target)
- Linear Regression
- Logistic Regression
- Random Forest models
- Feature importance ranking
- Correlation analysis with p-values
- Multicollinearity detection (VIF)
- Interaction effect testing
- Model performance metrics:
  - R², RMSE (Regression)
  - Accuracy, ROC-AUC (Classification)

---

### 📊 Visualizer (Interactive Data Studio)

- Distribution charts:
  - Histograms
  - Box plots
  - Violin plots
- Correlation heatmaps (Pearson & Spearman)
- Scatter plots with trend lines
- Pair plots
- Feature comparison charts
- Target variable analysis
- Model performance plots:
  - Actual vs Predicted
  - Residual plots
  - Confusion matrix

---

## ⚡ Interaction Testing

Detects whether two features combined improve model performance.

Example:

```
Base Model:
Sales ~ Price + Discount
R² = 0.65

With Interaction:
Sales ~ Price + Discount + (Price × Discount)
R² = 0.78  (+20% improvement)
```

The application automatically calculates performance improvement and provides interpretation guidance.

---

## 📂 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/data-detective.git
cd data-detective
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

The app will launch at:

```
http://localhost:8501
```

---

## 📁 Supported Data

- CSV (.csv)
- Excel (.xlsx)
- Tabular datasets
- Numeric predictors required for analysis
- Categorical target supported for classification

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Modeling | Scikit-learn |
| Statistical Testing | SciPy |
| Multicollinearity | Statsmodels (VIF) |
| Visualization | Matplotlib, Seaborn |

---

## 📊 Workflow

1. Upload dataset  
2. Select target variable  
3. Run automated analysis  
4. Explore feature importance  
5. Examine correlations and multicollinearity  
6. Test feature interactions  
7. Visualize distributions and relationships  
8. Evaluate model performance  

---

## 🎯 Use Cases

- Business analytics (sales drivers, churn prediction)
- Healthcare outcome modeling
- Real estate pricing analysis
- Marketing campaign optimization
- Academic research
- Exploratory modeling before production pipelines

---

## 🚀 Deployment

### Streamlit Cloud

1. Push the repository to GitHub
2. Connect via Streamlit Cloud
3. Deploy

---

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

Build and run:

```bash
docker build -t data-detective .
docker run -p 8501:8501 data-detective
```

---

## 🔮 Roadmap

- SHAP explainability integration
- Automated PDF report export
- Time-series support
- Multi-model comparison dashboard
- Code export (Python pipeline generator)

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a feature branch  
3. Submit a pull request  

---

## 📄 License

MIT License

---

## ℹ️ Disclaimer

This tool is designed for exploratory analysis and rapid insight generation.  
For production use, implement proper validation, feature engineering, and deployment best practices.
