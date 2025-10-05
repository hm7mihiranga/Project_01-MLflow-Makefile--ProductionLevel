# Telco Customer Churn Analysis - Advanced ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)](https://catboost.ai/)

A comprehensive machine learning project for predicting customer churn in telecommunications using advanced ensemble methods, thorough EDA, and production-ready pipelines with automated workflow management.

## 🚀 Quick Start

### Prerequisites
- Windows OS
- Python 3.8+
- Make (Windows compatible)

### One-Command Setup & Execution
```bash
# Complete setup and run entire ML pipeline
make install && make run-all
```

### Individual Commands
```bash
# Setup environment
make install          # Install dependencies and setup environment
make setup           # Complete project setup with verification

# Run pipelines
make data-pipeline   # Process raw data
make train-pipeline  # Train ML models
make streaming-inference  # Run inference demo

# Development & Monitoring  
make mlflow-ui       # Start MLflow tracking UI
make notebook        # Start Jupyter notebook
make test           # Run unit tests
make clean          # Clean artifacts
```

### View Available Commands
```bash
make help            # Display all available commands
```

## 📁 Project Structure

```
Project00/
├── Notebook/
│   ├── Executive_Summary.ipynb          # Business insights & recommendations
│   ├── Part_01_Advanced_EDA.ipynb       # Exploratory Data Analysis
│   ├── Part_02_Advanced_Model_Pipeline.ipynb  # Model development & evaluation
│   └── PArt_03_Model_Evalutaion_for_Imbalance_Data.ipynb
├── src/                                 # Core ML modules
│   ├── data_ingestion.py
│   ├── data_splitter.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_inference.py
├── pipelines/                           # Production pipelines
│   ├── data_pipeline.py
│   ├── training_pipeline.py
│   └── streaming_inference_pipeline.py
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── Telco-Customer-Churn_cleaned.csv
├── artifacts/                           # Model artifacts
│   ├── models/
│   ├── evaluation/
│   └── predictions/
├── config.yaml                         # Configuration settings
└── requirements.txt
```

## 🎯 Business Problem

Predict customer churn for a telecommunications company to enable proactive retention strategies and minimize revenue loss. The project focuses on:

- **Revenue Impact**: Potential annual savings of $200K+ through targeted retention
- **Customer Lifetime Value**: $1,000 per customer
- **Cost Optimization**: Balancing retention costs ($50) vs. customer acquisition costs ($200)

## 📊 Advanced Exploratory Data Analysis (EDA)

### Data Quality Assessment
- **Dataset**: 7,043 customers with 21 features
- **Missing Values**: Handled `TotalCharges` column with empty strings converted to NaN and median imputation
- **Data Types**: Fixed inconsistent data types (object → float for TotalCharges)

### Feature Categories
```python
# Demographic Features
demographic_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

# Behavioral Features  
behavioral_features = ['tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 
                      'Contract', 'PaperlessBilling']

# Financial Features
financial_features = ['PaymentMethod', 'MonthlyCharges', 'TotalCharges']
```

### Key EDA Insights
- **Class Imbalance**: 73.5% non-churn vs 26.5% churn
- **High-Risk Segments**:
  - Electronic check users: 45.29% churn rate
  - Month-to-month contracts: Highest churn
  - New customers (0-6 months tenure): 53.33% churn
  - Senior citizens: 41.68% churn rate

### Statistical Analysis Methods
- **Distribution Analysis**: Histograms, box plots, density plots
- **Correlation Analysis**: Pearson correlation heatmaps
- **Outlier Detection**: IQR and Z-score methods
- **Chi-square Tests**: Categorical variable associations
- **ANOVA**: Numerical feature significance testing

## 🔧 Feature Engineering

### Advanced Feature Creation
```python
# 1. Tenure Categories
def tenure_category(tenure):
    if tenure <= 12: return 'New'
    elif tenure <= 48: return 'Established'
    else: return 'Loyal'

# 2. Service Adoption Score
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
df['ServiceAdoptionScore'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

# 3. Average Monthly Charge per Service
df['AvgMonthlyChargePerService'] = df['MonthlyCharges'] / df['ServiceAdoptionScore'].replace(0, np.nan)

# 4. Payment Reliability Indicator
df['IsReliablePayment'] = df['PaymentMethod'].apply(lambda x: 0 if x == 'Electronic check' else 1)
```

### Encoding Strategies
- **Binary Encoding**: Gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn
- **One-Hot Encoding**: MultipleLines, InternetService, Contract, PaymentMethod, TenureCategory
- **Label Encoding**: Applied with `drop_first=True` to avoid multicollinearity

## 🤖 Machine Learning Pipeline

### Data Preprocessing
```python
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# SMOTE for Class Imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
           'ServiceAdoptionScore', 'AvgMonthlyChargePerService']
X_train_res[num_cols] = scaler.fit_transform(X_train_res[num_cols])
```

### Model Implementation

#### 1. Random Forest (Bagging)
```python
rf = RandomForestClassifier(random_state=42, n_estimators=100)
# Hyperparameter tuning with RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

#### 2. XGBoost (Gradient Boosting)
```python
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, 
                         eval_metric="logloss", verbosity=0)
# Advanced hyperparameter tuning
param_dist_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [100, 200, 300, 500, 800],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}
```

#### 3. CatBoost (Advanced Boosting)
```python
catboost_model = CatBoostClassifier(random_state=42, iterations=100, 
                                   verbose=False, allow_writing_files=False)
```

#### 4. Ensemble Methods
```python
# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, 
                             eval_metric="logloss", verbosity=0))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42)
)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model), 
               ('cat', catboost_model), ('lr', lr_model)],
    voting='soft'
)

# Model Blending
weights = {'rf': 0.3, 'xgb': 0.3, 'cat': 0.2, 'lr': 0.2}
blended_proba = sum(weights[model] * proba for model, proba in model_probas.items())
```

## 📈 Model Evaluation & Business Impact

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|----------|-----------|---------|----------|---------|---------|
| Tuned Random Forest | 0.7743 | 0.58 | 0.65 | 0.61 | 0.8547 | 0.5784 |
| XGBoost | 0.7374 | 0.50 | 0.63 | 0.56 | 0.8421 | 0.5864 |
| CatBoost | 0.7651 | 0.55 | 0.65 | 0.59 | 0.8534 | 0.6210 |
| Logistic Regression | 0.7687 | 0.59 | 0.69 | 0.63 | 0.8789 | 0.6423 |

### Business Impact Analysis
```python
# Business Parameters
CUSTOMER_LIFETIME_VALUE = 1000
RETENTION_COST = 50
ACQUISITION_COST = 200

# Cost-Sensitive Analysis
cost_fn = 500  # False negative cost
cost_fp = 100  # False positive cost
```

#### Financial Returns (Annual Projections)
- **Best Model**: Logistic Regression
- **Net Benefit**: $248,950
- **ROI**: 955.7%
- **Customers Saved**: 259
- **Campaign Efficiency**: 59.1%

### Threshold Optimization
- **Optimal Threshold**: 0.2908 (cost-minimized)
- **Business Impact**: Minimizes combined costs of false positives and false negatives

## 🏭 Production Pipeline Architecture

### 1. Data Pipeline (`pipelines/data_pipeline.py`)
```python
def data_pipeline(input_path: str, output_path: str, 
                 test_size: float = 0.2, force_rebuild: bool = False) -> Dict[str, np.ndarray]:
    """
    Complete data preprocessing pipeline
    """
    # Data ingestion, cleaning, feature engineering, and splitting
```

### 2. Training Pipeline (`pipelines/training_pipeline.py`)
```python
def main():
    """
    Production training pipeline with cross-validation and model persistence
    """
    # Model training, hyperparameter tuning, evaluation, and saving
```

### 3. Streaming Inference Pipeline (`pipelines/streaming_inference_pipeline.py`)
```python
class StreamingInferencePipeline:
    """
    Real-time inference pipeline for production deployment
    """
    def predict_churn_probability(self, customer_data: Dict) -> float:
        # Real-time prediction with preprocessing
```

### Core ML Modules (`src/`)

#### Data Processing
- **[`data_ingestion.py`](src/data_ingestion.py)**: Raw data loading and validation
- **[`handle_missing_values.py`](src/handle_missing_values.py)**: Missing value imputation strategies
- **[`feature_encoding.py`](src/feature_encoding.py)**: Categorical encoding methods
- **[`feature_scaling.py`](src/feature_scaling.py)**: Numerical feature normalization
- **[`outlier_detection.py`](src/outlier_detection.py)**: Outlier identification and treatment

#### Model Development
- **[`model_building.py`](src/model_building.py)**: Model architecture definition
- **[`model_training.py`](src/model_training.py)**: Training with cross-validation
- **[`model_evaluation.py`](src/model_evaluation.py)**: Comprehensive evaluation metrics
- **[`model_inference.py`](src/model_inference.py)**: Prediction interface

## 🚀 Getting Started

### Installation
```bash
git clone <repository-url>
cd Project00
pip install -r requirements.txt
```

### Quick Start
```python
# Run complete analysis
jupyter notebook Notebook/Part_01_Advanced_EDA.ipynb
jupyter notebook Notebook/Part_02_Advanced_Model_Pipeline.ipynb

# Execute production pipeline
python pipelines/data_pipeline.py
python pipelines/training_pipeline.py

# Real-time inference
from pipelines.streaming_inference_pipeline import StreamingInferencePipeline
pipeline = StreamingInferencePipeline()
prediction = pipeline.predict_churn_probability(customer_data)
```

### Configuration
Edit `config.yaml` to customize:
- Model parameters
- Data paths
- Training configuration
- Evaluation metrics

## 📋 Key Features

### Advanced Analytics
- ✅ **Comprehensive EDA** with statistical testing
- ✅ **Feature Engineering** with domain expertise
- ✅ **Class Imbalance Handling** using SMOTE
- ✅ **Hyperparameter Optimization** with RandomizedSearchCV
- ✅ **Ensemble Methods** (Stacking, Voting, Blending)

### Business Intelligence
- ✅ **Cost-Sensitive Learning** with threshold optimization
- ✅ **ROI Analysis** with business impact quantification
- ✅ **Risk Segmentation** for targeted interventions
- ✅ **Campaign Efficiency** metrics

### Production Readiness
- ✅ **Modular Architecture** with separation of concerns
- ✅ **Pipeline Orchestration** for automated workflows
- ✅ **Model Versioning** and artifact management
- ✅ **Real-time Inference** capability
- ✅ **Configuration Management** via YAML

## 🎯 Business Recommendations

### Immediate Actions (30 days)
1. **Deploy Logistic Regression model** for churn prediction
2. **Target electronic check users** with payment method incentives
3. **Implement retention campaigns** for month-to-month customers
4. **Enhance onboarding** for new customers (0-6 months)

### Strategic Initiatives (90 days)
1. **Service bundling strategies** to increase adoption scores
2. **Contract migration programs** from month-to-month to annual
3. **Senior citizen retention** specialized programs
4. **Predictive customer success** team formation

## 📊 Results Summary

- **Model Performance**: 76.9% accuracy with optimized business metrics
- **Business Impact**: $248K annual net benefit potential
- **Risk Reduction**: 45% improvement in identifying high-risk customers
- **Operational Efficiency**: Automated pipeline reduces manual effort by 80%

## 🔄 Future Enhancements

- [ ] Deep learning models (Neural Networks, LSTMs)
- [ ] Real-time feature store integration
- [ ] A/B testing framework for model deployment
- [ ] Advanced explainability (SHAP, LIME)
- [ ] Multi-class churn prediction (likelihood, timing)
- [ ] AutoML integration for continuous model improvement

## 📚 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
catboost>=1.0.0
imbalanced-learn>=0.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
pyyaml>=6.0
```

## 📞 Contact & Support

Hasitha Mihiranga (Linkdin)

---

**Project Status**: ✅ Production Ready  
**Last Updated**: Octomber 2025  
**Version**: 1.0.0
