# Purchase Value Prediction – ML Pipeline

![Status](https://img.shields.io/badge/Status-Private-brightred)
![Built with](https://img.shields.io/badge/Built%20with-Python%203.10-blue)
![R² Score](https://img.shields.io/badge/R²%20Score-0.8637-brightgreen)

> **An end-to-end machine learning pipeline for predicting customer purchase value from multi-session web behavior using a sophisticated two-stage ensemble with XGBoost, achieving 86.37% R² accuracy.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Technical Details](#technical-details)
- [Future Work](#future-work)

---

## 🎯 Overview

This production-ready ML pipeline predicts customer purchase values from web behavior data using a sophisticated **two-stage ensemble approach**. The system handles highly skewed data, extensive missing values (96%+), and delivers robust predictions through careful feature engineering and model optimization.

**Key Achievements:**
- ✅ **R² Score: 0.8637** (exceeds 75% threshold by 15%)
- ✅ **CPU-optimized** implementation (no GPU required)
- ✅ **Robust missing data handling** (96%+ missing rates)
- ✅ **Comprehensive model comparison** (5+ algorithms tested)
- ✅ **Production-ready** with proper validation and documentation

---

## 🚀 Core Features

| Module | Description |
|--------|-------------|
| **Data Preprocessing** | Handles 96%+ missing data, removes duplicates, and normalizes features |
| **Feature Engineering** | User aggregates, date features, target encoding, and polynomial interactions |
| **Two-Stage Ensemble** | Binary classifier (will buy?) + Regressor (purchase value) |
| **Hyperparameter Tuning** | Systematic grid search with early stopping and cross-validation |
| **Model Comparison** | Ridge, RandomForest, SGD, MLP, XGBoost with detailed benchmarking |
| **Visualization** | Target distribution plots and correlation heatmaps |
| **Production Output** | Clean submission format with proper error handling |

---

## 📊 Dataset

### **Statistics**
- **Training Set**: 116,023 rows × 52 columns *(not included in repo)*
- **Test Set**: 29,006 rows × 51 columns *(not included in repo)*
- **Sample Files**: `data/sample_train_data.csv`, `data/sample_test_data.csv` *(included for demo/testing)*
- **Target Variable**: `purchaseValue` (continuous, heavily right-skewed)

### **Data Quality Insights**
- **Missing Data**: Ad-related features have 96-97% missing rates
- **Top Correlates**: `totalHits`, `pageViews`, `sessionNumber`
- **Target Distribution**: 99th percentile cap at 483,870,000

### **Feature Categories**
- **User Behavior**: `pageViews`, `totalHits`, `sessionNumber`
- **Traffic Source**: Ad network data, referral paths, keywords
- **Device/Geo**: Browser, OS, country, mobile flags
- **Temporal**: Date features, weekday patterns, weekend effects

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Preprocessing  │───▶│ Feature Engine. │
│   (116k rows)   │    │  (Missing,      │    │ (User agg,      │
│                 │    │   Duplicates)   │    │  Date features) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Final Prediction│◀───│ Two-Stage       │◀───│ Model Training  │
│ (R² = 0.8637)   │    │ Ensemble        │    │ (XGBoost +      │
│                 │    │ (Classifier +   │    │  Calibration)   │
└─────────────────┘    │  Regressor)     │    └─────────────────┘
                       └─────────────────┘
```

### **Two-Stage Pipeline**
1. **Stage 1**: Binary classifier predicts purchase probability
2. **Stage 2**: Regressor predicts log-purchase value (buyers only)
3. **Final**: `prediction = P(buy) × exp(log_value)`

---

## 🏆 Results

### **Performance Metrics**

| Metric | Value | Status |
|--------|-------|--------|
| **Final R² Score** | **0.8637** | 🟢 Excellent |
| Classifier Accuracy | 0.9933 | 🟢 Excellent |
| PCA Variance Explained | 93% | 🟢 Good |
| Missing Data Handling | Robust | 🟢 Excellent |

### **Model Comparison**

**Binary Classification (Will Buy?)**
- **SVM (RBF)**: 99.33% accuracy ⭐
- **K-Nearest Neighbors**: 99.27% accuracy
- **Gaussian Naive Bayes**: 97.98% accuracy

**Regression (Purchase Value)**
- **Random Forest**: 93.39% R² ⭐
- **Ridge Regression**: 91.63% R²
- **SGD Regressor**: 91.11% R²
- **MLP Neural Network**: 90.63% R²

### **Hyperparameter Optimization**
- **Best Configuration**: `lr=0.1, depth=8, gamma=0, lambda=10`
- **Subsampling**: 0.6 (prevents overfitting)
- **Early Stopping**: Dynamic based on validation set size

---

## ⚡ Quick Start

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Installation**
```bash
# 1. Clone repository
git clone https://github.com/SNiPERxDD/purchase-value-prediction.git
cd purchase-value-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data files
# Place train_data.csv and test_data.csv in the data/ folder (not included in repo)
# Or use the provided sample files for testing/demo:
#   data/sample_train_data.csv
#   data/sample_test_data.csv

# 4. Run the pipeline
python src/Predictor.py
```

### **Expected Output**
- **Generated predictions**: `output/submission.csv` (29,007 predictions)
- **Sample format**: `output/sample_submission.csv` (10 sample predictions)
- **Memory usage**: < 3 GB
- **Processing time**: ~5-10 minutes (CPU only)

---

## 📁 Repository Structure

```
purchase-value-prediction/
├── 📊 data/
│   ├── sample_train_data.csv    # Sample training data (included)
│   ├── sample_test_data.csv     # Sample test data (included)
│   ├── train_data.csv           # Full training data (not included)
│   └── test_data.csv            # Full test data (not included)
├── 📁 src/
│   ├── Predictor.py            # Main prediction pipeline
│   └── HyperParams.py          # Hyperparameter tuning
├── 📤 output/
│   ├── submission.csv          # Generated predictions (29k rows)
│   ├── sample_submission.csv   # Sample output format (10 rows)
│   └── .gitkeep               # Directory structure
├── 📋 README.md               # Project documentation
├── 📊 RESULTS.md              # Detailed results analysis
├── 📦 requirements.txt        # Python dependencies
└── 🚫 .gitignore             # Git ignore rules
```

---

## 🔧 Technical Details

### **Feature Engineering Pipeline**
1. **Data Cleaning**: Remove duplicates and constant columns
2. **Date Features**: Month, weekday, weekend flags
3. **User Aggregates**: Cross-session behavior patterns
4. **Transformations**: Log1p for skewed numerics
5. **Target Encoding**: High-cardinality categoricals
6. **Interactions**: Polynomial features on top-5 numerics

### **Model Architecture**
- **Classifier**: XGBoost with calibration (balanced sampling)
- **Regressor**: XGBoost with early stopping
- **Validation**: 20% holdout with proper stratification
- **Optimization**: Grid search with cross-validation

### **Performance Optimizations**
- **CPU-friendly**: `tree_method='hist'` for XGBoost
- **Memory efficient**: Streaming data processing
- **Scalable**: Handles 100k+ rows efficiently

---

## 🚀 Future Work

- **Deep Learning**: RNN/Transformer for sequential patterns
- **Interpretability**: SHAP analysis for feature importance
- **Real-time**: API endpoints for live predictions
- **Monitoring**: Model drift detection and retraining
- **A/B Testing**: Automated model comparison framework

---

## 📈 Key Insights

### **Business Value**
- **High Accuracy**: 86.37% R² enables reliable revenue forecasting
- **Scalable**: CPU-only implementation reduces infrastructure costs
- **Robust**: Handles real-world data quality issues gracefully

### **Technical Excellence**
- **Production Ready**: Proper validation, error handling, documentation
- **Reproducible**: Complete environment setup and version control
- **Maintainable**: Clean code structure with comprehensive comments

---



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Contact

- **GitHub**: [@SNiPERxDD](https://github.com/SNiPERxDD)
- **Project Link**: [https://github.com/SNiPERxDD/purchase-value-prediction](https://github.com/SNiPERxDD/purchase-value-prediction)

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

</div> 
