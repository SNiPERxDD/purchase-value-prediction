# Purchase-Value Prediction (CPU-Only)

[![Status](https://img.shields.io/badge/Status-Public-brightgreen)](#)
[![Built with](https://img.shields.io/badge/Built%20with-Python%203.10+-blue)](#)
[![R² Score](https://img.shields.io/badge/R²%20Score-0.864-brightgreen)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

> **End-to-end ML pipeline that predicts customer purchase value from web-session data using a sophisticated two-stage ensemble approach.**

---

## 📋 Table of Contents

- [Project Snapshot](#1-project-snapshot)
- [What's Inside](#2-whats-inside)
- [How It Works](#3-how-it-works-short)
- [Quick Start](#4-run-it)
- [Results](#5-results-20-hold-out)
- [Next Steps](#6-next-steps-not-done-yet)

---

## 1 Project Snapshot

| Item                        | Notes                                                      |
|-----------------------------|------------------------------------------------------------|
| **Data**                    | 116k × 52 train / 29k × 51 test (CSV, not in repo)       |
| **Target skew**             | Heavy; capped at 99th percentile and log-scaled           |
| **Missing values**          | Up to 96% in ad-related columns                           |
| **Core model**              | XGBoost (`tree_method="hist"`, CPU)                        |
| **Hold-out split (20%)**    | Validation **R² ≈ 0.864**                                  |
| **Dependencies**            | Python 3.8+, NumPy, Pandas, scikit-learn, XGBoost, Seaborn |

---

## 2 What's Inside

| Folder / File      | Purpose                                   |
|--------------------|-------------------------------------------|
| `src/Predictor.py` | Main pipeline – read → train → predict     |
| `src/HyperParams.py` | Offline grid-search for best XGB settings |
| `data/`            | **Sample** CSVs (tiny, anonymised)        |
| `output/`          | Generated `submission.csv`                |
| `requirements.txt` | Exact library versions                    |

---

## 3 How It Works (short)

### **Pipeline Overview**
```
Raw Data → Preprocessing → Feature Engineering → Two-Stage Model → Predictions
```

### **Step-by-Step Process**
1. **Cleaning**  
   * Drop duplicates and constant columns  
   * Fill numeric NaNs with median; log-transform numerics  
2. **Feature Engineering**  
   * Date parts (`month`, `weekday`, `is_weekend`)  
   * Per-user aggregates (`u_pg_mean`, `u_sess_count`, …)  
   * Target-encode high-cardinality categoricals  
   * Polynomial interactions on top-5 numeric features  
3. **Model Training**  
   * Calibrated XGBClassifier on balanced data → `p(buy)`  
   * XGBRegressor on log target for buyers → `log(value)`  
4. **Prediction**  `ŷ = p(buy) × exp(log(value))`

---

## 4 Run It

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Quick Start**
```bash
# Clone & install
git clone https://github.com/SNiPERxDD/purchase-value-prediction.git
cd purchase-value-prediction
pip install -r requirements.txt

# Add your data files
# Place train_data.csv and test_data.csv in ./data/ folder

# Run the pipeline
python src/Predictor.py
# → output/submission.csv
```

**Performance**: ~5–10 min on modern laptop; RAM < 3 GB

---

## 5 Results (20% hold-out)

### **Performance Metrics**

| Stage | Metric | Performance |
|-------|--------|-------------|
| **Classifier (RBF SVM)** | Accuracy | **0.993** |
| **Regressor (XGB)** | R² on buyers | **0.93** |
| **Final Ensemble** | R² overall | **0.864** |

> **Note**: Numbers may vary slightly on different hardware/Python seeds.

### **Model Comparison**
- **Binary Classification**: SVM (RBF) achieves 99.3% accuracy
- **Regression**: XGBoost achieves 93% R² on buyer subset
- **Ensemble**: Combined approach delivers 86.4% R²

---

## 6 Next Steps (not done yet)

- [ ] **Unit tests** for feature pipeline
- [ ] **SHAP plots** for interpretability  
- [ ] **REST endpoint** for inference
- [ ] **Periodic retraining** script
- [ ] **Model monitoring** and drift detection

---

## 📊 Key Features

- ✅ **Two-stage ensemble** (classifier + regressor)
- ✅ **Robust missing data handling** (96%+ missing rates)
- ✅ **CPU-optimized** implementation
- ✅ **Production-ready** output format
- ✅ **Comprehensive documentation**

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

</div>
