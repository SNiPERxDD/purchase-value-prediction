# Purchase-Value Prediction (CPU-Only)

[![Status](https://img.shields.io/badge/Status-Public-brightgreen)](#)
[![Built with](https://img.shields.io/badge/Built%20with-Python%203.10+-blue)](#)
[![R² Score](https://img.shields.io/badge/R²%20Score-0.86-brightgreen)](#)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)](#)

> **End-to-end ML pipeline that predicts customer purchase value from web-session data using a sophisticated two-stage ensemble approach.**

---

## 📋 Table of Contents

- [1. Project Snapshot](#1-project-snapshot)
- [2. What's Inside](#2-whats-inside)
- [3. Project Structure](#3-project-structure)
- [4. How It Works](#4-how-it-works-short)
- [5. Quick Start](#5-run-it)
- [6. Results](#6-results-20-hold-out)
- [7. Next Steps](#7-next-steps-not-done-yet)

---

## 1. Project Snapshot

| Item                        | Notes                                                      |
|-----------------------------|------------------------------------------------------------|
| **Data**                    | 116k × 52 train / 29k × 51 test (CSV, not in repo)       |
| **Target skew**             | Heavy; capped at 99th percentile and log-scaled           |
| **Missing values**          | Up to 96% in ad-related columns                           |
| **Core model**              | XGBoost (`tree_method="hist"`, CPU)                        |
| **Hold-out split (20%)**    | Validation **R² ≈ 0.86**                                   |
| **Dependencies**            | Python 3.8+, NumPy, Pandas, scikit-learn, XGBoost, Seaborn |

---

## 2. What's Inside

| Folder / File      | Purpose                                   |
|--------------------|-------------------------------------------|
| `src/Predictor.py` | Main pipeline – read → train → predict     |
| `src/HyperParams.py` | Offline grid-search for best XGB settings |
| `data/`            | **Sample** CSVs (tiny, anonymised)        |
| `output/`          | Generated `prediction.csv`                |
| `requirements.txt` | Exact library versions                    |

---

## 3. Project Structure

```
purchase-value-prediction/
├── 📊 data/
│   ├── sample_train_data.csv    # Sample training data (included)
│   ├── sample_test_data.csv     # Sample test data (included)
│   ├── train_data.csv           # Full training data (not included)
│   └── test_data.csv            # Full test data (not included)
├── 📁 src/
│   ├── Predictor.py             # Main prediction pipeline
│   └── HyperParams.py           # Hyperparameter tuning
├── 📤 output/
│   ├── sample_prediction.csv    # Sample output format (10 rows)
│   ├── prediction.csv           # Generated predictions (not included)
│   ├── best_params.json         # Optimized hyperparameters (from tuning)
│   └── .gitkeep                 # Directory structure
├── 📋 README.md                 # Project documentation
├── 📊 RESULTS.md                # Detailed results analysis
├── 📦 requirements.txt          # Python dependencies
├── 📄 LICENSE                   # All rights reserved
└── 🚫 .gitignore                # Git ignore rules
```

---

## 4. How It Works (short)

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

## 5. Run It

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
# Place train_data.csv and test_data.csv in the project root or ./data/ folder
# The pipeline will automatically find them

# Run the pipeline

# Step 1: Run hyperparameter search (optional but recommended)
python src/HyperParams.py
# → Saves best parameters to output/best_params.json

# Step 2: Run the prediction pipeline
python src/Predictor.py
# → Automatically loads best parameters if available
# → output/prediction.csv
```

**Performance**: ~5–10 min on modern laptop; RAM < 3 GB

### **Dynamic Parameter Loading**
The pipeline now automatically uses optimized hyperparameters:

1. **HyperParams.py** saves best parameters to `output/best_params.json`
2. **Predictor.py** automatically loads these parameters if available
3. Falls back to sensible defaults if no tuning has been run

This eliminates manual parameter copying and ensures optimal performance.

---

## 6. Results (20% hold-out)

### **Performance Metrics**

| Stage | Metric | Performance |
|-------|--------|-------------|
| **Classifier (RBF SVM)** | Accuracy | **0.993** |
| **Regressor (XGB)** | R² on buyers | **0.93** |
| **Final Ensemble** | R² overall | **0.86** |

* Minor R² variation between local (0.86) and Kaggle (0.86) is expected because of
  * different BLAS / thread back-ends,
  * CPU instruction sets & floating-point determinism,
  * early-stopping selecting a slightly different best iteration, and
  * library version differences.
* No change to data or algorithms—just numerical noise.

### **Runtime Performance**

| Environment                | HyperParams runtime | Predictor runtime |
|----------------------------|--------------------:|------------------:|
| **MacBook Air (M2, 16-GB, CPU-only)** | ~4 min 42 s | ~1 min 21 s |
| **Kaggle "CPU Only" notebook**         | ~15 min 14 s | ~4 min 05 s |

### **Model Comparison**
- **Binary Classification**: SVM (RBF) achieves 99.3% accuracy
- **Regression**: XGBoost achieves 93% R² on buyer subset
- **Ensemble**: Combined approach delivers 0.86 R²

### **System Requirements**

> *Both HyperParams.py and Predictor.py are confirmed to run in **CPU-only** mode.
> No GPU libraries (CUDA, ROCm, Metal) are required.*

**Verification Checklist:**
- ✅ Confirm xgboost reports `tree_method='hist'` or `exact` (not `gpu_hist`)
- ✅ Confirm no GPU-specific libraries in requirements.txt
- ✅ Confirm no torch or tensorflow imports exist

---

## 7. Next Steps (not done yet)

- [ ] **Unit tests** for feature pipeline
- [ ] **SHAP plots** for interpretability  
- [ ] **REST endpoint** for inference
- [ ] **Periodic retraining** script
- [ ] **Model monitoring** and drift detection

---

## 🛠️ Recent Enhancements & Changelog

For a complete, versioned list of improvements—including dynamic parameter management, comprehensive error handling, production logging, and file naming standardization—see **[CHANGELOG.md](./CHANGELOG.md)**.

**Latest Release: v1.0.0** - Major production-readiness improvements with 80% reduction in pipeline failures and automated parameter management.

---

## 📊 Key Features

- ✅ **Two-stage ensemble** (classifier + regressor)
- ✅ **Dynamic parameter loading** (automatic best params from tuning)
- ✅ **Robust missing data handling** (96%+ missing rates)
- ✅ **CPU-optimized** implementation
- ✅ **Production-ready** output format
- ✅ **Comprehensive documentation**

---

## 📄 License

This repository is for academic demonstration purposes only.

All rights to the code and models are reserved by the author. No reuse, reproduction, or redistribution is permitted without explicit permission.

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

</div>
