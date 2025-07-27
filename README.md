# Purchase-Value Prediction (CPU-Only)

[![Built with Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![R² (hold-out)](https://img.shields.io/badge/Validation%20R²-0.864-lightgreen)](#)

End-to-end pipeline that predicts a customer's **`purchaseValue`** from web-session data using a two-stage approach:

1. **Classifier** – predicts whether the user will buy.  
2. **Regressor** – estimates the amount (log scale) for buyers.  

The product of the two gives the final prediction.

---

## 1 Project Snapshot

| Item                        | Notes                                                      |
|-----------------------------|------------------------------------------------------------|
| Data                        | 116k × 52 train / 29k × 51 test (CSV, not in repo)       |
| Target skew                 | Heavy; capped at 99th percentile and log-scaled           |
| Missing values              | Up to 96% in ad-related columns                           |
| Core model                  | XGBoost (`tree_method="hist"`, CPU)                        |
| Hold-out split (20%)        | Validation **R² ≈ 0.864**                                  |
| Dependencies                | Python 3.8+, NumPy, Pandas, scikit-learn, XGBoost, Seaborn |

---

## 2 What's Inside

| Folder / file      | Purpose                                   |
|--------------------|-------------------------------------------|
| `src/Predictor.py` | main pipeline – read → train → predict     |
| `src/HyperParams.py` | offline grid-search that found best XGB settings |
| `data/`            | **sample** CSVs (tiny, anonymised)        |
| `output/`          | generated `submission.csv`                |
| `requirements.txt` | exact library versions                    |

---

## 3 How It Works (short)

1. **Cleaning**  
   * drop duplicates and constant columns  
   * fill numeric NaNs with median; log-transform numerics  
2. **Feature engineering**  
   * date parts (`month`, `weekday`, `is_weekend`)  
   * per-user aggregates (`u_pg_mean`, `u_sess_count`, …)  
   * target-encode high-cardinality categoricals  
   * polynomial interactions on top-5 numeric features  
3. **Model training**  
   * calibr. XGBClassifier on balanced data → `p(buy)`  
   * XGBRegressor on log target for buyers → `log(value)`  
4. **Prediction**  `ŷ = p(buy) × exp(log(value))`

---

## 4 Run It

```bash
# clone & install
git clone https://github.com/SNiPERxDD/purchase-value-prediction.git
cd purchase-value-prediction
pip install -r requirements.txt

# put the real CSVs into ./data (keep names: train_data.csv, test_data.csv)

# train + predict
python src/Predictor.py
# → output/submission.csv
```
CPU time ~5–10 min on a modern laptop; RAM under 3 GB.

---

## 5 Results (20% hold-out)

| Stage | Metric |
|-------|--------|
| Classifier (RBF SVM) | Accuracy 0.993 |
| Regressor (XGB) | R² ≈ 0.93 on buyers |
| Final ensemble | R² ≈ 0.864 |

Numbers may vary slightly on different hardware/Python seeds.

---

## 6 Next Steps (not done yet)

- Unit tests for feature pipeline
- SHAP plots for interpretability  
- Simple REST endpoint for inference
- Periodic retraining script
