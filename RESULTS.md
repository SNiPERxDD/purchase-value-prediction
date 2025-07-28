# 📊 Detailed Results & Output Analysis

## 🎯 **Executive Summary**

This document contains the complete output and results from running both `HyperParams.py` and `Predictor.py` scripts, providing detailed insights into the model performance, data characteristics, and optimization process.

---

## 🔧 **Hyperparameter Tuning Results (HyperParams.py)**

### **Coarse Tuning Phase**
```
Coarse Best → subs=0.6, col=0.8, mcw=10 → R²=0.8580
```

**Parameters Tested:**
- **subsample**: [0.6, 0.8, 1.0]
- **colsample_bytree**: [0.6, 0.8, 1.0]  
- **min_child_weight**: [1, 5, 10]

**Best Configuration:**
- subsample: 0.6
- colsample_bytree: 0.8
- min_child_weight: 10
- Performance: R² = 0.8580

### **Refined Tuning Phase**
```
Refine Best → lr=0.1, depth=8, gamma=0, lambda=10 → R²=0.8637
```

**Parameters Tested:**
- **learning_rate**: [0.01, 0.03, 0.05, 0.1]
- **max_depth**: [6, 8, 10]
- **gamma**: [0, 1, 5]
- **reg_lambda**: [0, 1, 5, 10]

**Best Configuration:**
- learning_rate: 0.1
- max_depth: 8
- gamma: 0
- reg_lambda: 10
- Performance: R² = 0.8637

### **Final Validation**
```
🔧 FINAL Validation R²: 0.8636635060527729
```

---

## 📈 **Main Pipeline Results (Predictor.py)**

### **Dataset Overview**
```
=== TRAIN DATASET SHAPE === (116023, 52)
=== TEST DATASET SHAPE === (29006, 51)
```

### **Data Quality Analysis**

**Missing Data - Training Set:**
- `trafficSource.adContent`: 97.45%
- `trafficSource.adwordsClickInfo.isVideoAd`: 96.31%
- `trafficSource.adwordsClickInfo.page`: 96.31%
- `trafficSource.adwordsClickInfo.adNetworkType`: 96.31%
- `trafficSource.adwordsClickInfo.slot`: 96.31%

**Missing Data - Test Set:**
- `trafficSource.adContent`: 97.38%
- `trafficSource.adwordsClickInfo.adNetworkType`: 96.18%
- `trafficSource.adwordsClickInfo.isVideoAd`: 96.18%
- `trafficSource.adwordsClickInfo.page`: 96.18%
- `trafficSource.adwordsClickInfo.slot`: 96.18%

### **Feature Analysis**

**Top Numeric Correlates with Target:**
1. `totalHits`
2. `pageViews`
3. `sessionNumber`
4. `trafficSource.adwordsClickInfo.page`

**Dimensionality Reduction:**
- **PCA Components**: 7 (93% variance explained)
- **SelectKBest Features**: 10 selected features
  - `sessionNumber`
  - `pageViews`
  - `totalHits`
  - `weekday`
  - `is_weekend`
  - `u_pg_mean`
  - `u_pg_sum`
  - `u_pg_max`
  - `u_sess_count`
  - `u_mean_purchase`

### **Model Performance Comparison**

#### **Binary Classification Models (Will Buy?)**
| Model | Accuracy |
|-------|----------|
| **SVM (RBF)** | **0.9933** |
| K-Nearest Neighbors | 0.9927 |
| Gaussian Naive Bayes | 0.9798 |

#### **Regression Models (Purchase Value - Buyers Only)**
| Model | R² Score |
|-------|----------|
| **Random Forest** | **0.9339** |
| Ridge Regression | 0.9163 |
| SGD Regressor | 0.9111 |
| MLP Neural Network | 0.9063 |

### **Final Ensemble Performance**
```
🔧 FINAL R²: 0.8637
```

---

## 💡 **Key Insights Summary**

### **Data Characteristics**
- **Target Distribution**: Heavily right-skewed with 99th percentile cap at 483,870,000
- **Missing Data Pattern**: Ad-related features have 96-97% missing rates
- **Feature Importance**: User behavior metrics (`totalHits`, `pageViews`) are strongest predictors

### **Model Performance**
- **Two-Stage Ensemble**: Achieves R² = 0.8637 on validation set
- **Classifier Performance**: SVM achieves 99.33% accuracy for purchase prediction
- **Regression Performance**: Random Forest performs best (R² = 0.9339) for value prediction

### **Optimization Insights**
- **Subsampling**: 0.6 subsample rate optimal for preventing overfitting
- **Feature Selection**: 7 PCA components capture 93% of variance
- **Regularization**: L2 regularization (lambda=10) improves generalization

---

## 📊 **Visualizations Generated**

The pipeline generates two key visualizations:

1. **Target Distribution Plot**: Histogram of purchase values (capped at 99th percentile)
2. **Correlation Heatmap**: Feature correlation matrix with NaN masking

These visualizations help understand:
- The extreme skewness of the target variable
- Feature relationships and potential multicollinearity
- Data quality issues and missing value patterns

---

## 🎯 **Performance Benchmarks**

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Final R²** | **0.8637** | **Excellent** (>0.75 threshold) |
| Classifier Accuracy | 0.9933 | Excellent (>0.95) |
| PCA Variance Explained | 93% | Good (>90%) |
| Missing Data Handling | Robust | Handles 96%+ missing rates |

**Conclusion**: The model significantly outperforms the minimum threshold of 0.75 R², demonstrating excellent predictive power for customer purchase value prediction.

---

## 🛠️ Recent Enhancements

For detailed information about recent improvements including dynamic parameter management, comprehensive error handling, and production-ready logging, see **[CHANGELOG.md](./CHANGELOG.md)**.

The latest v1.0.0 release includes major production-readiness improvements that enhance pipeline reliability while maintaining the excellent model performance documented above. 