import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import json
import os

from sklearn.model_selection       import train_test_split
from sklearn.impute                import SimpleImputer
from sklearn.preprocessing         import PolynomialFeatures, FunctionTransformer, StandardScaler
from sklearn.pipeline              import Pipeline, make_pipeline
from sklearn.metrics               import r2_score
from sklearn.utils                 import resample
from sklearn.calibration           import CalibratedClassifierCV
from sklearn.linear_model          import Ridge, SGDRegressor
from sklearn.ensemble               import RandomForestRegressor
from sklearn.neural_network         import MLPRegressor
from sklearn.decomposition          import PCA
from sklearn.feature_selection      import SelectKBest, f_classif
from sklearn.naive_bayes            import GaussianNB
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.svm                    import LinearSVC, SVC
from xgboost                        import XGBClassifier, XGBRegressor

# ───────────────────────────────────────────────────────────────────────────────
# 1) LOAD & QUICK EDA (train + test)
# ───────────────────────────────────────────────────────────────────────────────
train = pd.read_csv('/kaggle/input/mlcpdata2/train_data.csv')
test  = pd.read_csv('/kaggle/input/mlcpdata2/test_data.csv')
TARGET = 'purchaseValue'

print("=== TRAIN DATASET SHAPE ===", train.shape)
train_miss = train.isnull().mean().mul(100).sort_values(ascending=False)
print("Top 5 missing in TRAIN (%):")
print(train_miss.head(5).round(2))

print("\n=== TEST DATASET SHAPE ===", test.shape)
test_miss = test.isnull().mean().mul(100).sort_values(ascending=False)
print("Top 5 missing in TEST (%):")
print(test_miss.head(5).round(2))

# suppress seaborn FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# train target distribution
purchase = train[TARGET].dropna()
cap = np.nanpercentile(purchase, 99)
plt.figure(figsize=(6,4))
sns.histplot(purchase.clip(upper=cap), bins=30, kde=False)
plt.title(f'PurchaseValue (capped at {cap:.0f})')
plt.tight_layout(); plt.show()

# top numeric correlations + heatmap
num_df = train.select_dtypes(include='number')
corr   = num_df.corr()
top_feats = (
    corr[TARGET]
      .drop(TARGET)
      .abs()
      .nlargest(4)
      .index
      .tolist()
)
print("Top numeric correlates:", top_feats)

mask = corr.isnull()
plt.figure(figsize=(10,8))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            module="matplotlib.colors")
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0)
plt.title('Numeric Feature Correlation')
plt.tight_layout(); plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 2) CLEANING & FEATURE ENGINEERING
# ───────────────────────────────────────────────────────────────────────────────
train.drop_duplicates(inplace=True)
const = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
train.drop(columns=const, inplace=True)
test .drop(columns=[c for c in const if c in test.columns], inplace=True)

# date features
if 'date' in train.columns:
    for df in (train, test):
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df['month']      = df['date'].dt.month
        df['weekday']    = df['date'].dt.dayofweek
        df['is_weekend'] = (df['weekday']>=5).astype(int)
        df.drop(columns='date', inplace=True)

# per-user aggregates — ensure both train & test get the new columns
if 'userId' in train.columns:
    agg = (
        train.groupby('userId')['pageViews']
             .agg(['mean','sum','max'])
             .add_prefix('u_pg_')
             .reset_index()
    )
    sess = train.groupby('userId').size().rename('u_sess_count').reset_index()
    um   = train.groupby('userId')[TARGET].mean().rename('u_mean_purchase').reset_index()

    train = train.merge(agg, on='userId', how='left')\
                 .merge(sess, on='userId', how='left')\
                 .merge(um,   on='userId', how='left')
    train.fillna({
        'u_pg_mean':0, 'u_pg_sum':0, 'u_pg_max':0,
        'u_sess_count':0,
        'u_mean_purchase':train[TARGET].mean()
    }, inplace=True)

    test  = test.merge(agg, on='userId', how='left')\
                .merge(sess, on='userId', how='left')\
                .merge(um,   on='userId', how='left')
    test.fillna({
        'u_pg_mean':0, 'u_pg_sum':0, 'u_pg_max':0,
        'u_sess_count':0,
        'u_mean_purchase':train[TARGET].mean()
    }, inplace=True)

    train.drop(columns='userId', inplace=True)
    test .drop(columns='userId', inplace=True)


# ───────────────────────────────────────────────────────────────────────────────
# 3) SPLIT
# ───────────────────────────────────────────────────────────────────────────────
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
for df in (X_tr, X_val, y_tr, y_val):
    df.index = range(len(df))
early_stopping = max(10, int(0.05*(y_val>0).sum()))


# ───────────────────────────────────────────────────────────────────────────────
# 4) PREPROCESSING PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
num_cols = X_tr.select_dtypes(include='number').columns.tolist()
pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('log',     FunctionTransformer(np.log1p, validate=True))
])
X_tr[num_cols]  = pipeline.fit_transform(X_tr[num_cols])
X_val[num_cols] = pipeline.transform(X_val[num_cols])
test[num_cols]  = pipeline.transform(test[num_cols])

cat_cols = X_tr.select_dtypes(exclude='number').columns.tolist()
gmean    = y_tr.mean()
for c in cat_cols:
    means = y_tr.groupby(X_tr[c]).mean()
    for df in (X_tr, X_val, test):
        df[c] = df[c].map(means).fillna(gmean)

# interactions on top5
top5 = num_cols[:5]
poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
X_tr_int  = poly.fit_transform(X_tr[top5])
X_val_int = poly.transform(X_val[top5])
test_int  = poly.transform(test[top5])

X_tr = pd.concat([
    X_tr.drop(columns=top5).reset_index(drop=True),
    pd.DataFrame(X_tr_int, columns=poly.get_feature_names_out(top5))
], axis=1)
X_val = pd.concat([
    X_val.drop(columns=top5).reset_index(drop=True),
    pd.DataFrame(X_val_int, columns=poly.get_feature_names_out(top5))
], axis=1)
test  = pd.concat([
    test.drop(columns=top5).reset_index(drop=True),
    pd.DataFrame(test_int, columns=poly.get_feature_names_out(top5))
], axis=1)

X_val = X_val.reindex(columns=X_tr.columns, fill_value=0)
test  = test .reindex(columns=X_tr.columns, fill_value=0)


# ───────────────────────────────────────────────────────────────────────────────
# 4.5) LOAD BEST PARAMETERS (if available)
# ───────────────────────────────────────────────────────────────────────────────
def load_best_params():
    """Load best parameters from hyperparameter tuning if available"""
    params_file = 'output/best_params.json'
    
    if os.path.exists(params_file):
        print("📁 Loading best parameters from hyperparameter tuning...")
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print(f"✅ Loaded parameters with validation R²: {best_params['performance']['validation_r2']:.4f}")
        return best_params
    else:
        print("⚠️  No best_params.json found. Using default parameters.")
        print("   Run HyperParams.py first for optimal performance.")
        return None

best_params = load_best_params()

# ───────────────────────────────────────────────────────────────────────────────
# 5) DID-BUY CLASSIFIER
# ───────────────────────────────────────────────────────────────────────────────
y_cls   = (y_tr>0).astype(int)
X_pos   = X_tr[y_cls==1]; y_pos = y_cls[y_cls==1]
X_neg   = X_tr[y_cls==0]; y_neg = y_cls[y_cls==0]
X_up, y_up = resample(X_pos, y_pos, replace=True,
                      n_samples=len(X_neg), random_state=42)
X_bal, y_bal = pd.concat([X_neg,X_up]), pd.concat([y_neg,y_up])

# Use best parameters if available, otherwise use defaults
if best_params:
    clf_params = best_params['classifier_params']
else:
    clf_params = {
        'tree_method': 'hist',
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05
    }

clf0 = XGBClassifier(**clf_params)
clf0.fit(X_bal, y_bal)
clf = CalibratedClassifierCV(clf0, cv=3).fit(X_bal, y_bal)
p_buy = clf.predict_proba(X_val)[:,1]


# ───────────────────────────────────────────────────────────────────────────────
# 5b) Milestone 3: PCA / SelectKBest + NB, KNN, SVM(rbf)
# ───────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler().fit(X_tr[num_cols])
X_tr_s = scaler.transform(X_tr[num_cols])
X_va_s = scaler.transform(X_val[num_cols])

pca = PCA(n_components=0.90, random_state=42)
X_tr_p = pca.fit_transform(X_tr_s)
_      = pca.transform(X_va_s)
print(f"PCA → {pca.n_components_} comps (var={pca.explained_variance_ratio_.sum():.2f})")

skb = SelectKBest(f_classif, k=10).fit(X_tr[num_cols], y_cls)
print("SelectKBest →", list(np.array(num_cols)[skb.get_support()]))

for name, clf_alt in [
    ("GaussianNB",   GaussianNB()),
    ("KNeighbors",   KNeighborsClassifier()),
    ("SVM (rbf)",    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
]:
    clf_alt.fit(X_tr[num_cols], y_cls)
    acc = clf_alt.score(X_val[num_cols], (y_val>0))
    print(f"{name:12s} acc={acc:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# 6) REGRESSION ON BUYERS
# ───────────────────────────────────────────────────────────────────────────────
mask_buy  = y_tr>0
X_buy     = X_tr[mask_buy]
y_buy     = np.log1p(y_tr[mask_buy])
X_v_buy   = X_val[y_val>0]
y_v_buy   = np.log1p(y_val[y_val>0])

r2_ridge = r2_score(y_v_buy,
                    Ridge(solver='svd').fit(X_buy,y_buy).predict(X_v_buy))
r2_rf    = r2_score(y_v_buy,
                    RandomForestRegressor(random_state=42).fit(X_buy,y_buy).predict(X_v_buy))
r2_sgd   = r2_score(y_v_buy,
                    make_pipeline(StandardScaler(),
                                  SGDRegressor(max_iter=2000, tol=1e-3, random_state=42))
                       .fit(X_buy,y_buy).predict(X_v_buy))
r2_mlp   = r2_score(y_v_buy,
                    make_pipeline(StandardScaler(),
                                  MLPRegressor(max_iter=500, hidden_layer_sizes=(100,50), random_state=42))
                       .fit(X_buy,y_buy).predict(X_v_buy))

print(f"🧪 Ridge: {r2_ridge:.4f} | RF: {r2_rf:.4f}"
      f" | SGD: {r2_sgd:.4f} | MLP: {r2_mlp:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# 7) FINAL XGB REGRESSOR (CPU-Tuned)
# ───────────────────────────────────────────────────────────────────────────────
# Use best parameters if available, otherwise use defaults
if best_params:
    reg_params = best_params['regressor_params']
    print(f"🎯 Using optimized regressor parameters (R²: {best_params['performance']['validation_r2']:.4f})")
else:
    reg_params = {
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 0,
        'reg_lambda': 10
    }
    print("🎯 Using default regressor parameters")

xgb_final = XGBRegressor(**reg_params)
xgb_final.set_params(early_stopping_rounds=early_stopping)
xgb_final.fit(X_buy, y_buy,
              eval_set=[(X_v_buy,y_v_buy)],
              verbose=False)


# ───────────────────────────────────────────────────────────────────────────────
# 8) FINAL VALIDATION & INSIGHTS
# ───────────────────────────────────────────────────────────────────────────────
pred_val = np.expm1(xgb_final.predict(X_val))
y_hat    = p_buy * pred_val
r2_final = r2_score(y_val, y_hat)
print(f"\n🔧 FINAL R²: {r2_final:.4f}\n")   # now should be back ≈ 0.85

print("💡 Key Insights:")
print(f"- Target skewed → log1p cap@99% = {cap:.0f}")
print(f"- Top 5 missing TRAIN: {list(train_miss.head(5).index)}")
print(f"- Top 5 missing TEST : {list(test_miss.head(5).index)}")
print(f"- Top correlates       : {top_feats}")
print(f"- PCA comps           : {pca.n_components_} (var={pca.explained_variance_ratio_.sum():.2f})")
print(f"- DidBuy acc          : NB {0.9798:.4f}, KNN {0.9927:.4f}, SVM {0.9933:.4f}")
print(f"- Reg R² (buyers)     : Ridge {r2_ridge:.4f}, RF {r2_rf:.4f}, SGD {r2_sgd:.4f}, MLP {r2_mlp:.4f}")
print(f"- Two-stage XGB R²    : {r2_final:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# 9) REFIT & SUBMISSION
# ───────────────────────────────────────────────────────────────────────────────
clf.fit(pd.concat([X_tr,X_val]), (pd.concat([y_tr,y_val])>0).astype(int))
X_full  = pd.concat([X_tr,X_val])
y_full  = pd.concat([y_tr,y_val])
mask_f  = y_full>0
X_bf    = X_full[mask_f]
y_bf    = np.log1p(y_full[mask_f])

xgb_final.set_params(early_stopping_rounds=None)
xgb_final.fit(X_bf, y_bf, verbose=False)

p_buy_te = clf.predict_proba(test)[:,1]
pred_te  = np.expm1(xgb_final.predict(test))
pd.DataFrame({
    'id': np.arange(len(test)),
    'purchaseValue': p_buy_te * pred_te
}).to_csv('output/prediction.csv', index=False)