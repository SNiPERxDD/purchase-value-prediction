import pandas as pd
import numpy as np
from sklearn.model_selection   import train_test_split
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import PolynomialFeatures
from sklearn.metrics           import r2_score
from sklearn.utils             import resample
from sklearn.calibration       import CalibratedClassifierCV
from xgboost                   import XGBClassifier, XGBRegressor

# ─── 1) load & clean ─────────────────────────────────────────────────
train = pd.read_csv('/kaggle/input/mlcpdata2/train_data.csv')
test  = pd.read_csv('/kaggle/input/mlcpdata2/test_data.csv')
TARGET = 'purchaseValue'

train.drop_duplicates(inplace=True)
const = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
train.drop(columns=const, inplace=True)
test .drop(columns=[c for c in const if c in test.columns], inplace=True)

# ─── 2) date features ─────────────────────────────────────────────────
if 'date' in train.columns:
    for df in (train, test):
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df['month']      = df['date'].dt.month
        df['weekday']    = df['date'].dt.dayofweek
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df.drop(columns='date', inplace=True)

# ─── 3) per-user aggregates ────────────────────────────────────────────
if 'userId' in train.columns:
    agg_pg = (
        train.groupby('userId')['pageViews']
             .agg(['mean','sum','max'])
             .add_prefix('u_pg_')
             .reset_index()
    )
    sess = train.groupby('userId').size().rename('u_sess_count').reset_index()
    um   = train.groupby('userId')[TARGET].mean().rename('u_mean_purchase').reset_index()

    train = (
        train
        .merge(agg_pg, on='userId', how='left')
        .merge(sess,   on='userId', how='left')
        .merge(um,     on='userId', how='left')
    )
    train.fillna({
        'u_pg_mean':0,'u_pg_sum':0,'u_pg_max':0,
        'u_sess_count':0,
        'u_mean_purchase': train[TARGET].mean()
    }, inplace=True)

    test = (
        test
        .merge(agg_pg, on='userId', how='left')
        .merge(sess,   on='userId', how='left')
        .merge(um,     on='userId', how='left')
    )
    test.fillna({
        'u_pg_mean':0,'u_pg_sum':0,'u_pg_max':0,
        'u_sess_count':0,
        'u_mean_purchase': train[TARGET].mean()
    }, inplace=True)

    train.drop(columns='userId', inplace=True)
    test .drop(columns='userId', inplace=True)

# ─── 4) train/val split ─────────────────────────────────────────────────
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
for df in (X_tr, X_val, y_tr, y_val):
    df.reset_index(drop=True, inplace=True)

early_stopping = max(10, int(0.05 * (y_val > 0).sum()))

# ─── 5) impute & log1p numerics ───────────────────────────────────────
num_cols = X_tr.select_dtypes(include='number').columns
imp = SimpleImputer(strategy='median').fit(X_tr[num_cols])
for df in (X_tr, X_val, test):
    df[num_cols] = imp.transform(df[num_cols])
    df[num_cols] = np.log1p(df[num_cols])

# ─── 6) target‐encode categoricals ────────────────────────────────────
cat_cols   = X_tr.select_dtypes(exclude='number').columns
global_mean = y_tr.mean()
for c in cat_cols:
    means = y_tr.groupby(X_tr[c]).mean()
    for df in (X_tr, X_val, test):
        df[c] = df[c].map(means).fillna(global_mean)

# ─── 7) interactions on top-5 numerics ────────────────────────────────
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

# ─── 8) Stage-1: balanced classifier ─────────────────────────────────
y_cls = (y_tr > 0).astype(int)
X_pos, y_pos = X_tr[y_cls==1], y_cls[y_cls==1]
X_neg, y_neg = X_tr[y_cls==0], y_cls[y_cls==0]
X_pos_up, y_pos_up = resample(
    X_pos, y_pos, replace=True,
    n_samples=len(X_neg), random_state=42
)
X_bal = pd.concat([X_neg, X_pos_up], axis=0)
y_bal = pd.concat([y_neg, y_pos_up], axis=0)

clf0 = XGBClassifier(
    tree_method='hist',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42, verbosity=0,
    n_estimators=200, max_depth=6, learning_rate=0.05
)
clf0.fit(X_bal, y_bal)
clf  = CalibratedClassifierCV(clf0, cv=3).fit(X_bal, y_bal)
p_buy = clf.predict_proba(X_val)[:,1]

# ─── 9) Stage-2: buyer-only split ─────────────────────────────────────
mask_buy  = y_tr > 0
X_buy     = X_tr[mask_buy]
y_buy     = np.log1p(y_tr[mask_buy])
X_v_buy   = X_val[y_val > 0]
y_v_buy   = np.log1p(y_val[y_val > 0])

# ─── 10) coarse tune (CPU-only) ───────────────────────────────────────
best_r2, (bs, bc, bm) = -np.inf, (0.8, 0.8, 10)
for subs in [0.6, 0.8, 1.0]:
    for cols in [0.6, 0.8, 1.0]:
        for mcw in [1, 5, 10]:
            m = XGBRegressor(
                tree_method='hist',
                random_state=42, verbosity=0,
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                subsample=subs, colsample_bytree=cols,
                min_child_weight=mcw
            )
            m.set_params(early_stopping_rounds=early_stopping)
            m.fit(X_buy, y_buy,
                  eval_set=[(X_v_buy, y_v_buy)],
                  verbose=False)
            pred = np.expm1(m.predict(X_val))
            r = r2_score(y_val, p_buy * pred)
            if r > best_r2:
                best_r2, (bs, bc, bm) = r, (subs, cols, mcw)

print(f"Coarse Best → subs={bs}, col={bc}, mcw={bm} → R²={best_r2:.4f}")

# ─── 11) refine tune: lr, depth, gamma, reg_lambda ────────────────────
best_r2_2, best_cfg = best_r2, (0.05, 6, 0, 1)
for lr in [0.01, 0.03, 0.05, 0.1]:
    for md in [6, 8, 10]:
        for gamma in [0, 1, 5]:
            for lam in [0, 1, 5, 10]:
                m = XGBRegressor(
                    tree_method='hist',
                    random_state=42, verbosity=0,
                    n_estimators=1000,
                    max_depth=md, learning_rate=lr,
                    subsample=bs, colsample_bytree=bc,
                    min_child_weight=bm,
                    gamma=gamma, reg_lambda=lam
                )
                m.set_params(early_stopping_rounds=early_stopping)
                m.fit(X_buy, y_buy,
                      eval_set=[(X_v_buy, y_v_buy)],
                      verbose=False)
                pred = np.expm1(m.predict(X_val))
                r = r2_score(y_val, p_buy * pred)
                if r > best_r2_2:
                    best_r2_2, best_cfg = r, (lr, md, gamma, lam)

print(f"Refine Best → lr={best_cfg[0]}, depth={best_cfg[1]}, gamma={best_cfg[2]}, lambda={best_cfg[3]} → R²={best_r2_2:.4f}")

# ─── 12) final fit & report ───────────────────────────────────────────
xgb_final = XGBRegressor(
    tree_method='hist',
    random_state=42, verbosity=0,
    n_estimators=1000,
    max_depth=best_cfg[1],
    learning_rate=best_cfg[0],
    subsample=bs, colsample_bytree=bc,
    min_child_weight=bm,
    gamma=best_cfg[2], reg_lambda=best_cfg[3]
)
xgb_final.set_params(early_stopping_rounds=early_stopping)
xgb_final.fit(X_buy, y_buy,
              eval_set=[(X_v_buy, y_v_buy)],
              verbose=False)

pred_val = np.expm1(xgb_final.predict(X_val))
y_hat    = p_buy * pred_val
print("🔧 FINAL Validation R²:", r2_score(y_val, y_hat))