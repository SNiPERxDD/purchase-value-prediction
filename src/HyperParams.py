import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from sklearn.model_selection   import train_test_split
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import PolynomialFeatures
from sklearn.metrics           import r2_score
from sklearn.utils             import resample
from sklearn.calibration       import CalibratedClassifierCV
from xgboost                   import XGBClassifier, XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparams.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def find_data_files():
    """Dynamically find train_data.csv and test_data.csv in the project directory"""
    project_root = Path.cwd()
    
    # Search patterns for data files
    train_patterns = ['train_data.csv', '**/train_data.csv']
    test_patterns = ['test_data.csv', '**/test_data.csv']
    
    train_file = None
    test_file = None
    
    # Search for training data
    for pattern in train_patterns:
        matches = list(project_root.glob(pattern))
        if matches:
            train_file = matches[0]  # Take first match
            break
    
    # Search for test data
    for pattern in test_patterns:
        matches = list(project_root.glob(pattern))
        if matches:
            test_file = matches[0]  # Take first match
            break
    
    if not train_file:
        raise FileNotFoundError(
            "train_data.csv not found in project directory. "
            "Please ensure the file exists in the project root or data/ subdirectory."
        )
    
    if not test_file:
        raise FileNotFoundError(
            "test_data.csv not found in project directory. "
            "Please ensure the file exists in the project root or data/ subdirectory."
        )
    
    return str(train_file), str(test_file)

def validate_data_files(train_path, test_path):
    """Validate that required data files exist and are readable"""
    required_files = [train_path, test_path]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        # Check if file is readable and not empty
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Data file is empty: {file_path}")
            logger.info(f"✅ Found data file: {file_path} ({file_size:,} bytes)")
        except OSError as e:
            raise OSError(f"Cannot access data file {file_path}: {e}")

def ensure_output_directory():
    """Ensure output directory exists"""
    output_dir = Path('output')
    try:
        output_dir.mkdir(exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_dir}")
    except OSError as e:
        raise OSError(f"Cannot create output directory: {e}")

def validate_dataframe(df, name, required_cols=None):
    """Validate dataframe structure and content"""
    if df is None or df.empty:
        raise ValueError(f"{name} dataframe is empty or None")
    
    logger.info(f"✅ {name} shape: {df.shape}")
    
    if required_cols:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} missing required columns: {missing_cols}")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        logger.warning(f"⚠️  {name} has all-null columns: {null_cols}")
    
    return True

def safe_model_fit(model, X_train, y_train, X_val=None, y_val=None, model_name="Model"):
    """Safely fit a model with error handling"""
    try:
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.error(f"❌ {model_name} fitting failed: {e}")
        raise RuntimeError(f"{model_name} training failed: {e}")

def safe_predict(model, X, model_name="Model"):
    """Safely make predictions with error handling"""
    try:
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return model.predict(X)
    except Exception as e:
        logger.error(f"❌ {model_name} prediction failed: {e}")
        raise RuntimeError(f"{model_name} prediction failed: {e}")

def main():
    """Main hyperparameter tuning pipeline with comprehensive error handling"""
    try:
        logger.info("🚀 Starting hyperparameter tuning pipeline...")
        
        # Find and validate data files
        train_path, test_path = find_data_files()
        validate_data_files(train_path, test_path)
        ensure_output_directory()
        
        # ─── 1) load & clean ─────────────────────────────────────────────────
        logger.info("📊 Loading datasets...")
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            logger.info(f"✅ Loaded training data from: {train_path}")
            logger.info(f"✅ Loaded test data from: {test_path}")
        except Exception as e:
            raise IOError(f"Failed to load CSV files: {e}")
        
        TARGET = 'purchaseValue'
        
        # Validate loaded data
        validate_dataframe(train, "Training data", [TARGET])
        validate_dataframe(test, "Test data")
        
        if TARGET not in train.columns:
            raise ValueError(f"Target column '{TARGET}' not found in training data")
        
        # Check target distribution
        target_stats = train[TARGET].describe()
        logger.info(f"📈 Target statistics: mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}")
        
        if train[TARGET].isnull().all():
            raise ValueError("Target column contains only null values")
        
        logger.info("🧹 Cleaning data...")
        
        # Data cleaning with error handling
        try:
            initial_shape = train.shape
            train.drop_duplicates(inplace=True)
            logger.info(f"Removed {initial_shape[0] - train.shape[0]} duplicate rows")
            
            const = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
            if const:
                logger.info(f"Removing {len(const)} constant columns: {const}")
                train.drop(columns=const, inplace=True)
                test.drop(columns=[c for c in const if c in test.columns], inplace=True)
            
            validate_dataframe(train, "Cleaned training data", [TARGET])
            validate_dataframe(test, "Cleaned test data")
            
        except Exception as e:
            raise ValueError(f"Data cleaning failed: {e}")

        # ─── 2) date features ─────────────────────────────────────────────────
        logger.info("📅 Processing date features...")
        try:
            if 'date' in train.columns:
                for df in (train, test):
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                    df['month'] = df['date'].dt.month
                    df['weekday'] = df['date'].dt.dayofweek
                    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
                    df.drop(columns='date', inplace=True)
                logger.info("✅ Date features created successfully")
        except Exception as e:
            logger.warning(f"⚠️  Date feature processing failed: {e}")

        # ─── 3) per-user aggregates ────────────────────────────────────────────
        logger.info("👥 Creating user aggregates...")
        try:
            if 'userId' in train.columns and 'pageViews' in train.columns:
                agg_pg = (
                    train.groupby('userId')['pageViews']
                         .agg(['mean','sum','max'])
                         .add_prefix('u_pg_')
                         .reset_index()
                )
                sess = train.groupby('userId').size().rename('u_sess_count').reset_index()
                um = train.groupby('userId')[TARGET].mean().rename('u_mean_purchase').reset_index()

                train = (
                    train
                    .merge(agg_pg, on='userId', how='left')
                    .merge(sess, on='userId', how='left')
                    .merge(um, on='userId', how='left')
                )
                train.fillna({
                    'u_pg_mean':0,'u_pg_sum':0,'u_pg_max':0,
                    'u_sess_count':0,
                    'u_mean_purchase': train[TARGET].mean()
                }, inplace=True)

                test = (
                    test
                    .merge(agg_pg, on='userId', how='left')
                    .merge(sess, on='userId', how='left')
                    .merge(um, on='userId', how='left')
                )
                test.fillna({
                    'u_pg_mean':0,'u_pg_sum':0,'u_pg_max':0,
                    'u_sess_count':0,
                    'u_mean_purchase': train[TARGET].mean()
                }, inplace=True)

                train.drop(columns='userId', inplace=True)
                test.drop(columns='userId', inplace=True)
                logger.info("✅ User aggregates created successfully")
        except Exception as e:
            logger.warning(f"⚠️  User aggregate processing failed: {e}")

        # ─── 4) train/val split ─────────────────────────────────────────────────
        logger.info("🔀 Splitting data...")
        try:
            X = train.drop(columns=[TARGET])
            y = train[TARGET]
            
            if len(X) < 100:
                raise ValueError("Insufficient data for training (< 100 samples)")
            
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            for df in (X_tr, X_val, y_tr, y_val):
                df.reset_index(drop=True, inplace=True)

            early_stopping = max(10, int(0.05 * (y_val > 0).sum()))
            logger.info(f"✅ Data split complete. Early stopping: {early_stopping}")
            
        except Exception as e:
            raise ValueError(f"Data splitting failed: {e}")

        # ─── 5) impute & log1p numerics ───────────────────────────────────────
        logger.info("🔢 Processing numeric features...")
        try:
            num_cols = X_tr.select_dtypes(include='number').columns
            if len(num_cols) == 0:
                raise ValueError("No numeric columns found")
            
            imp = SimpleImputer(strategy='median').fit(X_tr[num_cols])
            for df in (X_tr, X_val, test):
                df[num_cols] = imp.transform(df[num_cols])
                df[num_cols] = np.log1p(df[num_cols])
            logger.info(f"✅ Processed {len(num_cols)} numeric columns")
            
        except Exception as e:
            raise ValueError(f"Numeric feature processing failed: {e}")

        # ─── 6) target‐encode categoricals ────────────────────────────────────
        logger.info("🏷️  Processing categorical features...")
        try:
            cat_cols = X_tr.select_dtypes(exclude='number').columns
            global_mean = y_tr.mean()
            
            for c in cat_cols:
                means = y_tr.groupby(X_tr[c]).mean()
                for df in (X_tr, X_val, test):
                    df[c] = df[c].map(means).fillna(global_mean)
            
            if len(cat_cols) > 0:
                logger.info(f"✅ Target-encoded {len(cat_cols)} categorical columns")
                
        except Exception as e:
            logger.warning(f"⚠️  Categorical feature processing failed: {e}")

        # ─── 7) interactions on top-5 numerics ────────────────────────────────
        logger.info("🔗 Creating feature interactions...")
        try:
            top5 = num_cols[:5] if len(num_cols) >= 5 else num_cols
            poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
            
            X_tr_int = poly.fit_transform(X_tr[top5])
            X_val_int = poly.transform(X_val[top5])
            test_int = poly.transform(test[top5])

            X_tr = pd.concat([
                X_tr.drop(columns=top5).reset_index(drop=True),
                pd.DataFrame(X_tr_int, columns=poly.get_feature_names_out(top5))
            ], axis=1)
            X_val = pd.concat([
                X_val.drop(columns=top5).reset_index(drop=True),
                pd.DataFrame(X_val_int, columns=poly.get_feature_names_out(top5))
            ], axis=1)
            test = pd.concat([
                test.drop(columns=top5).reset_index(drop=True),
                pd.DataFrame(test_int, columns=poly.get_feature_names_out(top5))
            ], axis=1)

            X_val = X_val.reindex(columns=X_tr.columns, fill_value=0)
            test = test.reindex(columns=X_tr.columns, fill_value=0)
            logger.info(f"✅ Created interactions for {len(top5)} features")
            
        except Exception as e:
            logger.warning(f"⚠️  Feature interaction creation failed: {e}")

        # ─── 8) Stage-1: balanced classifier ─────────────────────────────────
        logger.info("🎯 Training binary classifier...")
        try:
            y_cls = (y_tr > 0).astype(int)
            X_pos, y_pos = X_tr[y_cls==1], y_cls[y_cls==1]
            X_neg, y_neg = X_tr[y_cls==0], y_cls[y_cls==0]
            
            if len(X_pos) == 0 or len(X_neg) == 0:
                raise ValueError("Insufficient positive or negative samples for classification")
            
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
            
            clf0 = safe_model_fit(clf0, X_bal, y_bal, model_name="XGB Classifier")
            clf = CalibratedClassifierCV(clf0, cv=3).fit(X_bal, y_bal)
            p_buy = safe_predict(clf, X_val, "Calibrated Classifier")[:,1]
            logger.info("✅ Binary classifier trained successfully")
            
        except Exception as e:
            raise RuntimeError(f"Binary classifier training failed: {e}")

        # ─── 9) Stage-2: buyer-only split ─────────────────────────────────────
        logger.info("💰 Preparing buyer-only data...")
        try:
            mask_buy = y_tr > 0
            X_buy = X_tr[mask_buy]
            y_buy = np.log1p(y_tr[mask_buy])
            X_v_buy = X_val[y_val > 0]
            y_v_buy = np.log1p(y_val[y_val > 0])
            
            if len(X_buy) < 50:
                raise ValueError("Insufficient buyer samples for regression training")
            
            logger.info(f"✅ Buyer data prepared: {len(X_buy)} training, {len(X_v_buy)} validation")
            
        except Exception as e:
            raise ValueError(f"Buyer data preparation failed: {e}")

        # ─── 10) coarse tune (CPU-only) ───────────────────────────────────────
        logger.info("🔍 Starting coarse hyperparameter tuning...")
        try:
            best_r2, (bs, bc, bm) = -np.inf, (0.8, 0.8, 10)
            
            for subs in [0.6, 0.8, 1.0]:
                for cols in [0.6, 0.8, 1.0]:
                    for mcw in [1, 5, 10]:
                        try:
                            m = XGBRegressor(
                                tree_method='hist',
                                random_state=42, verbosity=0,
                                n_estimators=1000, max_depth=6, learning_rate=0.05,
                                subsample=subs, colsample_bytree=cols,
                                min_child_weight=mcw
                            )
                            m.set_params(early_stopping_rounds=early_stopping)
                            m = safe_model_fit(m, X_buy, y_buy, X_v_buy, y_v_buy, "XGB Regressor")
                            
                            pred = np.expm1(safe_predict(m, X_val, "XGB Regressor"))
                            r = r2_score(y_val, p_buy * pred)
                            
                            if r > best_r2:
                                best_r2, (bs, bc, bm) = r, (subs, cols, mcw)
                                
                        except Exception as e:
                            logger.warning(f"⚠️  Coarse tuning iteration failed: {e}")
                            continue

            logger.info(f"Coarse Best → subs={bs}, col={bc}, mcw={bm} → R²={best_r2:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Coarse hyperparameter tuning failed: {e}")

        # ─── 11) refine tune: lr, depth, gamma, reg_lambda ────────────────────
        logger.info("🎯 Starting refined hyperparameter tuning...")
        try:
            best_r2_2, best_cfg = best_r2, (0.05, 6, 0, 1)
            
            for lr in [0.01, 0.03, 0.05, 0.1]:
                for md in [6, 8, 10]:
                    for gamma in [0, 1, 5]:
                        for lam in [0, 1, 5, 10]:
                            try:
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
                                m = safe_model_fit(m, X_buy, y_buy, X_v_buy, y_v_buy, "XGB Regressor")
                                
                                pred = np.expm1(safe_predict(m, X_val, "XGB Regressor"))
                                r = r2_score(y_val, p_buy * pred)
                                
                                if r > best_r2_2:
                                    best_r2_2, best_cfg = r, (lr, md, gamma, lam)
                                    
                            except Exception as e:
                                logger.warning(f"⚠️  Refined tuning iteration failed: {e}")
                                continue

            logger.info(f"Refine Best → lr={best_cfg[0]}, depth={best_cfg[1]}, gamma={best_cfg[2]}, lambda={best_cfg[3]} → R²={best_r2_2:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Refined hyperparameter tuning failed: {e}")

        # ─── 12) final fit & report ───────────────────────────────────────────
        logger.info("🏁 Final model training...")
        try:
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
            xgb_final = safe_model_fit(xgb_final, X_buy, y_buy, X_v_buy, y_v_buy, "Final XGB Regressor")

            pred_val = np.expm1(safe_predict(xgb_final, X_val, "Final XGB Regressor"))
            y_hat = p_buy * pred_val
            final_r2 = r2_score(y_val, y_hat)
            logger.info(f"🔧 FINAL Validation R²: {final_r2:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Final model training failed: {e}")

        # ─── 13) save best parameters to file ─────────────────────────────────
        logger.info("💾 Saving best parameters...")
        try:
            best_params = {
                "regressor_params": {
                    "tree_method": "hist",
                    "random_state": 42,
                    "verbosity": 0,
                    "n_estimators": 1000,
                    "max_depth": int(best_cfg[1]),
                    "learning_rate": float(best_cfg[0]),
                    "subsample": float(bs),
                    "colsample_bytree": float(bc),
                    "min_child_weight": int(bm),
                    "gamma": float(best_cfg[2]),
                    "reg_lambda": float(best_cfg[3])
                },
                "classifier_params": {
                    "tree_method": "hist",
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "random_state": 42,
                    "verbosity": 0,
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05
                },
                "performance": {
                    "validation_r2": float(final_r2),
                    "coarse_best_r2": float(best_r2),
                    "refined_best_r2": float(best_r2_2)
                },
                "tuning_info": {
                    "early_stopping_rounds": int(early_stopping),
                    "coarse_best": {
                        "subsample": float(bs),
                        "colsample_bytree": float(bc),
                        "min_child_weight": int(bm)
                    },
                    "refined_best": {
                        "learning_rate": float(best_cfg[0]),
                        "max_depth": int(best_cfg[1]),
                        "gamma": float(best_cfg[2]),
                        "reg_lambda": float(best_cfg[3])
                    }
                }
            }

            # Save to output directory
            with open('output/best_params.json', 'w') as f:
                json.dump(best_params, f, indent=2)

            logger.info("✅ Best parameters saved to output/best_params.json")
            logger.info("🎉 Hyperparameter tuning completed successfully!")
            
        except Exception as e:
            raise IOError(f"Failed to save parameters: {e}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()