import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection   import train_test_split
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import PolynomialFeatures
from sklearn.metrics           import r2_score
from sklearn.utils             import resample
from sklearn.calibration       import CalibratedClassifierCV
from xgboost                   import XGBClassifier, XGBRegressor

# Ensure output directory exists before configuring logging
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Configure logging with explicit file handling
def setup_logging():
    """Setup logging configuration with file and console output"""
    # Clear any existing handlers to avoid conflicts
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(output_dir / 'hyperparams.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

logger = setup_logging()

def find_data_files():
    """Dynamically find train_data.csv and test_data.csv in various locations"""
    project_root = Path.cwd()
    
    # Extended search patterns for different environments
    train_patterns = [
        'train_data.csv',
        '**/train_data.csv',
        '/kaggle/input/*/train_data.csv',
        '/kaggle/input/*/*/train_data.csv',
        '../input/*/train_data.csv',
        '../input/*/*/train_data.csv',
        'data/train_data.csv',
        './data/train_data.csv'
    ]
    
    test_patterns = [
        'test_data.csv',
        '**/test_data.csv', 
        '/kaggle/input/*/test_data.csv',
        '/kaggle/input/*/*/test_data.csv',
        '../input/*/test_data.csv',
        '../input/*/*/test_data.csv',
        'data/test_data.csv',
        './data/test_data.csv'
    ]
    
    train_file = None
    test_file = None
    
    logger.info(f"🔍 Searching for data files from: {project_root}")
    
    # Search for training data
    for pattern in train_patterns:
        try:
            if pattern.startswith('/') or pattern.startswith('../'):
                # Absolute or relative paths
                matches = list(Path('/').glob(pattern.lstrip('/'))) if pattern.startswith('/') else list(Path('.').glob(pattern))
            else:
                # Relative to project root
                matches = list(project_root.glob(pattern))
            
            if matches:
                train_file = matches[0]
                logger.info(f"✅ Found training data: {train_file}")
                break
        except Exception as e:
            logger.debug(f"Pattern {pattern} failed: {e}")
            continue
    
    # Search for test data
    for pattern in test_patterns:
        try:
            if pattern.startswith('/') or pattern.startswith('../'):
                # Absolute or relative paths
                matches = list(Path('/').glob(pattern.lstrip('/'))) if pattern.startswith('/') else list(Path('.').glob(pattern))
            else:
                # Relative to project root
                matches = list(project_root.glob(pattern))
            
            if matches:
                test_file = matches[0]
                logger.info(f"✅ Found test data: {test_file}")
                break
        except Exception as e:
            logger.debug(f"Pattern {pattern} failed: {e}")
            continue
    
    # If still not found, list available files for debugging
    if not train_file or not test_file:
        logger.info("🔍 Available files for debugging:")
        try:
            # List current directory
            logger.info(f"Current directory ({project_root}):")
            for item in project_root.iterdir():
                logger.info(f"  - {item}")
            
            # Check common Kaggle input paths
            kaggle_input = Path('/kaggle/input')
            if kaggle_input.exists():
                logger.info("Kaggle input directory:")
                for item in kaggle_input.rglob('*.csv'):
                    logger.info(f"  - {item}")
                    
        except Exception as e:
            logger.warning(f"Could not list files: {e}")
    
    if not train_file:
        raise FileNotFoundError(
            "train_data.csv not found. Searched in:\n"
            "- Current directory and subdirectories\n"
            "- /kaggle/input/ and subdirectories\n"
            "- ../input/ and subdirectories\n"
            "Please ensure the file exists in one of these locations."
        )
    
    if not test_file:
        raise FileNotFoundError(
            "test_data.csv not found. Searched in:\n"
            "- Current directory and subdirectories\n"
            "- /kaggle/input/ and subdirectories\n"
            "- ../input/ and subdirectories\n"
            "Please ensure the file exists in one of these locations."
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

def validate_dataframe(df, name, required_cols=None):
    """Validate dataframe structure and content"""
    if df is None or df.empty:
        raise ValueError(f"{name} dataframe is empty or None")
    
    logger.info(f"✅ {name} shape: {df.shape}")
    
    if required_cols:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} missing required columns: {missing_cols}")
    
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

def safe_predict(model, X, model_name="Model", prediction_type="auto"):
    """Safely make predictions with error handling"""
    try:
        # Explicit prediction type handling
        if prediction_type == "proba" or (prediction_type == "auto" and hasattr(model, 'predict_proba') and 'Classifier' in str(type(model))):
            return model.predict_proba(X)
        else:
            return model.predict(X)
    except Exception as e:
        logger.error(f"❌ {model_name} prediction failed: {e}")
        raise RuntimeError(f"{model_name} prediction failed: {e}")

def main():
    """Main hyperparameter tuning pipeline with comprehensive error handling and progress tracking"""
    try:
        # Initialize overall progress bar
        overall_steps = [
            "Loading & Validating Data",
            "Feature Engineering", 
            "Data Preprocessing",
            "Binary Classifier Training",
            "Coarse Hyperparameter Tuning",
            "Refined Hyperparameter Tuning",
            "Final Model Training",
            "Saving Best Parameters"
        ]
        
        with tqdm(total=len(overall_steps), desc="🚀 Hyperparameter Tuning Progress", unit="step", colour="blue") as pbar:
            logger.info("🚀 Starting hyperparameter tuning pipeline...")
            
            # Step 1: Loading & Validating Data
            pbar.set_description("📁 Loading & Validating Data")
            train_path, test_path = find_data_files()
            validate_data_files(train_path, test_path)
            
            logger.info("📊 Loading datasets...")
            try:
                train = pd.read_csv(train_path)
                test = pd.read_csv(test_path)
                logger.info(f"✅ Loaded training data from: {train_path}")
                logger.info(f"✅ Loaded test data from: {test_path}")
            except Exception as e:
                raise IOError(f"Failed to load CSV files: {e}")
            
            TARGET = 'purchaseValue'
            validate_dataframe(train, "Training data", [TARGET])
            validate_dataframe(test, "Test data")
            pbar.update(1)

            # Step 2: Feature Engineering
            pbar.set_description("🧹 Feature Engineering")
            logger.info("🧹 Cleaning and engineering features...")
            
            # Data cleaning
            initial_shape = train.shape
            train.drop_duplicates(inplace=True)
            logger.info(f"Removed {initial_shape[0] - train.shape[0]} duplicate rows")
            
            const = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
            if const:
                logger.info(f"Removing {len(const)} constant columns")
                train.drop(columns=const, inplace=True)
                test.drop(columns=[c for c in const if c in test.columns], inplace=True)

            # Date features
            if 'date' in train.columns:
                try:
                    for df in (train, test):
                        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                        df['month'] = df['date'].dt.month
                        df['weekday'] = df['date'].dt.dayofweek
                        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
                        df.drop(columns='date', inplace=True)
                    logger.info("✅ Date features created")
                except Exception as e:
                    logger.warning(f"⚠️  Date feature processing failed: {e}")

            # Per-user aggregates
            if 'userId' in train.columns and 'pageViews' in train.columns:
                try:
                    agg_pg = (
                        train.groupby('userId')['pageViews']
                             .agg(['mean','sum','max'])
                             .add_prefix('u_pg_')
                             .reset_index()
                    )
                    sess = train.groupby('userId').size().rename('u_sess_count').reset_index()
                    um = train.groupby('userId')[TARGET].mean().rename('u_mean_purchase').reset_index()

                    train = (train
                             .merge(agg_pg, on='userId', how='left')
                             .merge(sess, on='userId', how='left')
                             .merge(um, on='userId', how='left'))
                    train.fillna({
                        'u_pg_mean':0,'u_pg_sum':0,'u_pg_max':0,
                        'u_sess_count':0,
                        'u_mean_purchase': train[TARGET].mean()
                    }, inplace=True)

                    test = (test
                            .merge(agg_pg, on='userId', how='left')
                            .merge(sess, on='userId', how='left')
                            .merge(um, on='userId', how='left'))
                    test.fillna({
                        'u_pg_mean':0,'u_pg_sum':0,'u_pg_max':0,
                        'u_sess_count':0,
                        'u_mean_purchase': train[TARGET].mean()
                    }, inplace=True)

                    train.drop(columns='userId', inplace=True)
                    test.drop(columns='userId', inplace=True)
                    logger.info("✅ User aggregates created")
                except Exception as e:
                    logger.warning(f"⚠️  User aggregate processing failed: {e}")
            
            pbar.update(1)

            # Step 3: Data Preprocessing
            pbar.set_description("🔧 Data Preprocessing")
            logger.info("🔀 Splitting data...")
            
            X = train.drop(columns=[TARGET])
            y = train[TARGET]
            
            # Flexible sample size validation
            min_samples = 100
            if len(X) < min_samples:
                logger.warning(f"⚠️  Dataset has only {len(X)} samples (recommended: >{min_samples})")
                if len(X) < 50:
                    raise ValueError(f"Insufficient data for training ({len(X)} < 50 samples)")
                else:
                    logger.info("📉 Proceeding with small dataset - results may be less reliable")
            
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            for df in (X_tr, X_val, y_tr, y_val):
                df.reset_index(drop=True, inplace=True)

            early_stopping = max(10, int(0.05 * (y_val > 0).sum()))
            logger.info(f"✅ Data split complete. Early stopping: {early_stopping}")

            # Process numeric features
            num_cols = X_tr.select_dtypes(include='number').columns
            if len(num_cols) == 0:
                raise ValueError("No numeric columns found")
            
            imp = SimpleImputer(strategy='median').fit(X_tr[num_cols])
            for df in (X_tr, X_val, test):
                df[num_cols] = imp.transform(df[num_cols])
                df[num_cols] = np.log1p(df[num_cols])

            # Target-encode categoricals
            cat_cols = X_tr.select_dtypes(exclude='number').columns
            global_mean = y_tr.mean()
            for c in cat_cols:
                means = y_tr.groupby(X_tr[c]).mean()
                for df in (X_tr, X_val, test):
                    df[c] = df[c].map(means).fillna(global_mean)

            # Feature interactions
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
            logger.info(f"✅ Preprocessing complete. Final features: {X_tr.shape[1]}")
            pbar.update(1)

            # Step 4: Binary Classifier Training
            pbar.set_description("🎯 Binary Classifier Training")
            logger.info("🎯 Training binary classifier...")
            
            y_cls = (y_tr > 0).astype(int)
            X_pos, y_pos = X_tr[y_cls==1], y_cls[y_cls==1]
            X_neg, y_neg = X_tr[y_cls==0], y_cls[y_cls==0]
            
            if len(X_pos) == 0 or len(X_neg) == 0:
                raise ValueError("Insufficient positive or negative samples")
            
            X_pos_up, y_pos_up = resample(X_pos, y_pos, replace=True, n_samples=len(X_neg), random_state=42)
            X_bal = pd.concat([X_neg, X_pos_up], axis=0)
            y_bal = pd.concat([y_neg, y_pos_up], axis=0)

            clf0 = XGBClassifier(
                tree_method='hist', use_label_encoder=False, eval_metric='logloss',
                random_state=42, verbosity=0, n_estimators=200, max_depth=6, learning_rate=0.05
            )
            clf0 = safe_model_fit(clf0, X_bal, y_bal, model_name="XGB Classifier")
            clf = CalibratedClassifierCV(clf0, cv=3).fit(X_bal, y_bal)
            p_buy = safe_predict(clf, X_val, "Calibrated Classifier")[:,1]
            logger.info("✅ Binary classifier trained successfully")

            # Prepare buyer data
            mask_buy = y_tr > 0
            X_buy = X_tr[mask_buy]
            y_buy = np.log1p(y_tr[mask_buy])
            X_v_buy = X_val[y_val > 0]
            y_v_buy = np.log1p(y_val[y_val > 0])
            
            min_buyer_samples = 50
            if len(X_buy) < min_buyer_samples:
                logger.warning(f"⚠️  Only {len(X_buy)} buyer samples (recommended: >{min_buyer_samples})")
                if len(X_buy) < 20:
                    raise ValueError(f"Insufficient buyer samples for regression ({len(X_buy)} < 20)")
                else:
                    logger.info("📉 Proceeding with small buyer dataset")
            
            logger.info(f"✅ Buyer data prepared: {len(X_buy)} training, {len(X_v_buy)} validation")
            pbar.update(1)

            # Step 5: Coarse Hyperparameter Tuning
            pbar.set_description("🔍 Coarse Hyperparameter Tuning")
            logger.info("🔍 Starting coarse hyperparameter tuning...")
            
            best_r2, (bs, bc, bm) = -np.inf, (0.8, 0.8, 10)
            coarse_params = [(subs, cols, mcw) for subs in [0.6, 0.8, 1.0] 
                            for cols in [0.6, 0.8, 1.0] for mcw in [1, 5, 10]]
            
            for subs, cols, mcw in tqdm(coarse_params, desc="Coarse tuning", leave=False):
                try:
                    m = XGBRegressor(
                        tree_method='hist', random_state=42, verbosity=0,
                        n_estimators=1000, max_depth=6, learning_rate=0.05,
                        subsample=subs, colsample_bytree=cols, min_child_weight=mcw,
                        eval_metric='rmse'
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

            tqdm.write(f"Coarse Best → subs={bs}, col={bc}, mcw={bm} → R²={best_r2:.4f}")
            pbar.update(1)

            # Step 6: Refined Hyperparameter Tuning
            pbar.set_description("🎯 Refined Hyperparameter Tuning")
            logger.info("🎯 Starting refined hyperparameter tuning...")
            
            best_r2_2, best_cfg = best_r2, (0.05, 6, 0, 1)
            refined_params = [(lr, md, gamma, lam) for lr in [0.01, 0.03, 0.05, 0.1]
                             for md in [6, 8, 10] for gamma in [0, 1, 5] for lam in [0, 1, 5, 10]]
            
            for lr, md, gamma, lam in tqdm(refined_params, desc="Refined tuning", leave=False):
                try:
                    m = XGBRegressor(
                        tree_method='hist', random_state=42, verbosity=0, n_estimators=1000,
                        max_depth=md, learning_rate=lr, subsample=bs, colsample_bytree=bc,
                        min_child_weight=bm, gamma=gamma, reg_lambda=lam, eval_metric='rmse'
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

            tqdm.write(f"Refine Best → lr={best_cfg[0]}, depth={best_cfg[1]}, gamma={best_cfg[2]}, lambda={best_cfg[3]} → R²={best_r2_2:.4f}")
            pbar.update(1)

            # Step 7: Final Model Training
            pbar.set_description("🏁 Final Model Training")
            logger.info("🏁 Final model training...")
            
            xgb_final = XGBRegressor(
                tree_method='hist', random_state=42, verbosity=0, n_estimators=1000,
                max_depth=best_cfg[1], learning_rate=best_cfg[0], subsample=bs,
                colsample_bytree=bc, min_child_weight=bm, gamma=best_cfg[2],
                reg_lambda=best_cfg[3], eval_metric='rmse'
            )
            xgb_final.set_params(early_stopping_rounds=early_stopping)
            xgb_final = safe_model_fit(xgb_final, X_buy, y_buy, X_v_buy, y_v_buy, "Final XGB Regressor")

            pred_val = np.expm1(safe_predict(xgb_final, X_val, "Final XGB Regressor"))
            y_hat = p_buy * pred_val
            final_r2 = r2_score(y_val, y_hat)
            tqdm.write(f"🔧 FINAL Validation R²: {final_r2:.4f}")
            pbar.update(1)

            # Step 8: Saving Best Parameters
            pbar.set_description("💾 Saving Best Parameters")
            logger.info("💾 Saving best parameters...")
            
            best_params = {
                "regressor_params": {
                    "tree_method": "hist", "random_state": 42, "verbosity": 0, "n_estimators": 1000,
                    "max_depth": int(best_cfg[1]), "learning_rate": float(best_cfg[0]),
                    "subsample": float(bs), "colsample_bytree": float(bc),
                    "min_child_weight": int(bm), "gamma": float(best_cfg[2]),
                    "reg_lambda": float(best_cfg[3])
                },
                "classifier_params": {
                    "tree_method": "hist", "use_label_encoder": False, "eval_metric": "logloss",
                    "random_state": 42, "verbosity": 0, "n_estimators": 200,
                    "max_depth": 6, "learning_rate": 0.05
                },
                "performance": {
                    "validation_r2": float(final_r2),
                    "coarse_best_r2": float(best_r2),
                    "refined_best_r2": float(best_r2_2)
                },
                "tuning_info": {
                    "early_stopping_rounds": int(early_stopping),
                    "coarse_best": {"subsample": float(bs), "colsample_bytree": float(bc), "min_child_weight": int(bm)},
                    "refined_best": {"learning_rate": float(best_cfg[0]), "max_depth": int(best_cfg[1]), 
                                   "gamma": float(best_cfg[2]), "reg_lambda": float(best_cfg[3])}
                }
            }

            with open('output/best_params.json', 'w') as f:
                json.dump(best_params, f, indent=2)

            logger.info("✅ Best parameters saved to output/best_params.json")
            pbar.update(1)
            pbar.close()
            
            # Now safe to print final messages normally
            print("🎉 Hyperparameter tuning completed successfully!")
            print(f"📁 Best parameters saved to: output/best_params.json")
            print(f"🎯 Final Validation R²: {final_r2:.4f}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()