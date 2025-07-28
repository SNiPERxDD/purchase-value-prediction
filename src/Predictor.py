import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import json
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

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
    file_handler = logging.FileHandler(output_dir / 'predictor.log', mode='w')
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
    
    return True

def load_best_params():
    """Load best parameters from hyperparameter tuning if available"""
    params_file = 'output/best_params.json'
    
    try:
        if os.path.exists(params_file):
            logger.info("📁 Loading best parameters from hyperparameter tuning...")
            with open(params_file, 'r') as f:
                best_params = json.load(f)
            
            # Validate parameter structure
            required_keys = ['regressor_params', 'classifier_params', 'performance']
            missing_keys = set(required_keys) - set(best_params.keys())
            if missing_keys:
                logger.warning(f"⚠️  Parameter file missing keys: {missing_keys}. Using defaults.")
                return None
            
            logger.info(f"✅ Loaded parameters with validation R²: {best_params['performance']['validation_r2']:.4f}")
            return best_params
        else:
            logger.info("⚠️  No best_params.json found. Using default parameters.")
            logger.info("   Run HyperParams.py first for optimal performance.")
            return None
    except Exception as e:
        logger.warning(f"⚠️  Failed to load parameters: {e}. Using defaults.")
        return None

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

def safe_visualization(func, *args, **kwargs):
    """Safely create visualizations with error handling"""
    try:
        func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"⚠️  Visualization failed: {e}")

def main():
    """Main prediction pipeline with comprehensive error handling and progress tracking"""
    try:
        # Initialize overall progress bar
        overall_steps = [
            "Loading & Validating Data",
            "Data Quality Analysis", 
            "Feature Engineering",
            "Data Preprocessing",
            "Binary Classifier Training",
            "Additional Model Analysis",
            "Regression Model Testing",
            "Final XGB Training",
            "Results Generation",
            "Final Predictions"
        ]
        
        with tqdm(total=len(overall_steps), desc="🚀 Pipeline Progress", unit="step", colour="green") as pbar:
            logger.info("🚀 Starting prediction pipeline...")
            
            # Step 1: Find and validate data files
            pbar.set_description("📁 Loading & Validating Data")
            train_path, test_path = find_data_files()
            validate_data_files(train_path, test_path)
            pbar.update(1)
        
            # Step 2: Data Quality Analysis
            pbar.set_description("📈 Data Quality Analysis")
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
            
            logger.info("📈 Analyzing data quality...")
            try:
                print("=== TRAIN DATASET SHAPE ===", train.shape)
                train_miss = train.isnull().mean().mul(100).sort_values(ascending=False)
                print("Top 5 missing in TRAIN (%):")
                print(train_miss.head(5).round(2))

                print("\n=== TEST DATASET SHAPE ===", test.shape)
                test_miss = test.isnull().mean().mul(100).sort_values(ascending=False)
                print("Top 5 missing in TEST (%):")
                print(test_miss.head(5).round(2))
            except Exception as e:
                logger.warning(f"⚠️  Data quality analysis failed: {e}")

            # suppress seaborn FutureWarning
            warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

            logger.info("📊 Creating visualizations...")
            try:
                # train target distribution
                purchase = train[TARGET].dropna()
                if len(purchase) == 0:
                    raise ValueError("No valid target values found")
                
                cap = np.nanpercentile(purchase, 99)
                
                def create_target_histogram():
                    plt.figure(figsize=(6,4))
                    sns.histplot(purchase.clip(upper=cap), bins=30, kde=False)
                    plt.title(f'PurchaseValue (capped at {cap:.0f})')
                    plt.tight_layout()
                    plt.show()
                
                safe_visualization(create_target_histogram)

                # top numeric correlations + heatmap
                num_df = train.select_dtypes(include='number')
                if len(num_df.columns) > 1:
                    corr = num_df.corr()
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
                    def create_correlation_heatmap():
                        plt.figure(figsize=(10,8))
                        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0)
                        plt.title('Numeric Feature Correlation')
                        plt.tight_layout()
                        plt.show()
                    
                    safe_visualization(create_correlation_heatmap)
                else:
                    logger.warning("⚠️  Insufficient numeric columns for correlation analysis")
                    top_feats = []
                    
            except Exception as e:
                logger.warning(f"⚠️  Visualization creation failed: {e}")
                top_feats = []
                cap = 0
            
            pbar.update(1)

            # Step 3: Feature Engineering
            pbar.set_description("🧹 Feature Engineering")
            logger.info("🧹 Cleaning and engineering features...")
            try:
                # Data cleaning
                initial_shape = train.shape
                train.drop_duplicates(inplace=True)
                logger.info(f"Removed {initial_shape[0] - train.shape[0]} duplicate rows")
                
                const = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
                if const:
                    logger.info(f"Removing {len(const)} constant columns")
                    train.drop(columns=const, inplace=True)
                    test.drop(columns=[c for c in const if c in test.columns], inplace=True)

                # date features
                if 'date' in train.columns:
                    try:
                        for df in (train, test):
                            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                            df['month'] = df['date'].dt.month
                            df['weekday'] = df['date'].dt.dayofweek
                            df['is_weekend'] = (df['weekday']>=5).astype(int)
                            df.drop(columns='date', inplace=True)
                        logger.info("✅ Date features created")
                    except Exception as e:
                        logger.warning(f"⚠️  Date feature processing failed: {e}")

                # per-user aggregates
                if 'userId' in train.columns and 'pageViews' in train.columns:
                    try:
                        agg = (
                            train.groupby('userId')['pageViews']
                                 .agg(['mean','sum','max'])
                                 .add_prefix('u_pg_')
                                 .reset_index()
                        )
                        sess = train.groupby('userId').size().rename('u_sess_count').reset_index()
                        um = train.groupby('userId')[TARGET].mean().rename('u_mean_purchase').reset_index()

                        train = (train
                                 .merge(agg, on='userId', how='left')
                                 .merge(sess, on='userId', how='left')
                                 .merge(um, on='userId', how='left'))
                        train.fillna({
                            'u_pg_mean':0, 'u_pg_sum':0, 'u_pg_max':0,
                            'u_sess_count':0,
                            'u_mean_purchase':train[TARGET].mean()
                        }, inplace=True)

                        test = (test
                                .merge(agg, on='userId', how='left')
                                .merge(sess, on='userId', how='left')
                                .merge(um, on='userId', how='left'))
                        test.fillna({
                            'u_pg_mean':0, 'u_pg_sum':0, 'u_pg_max':0,
                            'u_sess_count':0,
                            'u_mean_purchase':train[TARGET].mean()
                        }, inplace=True)

                        train.drop(columns='userId', inplace=True)
                        test.drop(columns='userId', inplace=True)
                        logger.info("✅ User aggregates created")
                    except Exception as e:
                        logger.warning(f"⚠️  User aggregate processing failed: {e}")
                        
            except Exception as e:
                raise ValueError(f"Feature engineering failed: {e}")
            
            pbar.update(1)

            # Step 4: Data Preprocessing
            pbar.set_description("🔧 Data Preprocessing")
            logger.info("🔀 Splitting data...")
            try:
                X = train.drop(columns=[TARGET])
                y = train[TARGET]
                
                # Flexible sample size validation
                min_samples = 100
                if len(X) < min_samples:
                    logger.warning(f"⚠️  Dataset has only {len(X)} samples (recommended: >{min_samples})")
                    if len(X) < 50:  # Hard minimum
                        raise ValueError(f"Insufficient data for training ({len(X)} < 50 samples)")
                    else:
                        logger.info("📉 Proceeding with small dataset - results may be less reliable")
                
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                for df in (X_tr, X_val, y_tr, y_val):
                    df.index = range(len(df))
                
                early_stopping = max(10, int(0.05*(y_val>0).sum()))
                logger.info(f"✅ Data split complete. Early stopping: {early_stopping}")
                
            except Exception as e:
                raise ValueError(f"Data splitting failed: {e}")

            logger.info("🔧 Preprocessing features...")
            try:
                num_cols = X_tr.select_dtypes(include='number').columns.tolist()
                if len(num_cols) == 0:
                    raise ValueError("No numeric columns found")
                
                pipeline = Pipeline([
                    ('impute', SimpleImputer(strategy='median')),
                    ('log', FunctionTransformer(np.log1p, validate=True))
                ])
                X_tr[num_cols] = pipeline.fit_transform(X_tr[num_cols])
                X_val[num_cols] = pipeline.transform(X_val[num_cols])
                test[num_cols] = pipeline.transform(test[num_cols])

                cat_cols = X_tr.select_dtypes(exclude='number').columns.tolist()
                gmean = y_tr.mean()
                for c in cat_cols:
                    means = y_tr.groupby(X_tr[c]).mean()
                    for df in (X_tr, X_val, test):
                        df[c] = df[c].map(means).fillna(gmean)

                # interactions on top5
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
                
            except Exception as e:
                raise ValueError(f"Preprocessing failed: {e}")

            # Load best parameters
            best_params = load_best_params()
            pbar.update(1)

            # Step 5: Binary Classifier Training
            pbar.set_description("🎯 Binary Classifier Training")
            logger.info("🎯 Training binary classifier...")
            try:
                y_cls = (y_tr>0).astype(int)
                X_pos = X_tr[y_cls==1]; y_pos = y_cls[y_cls==1]
                X_neg = X_tr[y_cls==0]; y_neg = y_cls[y_cls==0]
                
                if len(X_pos) == 0 or len(X_neg) == 0:
                    raise ValueError("Insufficient positive or negative samples")
                
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
                clf0 = safe_model_fit(clf0, X_bal, y_bal, model_name="XGB Classifier")
                clf = CalibratedClassifierCV(clf0, cv=3).fit(X_bal, y_bal)
                p_buy = safe_predict(clf, X_val, "Calibrated Classifier")[:,1]
                logger.info("✅ Binary classifier trained successfully")
                
            except Exception as e:
                raise RuntimeError(f"Binary classifier training failed: {e}")
            
            pbar.update(1)

            # Step 6: Additional Model Analysis
            pbar.set_description("🔬 Additional Model Analysis")
            logger.info("🔬 Running additional model analysis...")
            try:
                scaler = StandardScaler().fit(X_tr[num_cols])
                X_tr_s = scaler.transform(X_tr[num_cols])
                X_va_s = scaler.transform(X_val[num_cols])

                pca = PCA(n_components=0.90, random_state=42)
                X_tr_p = pca.fit_transform(X_tr_s)
                _ = pca.transform(X_va_s)
                tqdm.write(f"PCA → {pca.n_components_} comps (var={pca.explained_variance_ratio_.sum():.2f})")

                skb = SelectKBest(f_classif, k=min(10, len(num_cols))).fit(X_tr[num_cols], y_cls)
                tqdm.write("SelectKBest → " + str(list(np.array(num_cols)[skb.get_support()])))

                # Test alternative classifiers with progress
                alt_models = [
                    ("GaussianNB", GaussianNB()),
                    ("KNeighbors", KNeighborsClassifier()),
                    ("SVM (rbf)", SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
                ]
                
                for name, clf_alt in tqdm(alt_models, desc="Testing alternative classifiers", leave=False):
                    try:
                        clf_alt.fit(X_tr[num_cols], y_cls)
                        acc = clf_alt.score(X_val[num_cols], (y_val>0))
                        tqdm.write(f"{name:12s} acc={acc:.4f}")
                    except Exception as e:
                        logger.warning(f"⚠️  {name} model failed: {e}")
                        
            except Exception as e:
                logger.warning(f"⚠️  Additional model analysis failed: {e}")
            
            pbar.update(1)

            # Step 7: Regression Model Testing
            pbar.set_description("💰 Regression Model Testing")
            logger.info("💰 Training regression models...")
            try:
                mask_buy = y_tr>0
                X_buy = X_tr[mask_buy]
                y_buy = np.log1p(y_tr[mask_buy])
                X_v_buy = X_val[y_val>0]
                y_v_buy = np.log1p(y_val[y_val>0])
                
                # Flexible buyer sample size validation
                min_buyer_samples = 50
                if len(X_buy) < min_buyer_samples:
                    logger.warning(f"⚠️  Only {len(X_buy)} buyer samples (recommended: >{min_buyer_samples})")
                    if len(X_buy) < 20:  # Hard minimum
                        raise ValueError(f"Insufficient buyer samples for regression ({len(X_buy)} < 20)")
                    else:
                        logger.info("📉 Proceeding with small buyer dataset - regression may be less reliable")

                # Test multiple regression models with progress
                models_to_test = [
                    ("Ridge", Ridge(solver='svd')),
                    ("RandomForest", RandomForestRegressor(random_state=42)),
                    ("SGD", make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, tol=1e-3, random_state=42))),
                    ("MLP", make_pipeline(StandardScaler(), MLPRegressor(max_iter=500, hidden_layer_sizes=(100,50), random_state=42)))
                ]
                
                r2_scores = {}
                for name, model in tqdm(models_to_test, desc="Testing regression models", leave=False):
                    try:
                        model.fit(X_buy, y_buy)
                        pred = model.predict(X_v_buy)
                        r2_scores[name] = r2_score(y_v_buy, pred)
                    except Exception as e:
                        logger.warning(f"⚠️  {name} model failed: {e}")
                        r2_scores[name] = 0.0

                tqdm.write(f"🧪 Ridge: {r2_scores.get('Ridge', 0):.4f} | RF: {r2_scores.get('RandomForest', 0):.4f}"
                           f" | SGD: {r2_scores.get('SGD', 0):.4f} | MLP: {r2_scores.get('MLP', 0):.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️  Regression model testing failed: {e}")
                r2_scores = {}
            
            pbar.update(1)

            # Step 8: Final XGB Training
            pbar.set_description("🎯 Final XGB Training")
            logger.info("🎯 Training final XGB regressor...")
            try:
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
                xgb_final = safe_model_fit(xgb_final, X_buy, y_buy, X_v_buy, y_v_buy, "Final XGB Regressor")
                logger.info("✅ Final XGB regressor trained successfully")
                
            except Exception as e:
                raise RuntimeError(f"Final XGB regressor training failed: {e}")
            
            pbar.update(1)

            # Step 9: Results Generation
            pbar.set_description("📊 Results Generation")
            logger.info("📊 Generating final results...")
            try:
                pred_val = np.expm1(safe_predict(xgb_final, X_val, "Final XGB Regressor"))
                y_hat = p_buy * pred_val
                r2_final = r2_score(y_val, y_hat)
                
                # Use tqdm.write() to avoid progress bar interference
                tqdm.write(f"\n🔧 FINAL R²: {r2_final:.4f}\n")

                tqdm.write("💡 Key Insights:")
                tqdm.write(f"- Target skewed → log1p cap@99% = {cap:.0f}")
                tqdm.write(f"- Top 5 missing TRAIN: {list(train_miss.head(5).index) if 'train_miss' in locals() else 'N/A'}")
                tqdm.write(f"- Top 5 missing TEST : {list(test_miss.head(5).index) if 'test_miss' in locals() else 'N/A'}")
                tqdm.write(f"- Top correlates       : {top_feats}")
                tqdm.write(f"- PCA comps           : {pca.n_components_ if 'pca' in locals() else 'N/A'}")
                tqdm.write(f"- DidBuy acc          : Various models tested")
                tqdm.write(f"- Reg R² (buyers)     : {r2_scores}")
                tqdm.write(f"- Two-stage XGB R²    : {r2_final:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️  Results generation failed: {e}")
                r2_final = 0.0
            
            pbar.update(1)

            # Step 10: Final Predictions
            pbar.set_description("🔄 Final Predictions")
            logger.info("🔄 Refitting on full data and generating predictions...")
            try:
                # Refit classifier on full data
                clf.fit(pd.concat([X_tr,X_val]), (pd.concat([y_tr,y_val])>0).astype(int))
                
                # Refit regressor on full buyer data
                X_full = pd.concat([X_tr,X_val])
                y_full = pd.concat([y_tr,y_val])
                mask_f = y_full>0
                X_bf = X_full[mask_f]
                y_bf = np.log1p(y_full[mask_f])

                xgb_final.set_params(early_stopping_rounds=None)
                xgb_final = safe_model_fit(xgb_final, X_bf, y_bf, model_name="Final Refitted XGB")

                # Generate test predictions
                p_buy_te = safe_predict(clf, test, "Final Classifier")[:,1]
                pred_te = np.expm1(safe_predict(xgb_final, test, "Final Regressor"))
                
                # Save predictions
                predictions_df = pd.DataFrame({
                    'id': np.arange(len(test)),
                    'purchaseValue': p_buy_te * pred_te
                })
                predictions_df.to_csv('output/prediction.csv', index=False)
                logger.info("✅ Predictions saved to output/prediction.csv")
                
            except Exception as e:
                raise RuntimeError(f"Final prediction generation failed: {e}")
            
            # Complete the final step and close progress bar
            pbar.update(1)
            pbar.close()
            
            # Now safe to print final messages normally
            print("🎉 Prediction pipeline completed successfully!")
            print(f"📁 Results saved to: output/prediction.csv")
            print(f"🎯 Final R² Score: {r2_final:.4f}")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()