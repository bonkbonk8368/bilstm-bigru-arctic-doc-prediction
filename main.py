#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:09:57 2025

@author: gy

Refined Multivariate Time Series Forecaster using a 2-Layer Bidirectional LSTM-GRU RNN.
Optimized for performance, stability, and CUDA/cuDNN compatibility.
"""

import gc
import os
import random
import sys
import time
import warnings
from typing import Dict, Tuple, Any

# --- Environment Configuration for GPU/Performance ---
# Specify which GPU to use (e.g., "0" or "1"). Set to "" to use CPU if desired, or if GPU is unavailable.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Disable oneDNN optimizations if prioritizing cuDNN for RNNs (often beneficial for LSTM/GRU on GPU).
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Core libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis libraries for advanced EDA
from scipy.stats import shapiro

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.stattools import adfuller
    import statsmodels.graphics.tsaplots as tsaplots

    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: Statsmodels not found. Advanced EDA (VIF, ADF, ACF/PACF plots) will be disabled.")
    STATSMODELS_AVAILABLE = False
    pass

# TensorFlow and Keras for deep learning
try:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                            ReduceLROnPlateau, History)
    from tensorflow.keras.layers import (LSTM, GRU, Bidirectional, Dense, Dropout,
                                         Input, LayerNormalization)
    # LogCosh loss is robust to outliers
    from tensorflow.keras.losses import LogCosh
    from tensorflow.keras.metrics import RootMeanSquaredError
    from tensorflow.keras.models import Sequential, load_model
    # AdamW (Weight Decay) often generalizes better than standard Adam
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.regularizers import l2
except ImportError:
    print("Error: TensorFlow not found. Please install it (e.g., 'pip install tensorflow').")
    sys.exit(1)


class Config:
    """Configuration class for hyperparameters and settings."""
    # --- Data & Output Paths ---
    CHEM_DATA_PATH: str = "River_water_chemistry_10.17897_1GTF-SX86_data.txt"
    DISCHARGE_DATA_PATH: str = "Discharge at a cross section of the river, (m3_s)_10.17897_A308-6075_data.txt"
    PLOTS_DIR: str = "plots"
    MODEL_SAVE_PATH: str = "multivariate_forecaster.keras"
    BENCHMARK_MODEL_SAVE_PATH: str = "benchmark_univariate_forecaster.keras"

    # --- Reproducibility ---
    SEED: int = 42

    # --- Data Preprocessing ---
    TRAIN_SPLIT: float = 0.75
    VAL_SPLIT: float = 0.15
    VIF_THRESHOLD: int = 10  # Threshold for multicollinearity check

    # --- Model Architecture ---
    WINDOW_SIZE: int = 45  # Lookback period (days)
    LSTM_UNITS_1: int = 128
    GRU_UNITS_2: int = 64
    DENSE_UNITS: int = 32
    DROPOUT_RATE: float = 0.4
    L2_REG: float = 1e-4

    # --- Training Parameters ---
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-6
    BATCH_SIZE: int = 64
    EPOCHS: int = 400
    EARLY_STOPPING_PATIENCE: int = 60
    LR_PLATEAU_PATIENCE: int = 30
    # Factor to emphasize higher magnitude samples during training
    SAMPLE_WEIGHT_FACTOR: float = 1.5

    # --- Evaluation & Analysis ---
    MC_DROPOUT_SAMPLES: int = 50  # Samples for uncertainty estimation
    ACF_PACF_LAGS: int = 30


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    # Enforce deterministic behavior (crucial for reproducibility, especially on GPU)
    tf.config.experimental.enable_op_determinism()
    random.seed(seed)
    np.random.seed(seed)


def setup_environment(cfg: Config) -> None:
    """Initializes the script environment, configures plotting, and sets up GPU."""
    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)
    print(f"Plots will be saved to the '{cfg.PLOTS_DIR}/' directory.")

    # GPU Configuration Check
    print("\n--- GPU Setup ---")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

    if gpus:
        print(f"Found {len(gpus)} GPU(s) available.")
        try:
            # Enable memory growth to avoid allocating all GPU memory at startup
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")
    else:
        print("No GPU found. Model will run on CPU.")
    print("-" * 30)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_clean_data(cfg: Config) -> pd.DataFrame:
    """Loads, cleans, merges, resamples, interpolates, and prepares the dataset."""
    print("\nLoading and preprocessing data...")
    try:
        chem_raw = pd.read_csv(cfg.CHEM_DATA_PATH, delimiter="\t")
        dis_raw = pd.read_csv(cfg.DISCHARGE_DATA_PATH, delimiter="\t")
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        return pd.DataFrame()

    # Replace missing value placeholders
    MISSING_VALUE_CODE = -9999
    chem_raw.replace(MISSING_VALUE_CODE, np.nan, inplace=True)
    dis_raw.replace(MISSING_VALUE_CODE, np.nan, inplace=True)

    # 1. Clean Chemistry Data
    if 'DOC_ppm' not in chem_raw.columns:
        print("Error: Target column 'DOC_ppm' not found.")
        return pd.DataFrame()

    if (doc_missing_pct := chem_raw['DOC_ppm'].isna().mean() * 100) > 50:
        warnings.warn(
            f"Warning: Over {doc_missing_pct:.1f}% of the target 'DOC_ppm' is missing."
        )

    chem = chem_raw.drop(columns=['DTN_ppb', 'Time', 'NH4-N_ppb', 'NO3-N_ppb'], errors='ignore')
    chem['Date'] = pd.to_datetime(chem['Date'], errors='coerce')
    chem.dropna(subset=['Date'], inplace=True)

    # Handle values below detection limit (BDL)
    dl_cols = ['Cl_ppm', 'SO4_ppm', 'Na_ppm', 'Mg_ppm', 'K_ppm', 'Ca_ppm', 'Fe_ppm', 'Al_ppm']
    DETECTION_LIMIT_PROXY = 0.05
    for col in dl_cols:
        if col in chem.columns:
            chem[col] = chem[col].replace(0, DETECTION_LIMIT_PROXY)

    chem.set_index('Date', inplace=True)
    # Resample to daily frequency
    daily_chem = chem.resample('D').mean(numeric_only=True)

    # 2. Clean Discharge Data
    discharge_col = next((col for col in dis_raw.columns if 'Q (m3/s)' in col or 'm3_s' in col), None)
    if discharge_col is None:
        print("Error: Could not identify the discharge column.")
        return pd.DataFrame()

    dis = dis_raw.drop(columns=['Time', 'Quality Flag'], errors='ignore')
    dis['Date'] = pd.to_datetime(dis['Date'], errors='coerce')
    dis.dropna(subset=['Date'], inplace=True)
    dis.set_index('Date', inplace=True)
    daily_dis = dis[[discharge_col]].resample('D').mean().rename(columns={discharge_col: 'Discharge'})

    # 3. Merge Datasets
    df = pd.merge(daily_chem, daily_dis, left_index=True, right_index=True, how='outer')

    # 4. Initial Visualizations (Pre-Interpolation)
    generate_raw_data_visualizations(df, cfg)

    # 5. Interpolation
    # Time-weighted interpolation is suitable for time series data
    print("Interpolating missing data using time-weighted method...")
    df.interpolate(method='time', axis=0, limit_direction='both', inplace=True)

    # Ensure target variable has no NaNs remaining
    df = df[df['DOC_ppm'].notna()].copy()

    # 6. Post-Interpolation Visualization (ACF/PACF)
    plot_acf_pacf(df, cfg)
    
    # 7. Post-Interpolation, pre-feature-engineerng normality & stationarity tests
    prelim_eda(df, cfg)

    print(f"Data loading and cleaning complete. Final Shape: {df.shape}")
    return df


def generate_raw_data_visualizations(df: pd.DataFrame, cfg: Config):
    """Generates visualizations for the raw, merged data before interpolation."""
    # Save basic statistics
    stats_df = pd.DataFrame(df.describe())
    stats_df.to_csv("Raw_stats_after_cleaning_description.csv", index=True)

    # Plot raw scatterplot (Pairplot)
    try:
        # Sample if dataset is very large to speed up plotting
        g = sns.pairplot(df)
        g.fig.suptitle('Scatterplot (Original variables vs. DOC) and Histograms', fontsize=16, y=1.02)
        g.fig.tight_layout()
        g.fig.savefig(os.path.join(cfg.PLOTS_DIR, 'raw_scatterplot.png'), dpi=300)
        plt.close(g.fig)
    except Exception as e:
        print(f"Could not generate pairplot: {e}")

    # Plot raw time series
    for column in df.columns:
        plt.figure(figsize=(20, 6))
        plt.plot(df.index, df[column], label=column, linewidth=0.5, alpha=0.5)
        # Scatter points to highlight actual measurements vs gaps
        plt.scatter(df.index, df[column], s=10, alpha=0.8)
        plt.title(f'Raw Time Series: {column}', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(cfg.PLOTS_DIR, f'{column}_raw_timeseries.png'), dpi=300)
        plt.show()
        plt.close()

    # Plot raw correlation matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (Original variables vs. DOC)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, 'raw_multivariate_correlation_heatmap.png'), dpi=300)
    plt.show()
    plt.close()
    
    
def prelim_eda(df: pd.DataFrame, cfg: Config):
    """Generate Shapiro-Wilks and ADF test result tables before feature engineering"""
    
    print("\nPerforming advanced EDA and feature selection...")

    if not STATSMODELS_AVAILABLE:
        print("Statsmodels not available. Skipping Normality and Stationarity test.")
        return df

    # Setup directory for saving EDA statistics
    out_dir = os.path.join(cfg.PLOTS_DIR, "eda_stats_beforeengineer")
    os.makedirs(out_dir, exist_ok=True)

    # Shapiro-Wilk Test (Normality)
    print("\n--- Shapiro-Wilk Test (Normality) ---")
    shapiro_rows = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 3:
            try:
                stat, p = shapiro(s)
                is_normal = p >= 0.05
                if not is_normal:
                    print(f"  - Column '{col}': Likely NOT normally distributed (p={p:.4f})")
                shapiro_rows.append({"feature": col, "p_value": float(p), "normal_at_0.05": bool(is_normal)})
            except Exception as e:
                print(f"  - Error processing '{col}': {e}")

    if shapiro_rows:
        pd.DataFrame(shapiro_rows).sort_values("p_value").to_csv(os.path.join(out_dir, "shapiro_wilk_results_beforeengineer.csv"),
                                                                 index=False)

    # Augmented Dickey-Fuller Test (Stationarity)
    print("\n--- Augmented Dickey-Fuller Test (Stationarity) ---")
    adf_rows = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) > 10:
            try:
                result = adfuller(s, autolag='AIC')
                p = result[1]
                is_stationary = p < 0.05
                print(f"  - Column '{col}': p-value={p:.4f} -> {'Stationary' if is_stationary else 'Non-Stationary'}")
                adf_rows.append({"feature": col, "p_value": float(p), "stationary_at_0.05": bool(is_stationary)})
            except Exception as e:
                print(f"  - Error processing '{col}': {e}")
    if adf_rows:
        pd.DataFrame(adf_rows).to_csv(os.path.join(out_dir, "adf_results_beforeengineer.csv"), index=False)

    
    

def plot_acf_pacf(df: pd.DataFrame, cfg: Config):
    """Generates Autocorrelation and Partial Autocorrelation plots."""
    if not STATSMODELS_AVAILABLE:
        print("Skipping ACF/PACF plots as statsmodels is not available.")
        return

    print("Generating ACF/PACF plots...")
    for column in df.columns:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            tsaplots.plot_acf(df[column], lags=cfg.ACF_PACF_LAGS, ax=axes[0], title=f'ACF for {column}')
            # Using 'ywm' (Yule-Walker) method as it is generally robust
            tsaplots.plot_pacf(df[column], lags=cfg.ACF_PACF_LAGS, ax=axes[1], title=f'PACF for {column}', method='ywm')
            plt.suptitle(f'Autocorrelation Analysis: {column}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(cfg.PLOTS_DIR, f'{column}_acf_pacf_plots.png'), dpi=300)
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Could not plot ACF/PACF for {column}: {e}")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_univariate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features for the benchmark model using only time and the target (DOC).
       Ensures no target leakage."""
    print("\nEngineering features for benchmark univariate model...")
    df_uni = pd.DataFrame(index=df.index)
    TARGET = 'DOC_ppm'

    # Log transform target to stabilize variance
    df_uni['log_DOC'] = np.log1p(df[TARGET].clip(lower=0))

    # Cyclical time features (Seasonality)
    doy = df_uni.index.dayofyear
    days_in_year = 365.25
    df_uni['doy_sin'] = np.sin(2 * np.pi * doy / days_in_year)
    df_uni['doy_cos'] = np.cos(2 * np.pi * doy / days_in_year)

    # Rolling features on the target
    # Standard Moving Average (SMA)
    df_uni['DOC_sma15'] = df_uni['log_DOC'].rolling(window=15).mean()
    #df_uni['DOC_roll7_mean'] = df_uni['log_DOC'].shift(1).rolling(window=7).mean()
    df_uni['DOC_roll7_std'] = df_uni['log_DOC'].rolling(window=7).std()

    # Lag features on the target
    df_uni['DOC_lag1'] = df_uni['log_DOC'].shift(1)
    df_uni['DOC_lag2'] = df_uni['log_DOC'].shift(2)
    df_uni['DOC_lag8'] = df_uni['log_DOC'].shift(8)

    # Drop NaNs created by rolling/lagging operations
    df_uni.dropna(inplace=True)
    print(f"Univariate feature engineering complete. Shape: {df_uni.shape}")
    return df_uni


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-based, interaction, and transformed features for the main multivariate model.
       Discharge is highly focused due to its theorectical importance linked with DOC."""
    print("\nEngineering features for multivariate model...")
    if df.empty:
        return df
    df_eng = df.copy()
    
    # 1. Log transformations for skewed variables and the target
    log_cols = {
        'Cl_ppm': 'log_Cl', 'SO4_ppm': "log_SO4", 'Na_ppm': 'log_Na', 'Mg_ppm': 'log_Mg',
        'K_ppm': 'log_K', 'Ca_ppm': 'log_Ca', 'Fe_ppm': 'log_Fe', 'Al_ppm': 'log_Al',
        'Discharge': 'log_Discharge', 'DOC_ppm': 'log_DOC'
    }
    for old_name, new_name in log_cols.items():
        if old_name in df_eng.columns:
            df_eng[new_name] = np.log1p(df_eng[old_name].clip(lower=0))
    df_eng.drop(columns=log_cols.keys(), errors='ignore', inplace=True)

    # 2. Cyclical time features
    doy = df_eng.index.dayofyear
    days_in_year = 365.25
    df_eng['doy_sin'] = np.sin(2 * np.pi * doy / days_in_year)
    df_eng['doy_cos'] = np.cos(2 * np.pi * doy / days_in_year)

    # 3. Rolling, lagged features (Using SMA instead of EWMA)
    if 'log_DOC' in df_eng.columns:
        # Standard Moving Average (SMA)
        df_eng['DOC_sma15'] = df_eng['log_DOC'].rolling(window=15).mean()
        #df_eng['DOC_roll7_mean'] = df_eng['log_DOC'].shift(1).rolling(window=7).mean()
        df_eng['DOC_roll7_std'] = df_eng['log_DOC'].rolling(window=7).std()

        # Lag features on the target
        df_eng['DOC_lag1'] = df_eng['log_DOC'].shift(1)
        df_eng['DOC_lag2'] = df_eng['log_DOC'].shift(2)
        df_eng['DOC_lag8'] = df_eng['log_DOC'].shift(8)

    if 'log_Discharge' in df_eng.columns:
        # Standard Moving Average (SMA)
        df_eng['Discharge_sma15'] = df_eng['log_Discharge'].rolling(window=15).mean()
        # Volatility
        df_eng['Discharge_roll7_std'] = df_eng['log_Discharge'].rolling(window=7).std()
        # Rate of change
        df_eng['Discharge_diff1'] = df_eng['log_Discharge'].diff(1)
        # Lag
        df_eng['Discharge_lag1'] = df_eng['log_Discharge'].shift(1)

    '''
    if 'Tw_C' in df_eng.columns:  # Water Temperature
        # SMA
        df_eng['Tw_sma7'] = df_eng['Tw_C'].rolling(window=7).mean()
        df_eng['Tw_lag1'] = df_eng['Tw_C'].shift(1)

        # Interaction
        if 'log_Discharge' in df_eng.columns:
            df_eng['inter_Discharge_Temp'] = df_eng['log_Discharge'] * df_eng['Tw_C']
    '''
    
    # 4. Final Cleanup
    initial_rows = len(df_eng)
    df_eng.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(df_eng)} initial rows due to feature engineering NaNs.")
    print(f"Multivariate feature engineering complete. Final shape: {df_eng.shape}")
    return df_eng


# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) AND FEATURE SELECTION
# =============================================================================

def pre_visualize_eda(df: pd.DataFrame, cfg: Config) -> None:
    """Generates EDA visualizations before feature selection."""
    print("\nPerforming initial EDA on ALL engineered multivariate features...")
    if df.empty: return

    # Correlation Heatmap (All engineered features)
    plt.figure(figsize=(20, 16))
    try:
        # Using Spearman correlation for robustness
        corr_matrix = df.corr(method='spearman')
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Spearman Correlation Heatmap (All Engineered Features vs. log_DOC)', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.PLOTS_DIR, 'pre_feature_selection_multivariate_correlation_heatmap.png'), dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Could not generate correlation heatmap: {e}")


def perform_selection(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Performs VIF-based feature selection."""
    print("\nPerforming VIF feature selection...")
    TARGET = 'log_DOC'

    if not STATSMODELS_AVAILABLE:
        print("Statsmodels not available. Skipping advanced EDA and VIF selection.")
        return df

    # Setup directory for saving EDA statistics
    out_dir = os.path.join(cfg.PLOTS_DIR, "vif_stats_afterengineer")
    os.makedirs(out_dir, exist_ok=True)


    # VIF-based Feature Selection (Iterative Reduction)
    print(f"\n--- VIF-based Feature Selection (Threshold < {cfg.VIF_THRESHOLD}) ---")
    features_df = df.drop(columns=[TARGET], errors='ignore').copy()
    features_df = features_df.select_dtypes(include=np.number)

    if features_df.empty or len(features_df.columns) < 2:
        print("Not enough features for VIF analysis.")
        return df[features_df.columns.tolist() + [TARGET]]

    # Add a constant for VIF calculation
    features_with_const = features_df.assign(const=1)
    vif_history = []
    iteration = 0

    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = features_with_const.columns
        try:
            vif_data["VIF"] = [
                variance_inflation_factor(features_with_const.values.astype(float), i)
                for i in range(features_with_const.shape[1])
            ]
        except Exception as e:
            print(f"Error during VIF calculation (e.g., perfect collinearity): {e}. Stopping.")
            break

        vif_data_no_const = vif_data[vif_data['feature'] != 'const'].copy()

        if vif_data_no_const.empty: break

        max_vif = vif_data_no_const['VIF'].max()
        if max_vif < cfg.VIF_THRESHOLD:
            print("\nAll remaining features have VIF below threshold. Selection complete.")
            # Record final state
            vif_data_no_const["iteration"] = iteration
            vif_data_no_const["dropped_this_iter"] = False
            vif_history.append(vif_data_no_const.copy())
            break

        drop_feature = vif_data_no_const.sort_values('VIF', ascending=False)['feature'].iloc[0]

        # Record history
        vif_data_no_const["iteration"] = iteration
        vif_data_no_const["dropped_this_iter"] = vif_data_no_const["feature"].eq(drop_feature)
        vif_history.append(vif_data_no_const.copy())

        print(f"  - Iteration {iteration}: Dropping '{drop_feature}' (VIF = {max_vif:.2f})")
        features_with_const = features_with_const.drop(columns=[drop_feature])
        iteration += 1

    selected_features = features_with_const.drop(columns=['const'], errors='ignore').columns.tolist()
    print(f"\nFinal selected features: {selected_features}")

    if vif_history:
        pd.concat(vif_history, ignore_index=True).to_csv(os.path.join(out_dir, "vif_results.csv"), index=False)

    final_df = df[selected_features + [TARGET]]
    print(f"Data shape after VIF selection: {final_df.shape}")
    return final_df


def visualize_eda(df: pd.DataFrame, target_col: str, cfg: Config) -> None:
    """Generates EDA visualizations for the selected multivariate features."""
    print("\nPerforming EDA on selected multivariate features (Post-Selection)...")
    if df.empty: return

    # Correlation Heatmap (Post-selection)
    plt.figure(figsize=(16, 12))
    try:
        corr_matrix = df.corr(method='spearman')
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title(f'Spearman Correlation Heatmap (VIF Selected Features vs. {target_col})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.PLOTS_DIR, 'post_selection_multivariate_correlation_heatmap.png'), dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Could not generate post-selection heatmap: {e}")

    # ACF and PACF Plots for the Target Variable (revisited)
    if STATSMODELS_AVAILABLE:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            tsaplots.plot_acf(df[target_col], lags=cfg.ACF_PACF_LAGS, ax=axes[0], title=f'ACF for {target_col}')
            tsaplots.plot_pacf(df[target_col], lags=cfg.ACF_PACF_LAGS, ax=axes[1], title=f'PACF for {target_col}',
                               method='ywm')
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.PLOTS_DIR, 'doc_acf_pacf_plots_final.png'), dpi=300)
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Could not plot final ACF/PACF for {target_col}: {e}")


# =============================================================================
# DATA PREPARATION FOR MODELING
# =============================================================================

def prepare_data_for_model(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Splits, scales, and sequences data for the RNN model, ensuring no data leakage."""
    print("\nScaling and sequencing data...")
    if df.empty: return {}

    TARGET = 'log_DOC'
    feature_cols = [c for c in df.columns if c != TARGET]
    X, y = df[feature_cols], df[TARGET]

    # 1. Temporal Split (Chronological order must be maintained)
    n = len(X)
    n_train = int(n * cfg.TRAIN_SPLIT)
    n_val = int(n * cfg.VAL_SPLIT)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    print(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    if len(X_test) < cfg.WINDOW_SIZE or len(X_val) < cfg.WINDOW_SIZE:
        print("Error: Validation or Test set is smaller than the window size. Cannot create sequences.")
        return {}

    # 2. Scale Data (MinMaxScaler)
    # CRITICAL: Fit scalers ONLY on the training data to prevent leakage from future data.
    X_scaler, y_scaler = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))

    X_train_sc = X_scaler.fit_transform(X_train)
    y_train_sc = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Transform validation and test sets using the training scalers
    X_val_sc = X_scaler.transform(X_val)
    y_val_sc = y_scaler.transform(y_val.values.reshape(-1, 1))
    X_test_sc = X_scaler.transform(X_test)
    y_test_sc = y_scaler.transform(y_test.values.reshape(-1, 1))

    # 3. Create Sequences (Sliding Window)
    def create_sequences(X_data: np.ndarray, y_data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Converts time series data into sequences (X) and targets (y)."""
        X_seq, y_seq = [], []
        for i in range(window_size, len(X_data)):
            # Input sequence: from (i-window_size) up to (i-1)
            X_seq.append(X_data[i - window_size:i, :])
            # Target: at time i
            y_seq.append(y_data[i])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train_sc, y_train_sc, cfg.WINDOW_SIZE)
    X_val_seq, y_val_seq = create_sequences(X_val_sc, y_val_sc, cfg.WINDOW_SIZE)
    X_test_seq, y_test_seq = create_sequences(X_test_sc, y_test_sc, cfg.WINDOW_SIZE)

    # 4. Create Sample Weights
    train_sample_weights = 1.0 + (y_train_seq.flatten() * cfg.SAMPLE_WEIGHT_FACTOR)

    print(f"Sequencing complete. Train shape: {X_train_seq.shape}, Test shape: {X_test_seq.shape}")

    # Ensure data types are float32 for efficient GPU computation
    return {
        "X_train_seq": X_train_seq.astype('float32'), "y_train_seq": y_train_seq.astype('float32'),
        "X_val_seq": X_val_seq.astype('float32'), "y_val_seq": y_val_seq.astype('float32'),
        "X_test_seq": X_test_seq.astype('float32'), "y_test_seq": y_test_seq.astype('float32'),
        "y_test_dates": y_test.index[cfg.WINDOW_SIZE:],
        "y_scaler": y_scaler, "X_scaler": X_scaler,
        "train_sample_weights": train_sample_weights.astype('float32')
    }


# =============================================================================
# MODEL BUILDING AND TRAINING
# =============================================================================

def build_model(input_shape: Tuple[int, int], cfg: Config) -> Sequential:
    """Builds the 2-layer Bidirectional LSTM/GRU hybrid model optimized for cuDNN."""
    # Note: Default parameters (activation='tanh', recurrent_dropout=0) ensure cuDNN compatibility.

    model = Sequential(name="Hybrid_BiLSTM_BiGRU_Forecaster")
    model.add(Input(shape=input_shape))

    # Layer 1: Bidirectional LSTM
    model.add(Bidirectional(
        LSTM(cfg.LSTM_UNITS_1,
             return_sequences=True,  # Pass sequence to the next RNN layer
             kernel_regularizer=l2(cfg.L2_REG)),
        name="BiLSTM_Layer1"
    ))
    # LayerNormalization stabilizes training in RNNs
    model.add(LayerNormalization(name="LayerNorm1"))
    model.add(Dropout(cfg.DROPOUT_RATE, name="Dropout1"))

    # Layer 2: Bidirectional GRU
    model.add(Bidirectional(
        GRU(cfg.GRU_UNITS_2,
            return_sequences=False,  # Only need the final state from the last RNN layer
            kernel_regularizer=l2(cfg.L2_REG)),
        name="BiGRU_Layer2"
    ))
    model.add(LayerNormalization(name="LayerNorm2"))
    model.add(Dropout(cfg.DROPOUT_RATE, name="Dropout2"))

    # Dense Layer for interpretation
    # 'swish' activation often outperforms 'relu'
    model.add(Dense(cfg.DENSE_UNITS, activation='swish', kernel_regularizer=l2(cfg.L2_REG), name="Dense_Interp"))

    # Output Layer
    # 'linear' activation for regression. Explicit float32 for numerical stability.
    model.add(Dense(1, activation='linear', dtype='float32', name="Output"))

    # Compilation
    optimizer = AdamW(learning_rate=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    # LogCosh loss is robust to outliers
    model.compile(loss=LogCosh(),
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


def train_model(model: Sequential, data: Dict[str, Any], cfg: Config, model_path: str) -> Tuple[History, Sequential]:
    """Trains the model using an efficient tf.data pipeline."""
    print(f"\nStarting training. Saving best model to '{model_path}'...")
    model.summary()

    # Callbacks configuration
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=cfg.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=cfg.LR_PLATEAU_PATIENCE, min_lr=1e-7, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=1)
    ]

    # Create tf.data Datasets for efficient input pipeline (Best Practice)
    # Training dataset (includes sample weights)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (data['X_train_seq'], data['y_train_seq'], data['train_sample_weights'])
    )
    # Optimization: Cache, shuffle, batch, and prefetch
    train_ds = train_ds.cache().shuffle(buffer_size=len(data['X_train_seq'])).batch(cfg.BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(
        (data['X_val_seq'], data['y_val_seq'])
    ).batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Training execution
    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=cfg.EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2  # One line per epoch
    )
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    # Load the best model saved by ModelCheckpoint
    try:
        best_model = load_model(model_path)
        print(f"Successfully loaded best model from {model_path}")
        return history, best_model
    except Exception as e:
        print(f"Warning: Could not load best model from {model_path}. Returning last epoch model. Error: {e}")
        return history, model


def visualize_training_history(history: History, cfg: Config, model_name: str) -> None:
    """Plots and saves the training and validation loss curves."""
    if not history or not history.history:
        return

    hist_df = pd.DataFrame(history.history)
    # Find the epoch with the minimum validation loss
    best_epoch = hist_df['val_loss'].idxmin() + 1
    min_val_loss = hist_df['val_loss'].min()

    save_path = os.path.join(cfg.PLOTS_DIR, f'{model_name}_training_history.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name.capitalize()} Model Training History (Best Epoch: {best_epoch})', fontsize=18)

    # Plot Loss (LogCosh)
    ax1.plot(hist_df['loss'], label='Train Loss', color='blue')
    ax1.plot(hist_df['val_loss'], label='Validation Loss', color='orange')
    ax1.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch')
    ax1.scatter(best_epoch - 1, min_val_loss, color='r', marker='o', s=100, zorder=5)
    ax1.set_title('Model Loss (LogCosh)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot MAE (Scaled)
    ax2.plot(hist_df['mae'], label='Train MAE', color='blue')
    ax2.plot(hist_df['val_mae'], label='Validation MAE', color='orange')
    ax2.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch')
    ax2.set_title('Model MAE (Scaled)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
    """Calculates and prints comprehensive regression performance metrics."""
    # Ensure inputs are flat arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Standard Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Percentage Errors
    # Epsilon prevents division by zero
    epsilon = 1e-10

    # WAPE (Weighted Absolute Percentage Error) - Preferred metric, robust
    sum_abs_true = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / np.maximum(sum_abs_true, epsilon) * 100

    # MAPE (Mean Absolute Percentage Error) - Use with caution if data contains zeros/small values
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

    metrics = {
        "R2": r2, "RMSE": rmse, "MAE": mae,
        "WAPE": wape, "MAPE": mape
    }

    # Print results
    print("\n" + "=" * 65)
    print(f"--- Performance Metrics on {dataset_name} (Original Scale) ---")
    print(f"  R-squared (R)                         : {r2:.4f}")
    print("-" * 65)
    print("  Error Metrics:")
    print(f"  Root Mean Squared Error (RMSE)         : {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE)              : {mae:.4f}")
    print("-" * 65)
    print("  Percentage Errors:")
    print(f"  Weighted Absolute Percentage Error (WAPE) : {wape:.2f}%")
    print(f"  Mean Absolute Percentage Error (MAPE)     : {mape:.2f}%")
    print("=" * 65)

    return metrics


def evaluate_model_with_uncertainty(
        model: Sequential,
        data: Dict[str, Any],
        cfg: Config,
        model_name: str
) -> None:
    """Evaluates the model on the test set using MC Dropout for uncertainty estimation."""
    print(f"\nEvaluating {model_name} model on test set with uncertainty...")

    # Helper function for inverse transformation (Scaling -> Log -> Original)
    def inverse_transform(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        inversed_log = scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()
        # Inverse log1p transformation (expm1)
        inversed_orig = np.expm1(inversed_log)
        return np.maximum(0, inversed_orig)  # Ensure non-negative

    y_true_orig = inverse_transform(data['y_test_seq'], data['y_scaler'])

    # --- Monte Carlo Dropout (MC Dropout) ---
    print(f"Performing MC Dropout ({cfg.MC_DROPOUT_SAMPLES} samples)...")

    # Use tf.function to optimize the prediction loop
    @tf.function
    def predict_mc(inputs):
        # Setting training=True enables dropout during inference
        return model(inputs, training=True)

    mc_predictions_scaled = []
    for _ in range(cfg.MC_DROPOUT_SAMPLES):
        preds = predict_mc(data['X_test_seq']).numpy()
        mc_predictions_scaled.append(preds)

    # Stack predictions: Shape (N_SAMPLES, N_TEST_POINTS, 1)
    mc_stack_sc = np.stack(mc_predictions_scaled)

    # Calculate statistics in the scaled domain
    mean_preds_sc = mc_stack_sc.mean(axis=0).flatten()
    # 95% Prediction Interval
    lower_bound_sc = np.quantile(mc_stack_sc, 0.025, axis=0).flatten()
    upper_bound_sc = np.quantile(mc_stack_sc, 0.975, axis=0).flatten()

    # Transform statistics back to the original scale
    y_pred_mc_mean = inverse_transform(mean_preds_sc, data['y_scaler'])
    y_pred_lower = inverse_transform(lower_bound_sc, data['y_scaler'])
    y_pred_upper = inverse_transform(upper_bound_sc, data['y_scaler'])

    # --- Evaluation Metrics ---
    metrics = evaluate_performance(y_true_orig, y_pred_mc_mean, f"{model_name} Test Set (MC Mean)")
    
    # --- Save metrics & outputs ---
    metrics_dir = os.path.join(cfg.PLOTS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

   # Per-model metrics file
    metrics_row = {"model": model_name, "n_test": int(len(y_true_orig)), **metrics}
    pd.DataFrame([metrics_row]).to_csv( 
       os.path.join(metrics_dir, f"{model_name.lower()}_metrics.csv"), index=False
       )

   # Combined metrics (append/replace this model)
    combined_path = os.path.join(metrics_dir, "all_models_metrics.csv")
    try:
        if os.path.exists(combined_path):
            existing = pd.read_csv(combined_path)
            existing = existing[existing["model"] != model_name]
            out = pd.concat([existing, pd.DataFrame([metrics_row])], ignore_index=True)
        else: 
            out = pd.DataFrame([metrics_row])
        out.to_csv(combined_path, index=False)
    except Exception as e:
        print(f"Could not update combined metrics file: {e}")

    # Prepare results DataFrame for visualization
    results_df = pd.DataFrame({
        'Observed': y_true_orig,
        'Predicted': y_pred_mc_mean,
        'Lower_CI': y_pred_lower,
        'Upper_CI': y_pred_upper
    }, index=data['y_test_dates'])
    
    # Save predictions & residuals, useful for later analysis
    results_df.to_csv(os.path.join(metrics_dir, f"{model_name.lower()}_predictions.csv"))
    residuals = results_df['Observed'] - results_df['Predicted']
    residuals.to_csv(os.path.join(metrics_dir, f"{model_name.lower()}_residuals.csv"), header=['residual'])

    # --- Visualization ---

    # Time Series Plot
    plt.figure(figsize=(16, 8))
    plt.plot(results_df.index, results_df['Observed'], label='Observed DOC', color='black', linewidth=1.5)
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted DOC (Mean)', color='royalblue', linestyle='--')
    plt.fill_between(results_df.index, results_df['Lower_CI'], results_df['Upper_CI'], color='royalblue', alpha=0.3,
                     label='95% Prediction Interval')

    plt.title(f'{model_name} Model: Observed vs. Predicted DOC with Uncertainty', fontsize=18)
    plt.xlabel('Date');
    plt.ylabel('DOC (ppm)');
    plt.legend(loc='upper left');
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f'{model_name.lower()}_timeseries_prediction.png'), dpi=300)
    plt.show()
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.regplot(x='Observed', y='Predicted', data=results_df, scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red', 'label': 'Regression Line'})
    # 1:1 Line
    min_val = min(results_df.Observed.min(), results_df.Predicted.min())
    max_val = max(results_df.Observed.max(), results_df.Predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')

    r2_val = metrics.get("R2", 0.0)
    plt.title(f'{model_name} Model: Observed vs. Predicted (R={r2_val:.4f})', fontsize=16)
    plt.xlabel('Observed DOC (ppm)');
    plt.ylabel('Predicted DOC (ppm)');
    plt.axis('equal');
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f'{model_name.lower()}_scatter_prediction.png'), dpi=300)
    plt.show()
    plt.close()

    # 3. Residuals Plot
    residuals = results_df['Observed'] - results_df['Predicted']
    plt.figure(figsize=(16, 6))
    plt.plot(results_df.index, residuals, color='purple', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'{model_name} Model: Prediction Residuals Over Time', fontsize=16)
    plt.xlabel('Date');
    plt.ylabel('Residuals (Observed - Predicted)');
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f'{model_name.lower()}_residuals_plot.png'), dpi=300)
    plt.show()
    plt.close()


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def run_benchmark_model(df_clean: pd.DataFrame, cfg: Config) -> None:
    """Runs the pipeline for the univariate benchmark model."""
    print("\n" + "=" * 80 + "\n" + "RUNNING BENCHMARK UNIVARIATE MODEL".center(80) + "\n" + "=" * 80)

    # 1. Feature Engineering (Univariate, ensuring no leakage)
    df_uni = engineer_univariate_features(df_clean)
    if df_uni.empty: return

    # 2. Data Preparation
    data_dict = prepare_data_for_model(df_uni, cfg)
    if not data_dict: return

    # 3. Model Building
    input_shape = (data_dict['X_train_seq'].shape[1], data_dict['X_train_seq'].shape[2])
    model = build_model(input_shape, cfg)

    # 4. Training
    history, best_model = train_model(model, data_dict, cfg, model_path=cfg.BENCHMARK_MODEL_SAVE_PATH)

    # 5. Evaluation
    visualize_training_history(history, cfg, model_name="benchmark")
    evaluate_model_with_uncertainty(best_model, data_dict, cfg, model_name="Benchmark")
    
    print(f"Saved univariate metrics to: {os.path.join(cfg.PLOTS_DIR, 'metrics', 'benchmark_metrics.csv')}")
    
    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 80 + "\n" + "BENCHMARK MODEL RUN COMPLETE".center(80) + "\n" + "=" * 80)


def main() -> None:
    """Main execution function."""
    cfg = Config()
    set_seed(cfg.SEED)
    setup_environment(cfg)

    # Data Loading and Cleaning
    df_raw = load_and_clean_data(cfg)
    if df_raw.empty:
        print("Execution halted due to data loading failure.")
        return

    # --- 1. Run Benchmark Model ---
    run_benchmark_model(df_raw, cfg)

    # --- 2. Run Main Multivariate Model ---
    print("\n" + "=" * 80 + "\n" + "RUNNING MAIN MULTIVARIATE MODEL".center(80) + "\n" + "=" * 80)

    # 2a. Feature Engineering (Multivariate)
    df_featured = engineer_features(df_raw)
    if df_featured.empty: return

    # 2b. EDA and Feature Selection
    pre_visualize_eda(df_featured, cfg=cfg)
    df_selected = perform_selection(df_featured, cfg)
    if df_selected.empty or len(df_selected.columns) <= 1:
        print("Main model halted: Insufficient features after selection.")
        return
    visualize_eda(df_selected, 'log_DOC', cfg=cfg)

    # 2c. Data Preparation
    data_dict_multi = prepare_data_for_model(df_selected, cfg)
    if not data_dict_multi: return

    # 2d. Model Building
    input_shape = (data_dict_multi['X_train_seq'].shape[1], data_dict_multi['X_train_seq'].shape[2])
    model_multi = build_model(input_shape, cfg)

    # 2e. Training
    history_multi, best_model_multi = train_model(model_multi, data_dict_multi, cfg, model_path=cfg.MODEL_SAVE_PATH)

    # 2f. Evaluation
    visualize_training_history(history_multi, cfg, model_name="multivariate")
    evaluate_model_with_uncertainty(best_model_multi, data_dict_multi, cfg, model_name="Multivariate")

    print("\n" + "=" * 80 + "\n" + "MAIN MULTIVARIATE MODEL RUN COMPLETE".center(80) + "\n" + "=" * 80)


if __name__ == '__main__':
    # Robust execution wrapper
    try:
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
