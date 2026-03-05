#market maker
"""
Minute-Level Model Training & Evaluation Pipeline
------------------------------------------------------
Features:
- "Sniper" Mode: Trade only when confidence > 0.60
- Anti-Overfitting: Aggressive Regularization (L2, Gamma, Min Child Weight)
- Real Cost Simulation: IBKR Commissions + Slippage
- Ensemble Architecture (XGBoost + Random Forest on GPU)
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List

# Sklearn Imports
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    f1_score, roc_auc_score, average_precision_score, log_loss
)
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone

# XGBoost
import xgboost as xgb

SHAP_AVAILABLE = False

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook")

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
TICKER = "AAPL"
DATA_DIR = "./data"
INPUT_FILE = f"{DATA_DIR}/{TICKER}.US_minute_engineered_v3.csv"
OUTPUT_DIR = "./data/model_results_v4"

# 1. Trading Logic
TARGET_HORIZON = 1  # Predict next minute
CONFIDENCE_THRESHOLD = 0.55  # Only Buy if Prob > 60% (Sniper Mode)
THRESHOLD = 0.0000  # 0.0 = Simple Up/Down target generation

# 2. Transaction Costs (IBKR Pro + Spread)
# IBKR Fixed: $0.005 per share.
# Spread/Slippage: Est. 1 basis point (0.01%) on entry.
AVG_PRICE_EST = 75.0   # Approx price of ticker (for converting fixed fee to %)
FIXED_COMMISSION = 0.005
SLIPPAGE_BPS = 0.0001

# 3. Training Config
TEST_SIZE_PCT = 0.20
CV_SPLITS = 5
ITERATIONS = 20     # Increased iterations for tighter grid
RANDOM_STATE = 42

# GPU Configuration
USE_GPU = True
DEVICE = 'cuda' if USE_GPU else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# 1. Data Loading & Target Generation
# --------------------------------------------------------------------
def load_and_prep_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features, generate target, and return data + raw prices for backtest."""
    print(f"Loading data from {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Handle Index
    date_col = None
    for c in df.columns:
        if c.lower() in ['date', 'datetime', 'timestamp', 'time']:
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
    else:
        print("Warning: No date column found. Using integer index.")

    # Keep raw prices for backtesting later
    raw_prices = df[['close']].copy()

    # Generate Target: 1 if Return(t+horizon) > Threshold
    future_returns = df['close'].pct_change(periods=TARGET_HORIZON).shift(-TARGET_HORIZON)
    y = (future_returns > THRESHOLD).astype(int)

    # Valid indices (drop NaNs created by shifting or feature engineering)
    valid_indices = ~(df.isna().any(axis=1) | y.isna())

    X = df.loc[valid_indices].copy()
    y = y.loc[valid_indices].copy()
    raw_prices = raw_prices.loc[valid_indices].copy()

    # Drop non-feature columns
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'returns',
                 'log_returns', 'gap_flag', 'target']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

    print(f"Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class Balance: {y.mean():.2%} positive class")

    return X, y, raw_prices

# --------------------------------------------------------------------
# 2. Time Series Splitting
# --------------------------------------------------------------------
def time_series_train_test_split(X, y, prices, test_size=0.2):
    """Split data respecting temporal order."""
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    prices_train, prices_test = prices.iloc[:split_idx], prices.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, prices_test

# --------------------------------------------------------------------
# 3. Model Optimization (XGBoost GPU with Aggressive Regularization)
# --------------------------------------------------------------------
def optimize_xgboost_gb(X_train, y_train):
    """
    Gradient Boosting optimization with aggressive regularization
    to fix the diverging loss curve.
    """
    print("\n" + "="*60)
    print(f"OPTIMIZING XGBOOST (GPU: {DEVICE}) - {ITERATIONS} Iterations")
    print("="*60)

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device=DEVICE,
        random_state=RANDOM_STATE
    )

    # Aggressive Grid against Overfitting
    param_dist = {
        'n_estimators': [150, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05],     # Slower learning
        'max_depth': [3, 4, 5],                  # Shallower trees (less memory/overfit)
        'min_child_weight': [5, 10, 20],         # Require more data per leaf (Kills noise)
        'subsample': [0.5, 0.6, 0.7],            # See fewer rows per tree
        'colsample_bytree': [0.5, 0.6, 0.7],     # See fewer features per tree
        'reg_alpha': [0.1, 1.0, 5.0],            # L1 Regularization
        'reg_lambda': [10.0, 50.0, 100.0],       # Strong L2 Regularization (Critical)
        'gamma': [1.0, 5.0]                      # Minimum split loss
    }

    # TimeSeriesSplit for Cross-Validation
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=ITERATIONS,
        scoring='precision', # Optimized for Precision now (for Sniper mode)
        cv=tscv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"Best Parameters: {search.best_params_}")
    print(f"Best CV Precision: {search.best_score_:.4f}")

    return search.best_estimator_

def get_gpu_random_forest():
    """XGBoost configured as Random Forest on GPU"""
    print(f"\nCONFIGURING RANDOM FOREST (GPU: {DEVICE})...")
    return xgb.XGBClassifier(
        n_estimators=1,
        num_parallel_tree=300,
        learning_rate=1.0,
        max_depth=5,            # Reduced depth
        subsample=0.6,          # More bagging
        colsample_bynode=0.6,   # More feature bagging
        reg_lambda=10.0,        # Added regularization to RF
        objective='binary:logistic',
        tree_method='hist',
        device=DEVICE,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# --------------------------------------------------------------------
# 4. Diagnostics & Learning Curves
# --------------------------------------------------------------------
def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    """Plot Training vs Validation Loss."""
    print("Generating Learning Curves...")

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Loss: Training vs Validation')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
    plt.close()

# --------------------------------------------------------------------
# 5. Financial Diagnostics (Net of Costs) & Sniper Logic
# --------------------------------------------------------------------
def calculate_net_returns(model, X_test, prices_test):
    print(f"\nCALCULATING NET RETURNS (Threshold: {CONFIDENCE_THRESHOLD})...")

    # Get Probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Apply Sniper Threshold
    # If prob > 0.60, Signal = 1. Else 0.
    signals = (probs > CONFIDENCE_THRESHOLD).astype(int)

    # Calculate Costs
    # 1. Commission % = Fixed Fee / Stock Price
    comm_pct = FIXED_COMMISSION / AVG_PRICE_EST
    # 2. Total Cost % per trade = Commission + Slippage
    cost_per_trade_pct = comm_pct + SLIPPAGE_BPS

    print(f"  - Est. Stock Price: ${AVG_PRICE_EST}")
    print(f"  - IBKR Comm: ${FIXED_COMMISSION} ({comm_pct*10000:.2f} bps)")
    print(f"  - Slippage: {SLIPPAGE_BPS*10000:.2f} bps")
    print(f"  - Total Cost per Trade: {cost_per_trade_pct*100:.4f}%")

    # Market Returns (Close-to-Close of NEXT minute)
    # We buy at T (signal), hold for 1 min.
    market_returns = prices_test.pct_change().shift(-1).fillna(0)['close']
    market_returns = market_returns.iloc[:len(signals)]

    # Gross Strategy Returns
    gross_returns = market_returns * signals

    # Calculate Cost Drag
    # We pay cost when we ENTER a trade (assuming exit is frictionless or included in slippage)
    # Simplification: Pay cost every time signal is 1 (entry)
    cost_drag = signals * cost_per_trade_pct

    # Net Returns
    net_returns = gross_returns - cost_drag

    # Plotting
    cum_market = (1 + market_returns).cumprod()
    cum_gross = (1 + gross_returns).cumprod()
    cum_net = (1 + net_returns).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(cum_market, label="Buy & Hold", color='gray', alpha=0.5)
    plt.plot(cum_gross, label="Model Gross", color='green', alpha=0.6, linestyle='--')
    plt.plot(cum_net, label="Model Net (Real)", color='blue', linewidth=2)

    plt.title(f"Net Profit Analysis (Thresh > {CONFIDENCE_THRESHOLD})")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/net_cumulative_returns.png")
    plt.close()

    total_trades = signals.sum()
    final_return = cum_net.iloc[-1] - 1
    print(f"\n  RESULTS:")
    print(f"  - Trades Taken: {total_trades}")
    print(f"  - Final Net Return: {final_return:.2%}")

    return signals, probs

def plot_custom_confusion_matrix(y_test, signals):
    """Confusion Matrix based on the custom threshold"""
    cm = confusion_matrix(y_test, signals)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Threshold > {CONFIDENCE_THRESHOLD})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_threshold.png")
    plt.close()

def plot_calibration_curve_func(y_test, probs):
    """Check if probability output is reliable."""
    print("Generating Calibration Curve...")

    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/calibration_curve.png")
    plt.close()

def check_permutation_importance(model, X_val, y_val):
    """Check feature stability."""
    print("Calculating Permutation Importance...")
    if len(X_val) > 2000:
        X_sub = X_val.sample(2000, random_state=42)
        y_sub = y_val.loc[X_sub.index]
    else:
        X_sub, y_sub = X_val, y_val

    result = permutation_importance(
        model, X_sub, y_sub, n_repeats=5, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()[-20:]
    plt.figure(figsize=(10, 8))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=X_val.columns[sorted_idx]
    )
    plt.title("Permutation Importances (Validation Set)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/permutation_importance.png")
    plt.close()

def plot_shap_summary(model, X_test):
    """SHAP Interpretability."""
    if not SHAP_AVAILABLE:
        return
    print("Generating SHAP Summary...")
    try:
        if isinstance(model, VotingClassifier):
            estimator = model.named_estimators_['gb']
        else:
            estimator = model

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test.iloc[:1000])

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test.iloc[:1000], show=False)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_summary.png")
        plt.close()
    except Exception as e:
        print(f"SHAP Error: {e}")

# --------------------------------------------------------------------
# 6. Main Pipeline
# --------------------------------------------------------------------
def main():
    gc.collect()
    print("="*60)
    print(f"MODEL TRAINING v4 (Sniper Mode > {CONFIDENCE_THRESHOLD})")
    print("="*60)

    # 1. Load Data
    try:
        X, y, prices = load_and_prep_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test, prices_test = time_series_train_test_split(
        X, y, prices, test_size=TEST_SIZE_PCT
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Optimize XGBoost (GPU)
    # Using stricter regularization grid
    best_xgb = optimize_xgboost_gb(X_train, y_train)

    # 3b. Learning Curve Check
    val_cut = int(len(X_train) * 0.9)
    X_tr_plot, X_val_plot = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_tr_plot, y_val_plot = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

    xgb_plotter = clone(best_xgb)
    plot_learning_curves(xgb_plotter, X_tr_plot, y_tr_plot, X_val_plot, y_val_plot)
    del xgb_plotter, X_tr_plot, X_val_plot, y_tr_plot, y_val_plot
    gc.collect()

    # 4. Random Forest on GPU
    rf_model = get_gpu_random_forest()
    rf_model.fit(X_train, y_train)

    # 5. Ensemble (Voting)
    print("\nTraining Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('gb', best_xgb),
            ('rf', rf_model)
        ],
        voting='soft',
        n_jobs=1
    )
    ensemble.fit(X_train, y_train)

    # 6. Comprehensive Evaluation with Real Constraints
    print("\n" + "="*60)
    print("FINAL EVALUATION & DIAGNOSTICS")
    print("="*60)

    # Calculate Net Returns and get Signals/Probs
    signals, probs = calculate_net_returns(ensemble, X_test, prices_test)

    # Metrics based on threshold
    plot_custom_confusion_matrix(y_test, signals)

    # Additional Diagnostics
    plot_calibration_curve_func(y_test, probs)
    check_permutation_importance(ensemble, X_test, y_test)
    plot_shap_summary(ensemble, X_test)

    print("\n" + "="*60)
    print("DONE. Results saved to:", OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()
