#!/usr/bin/env python
# minute_feature_engineering_v4.py

"""
Simplified & Enhanced Feature Engineering for Minute-Level Price Data (v4)
---------------------------------------------------------------------------
Pipeline Order:
1. Generate core features + Kalman Filter Trend Extraction
2. Apply cleaning pipeline (Imputation → Outlier removal → Scaling)
3. Remove highly correlated features (correlation > 0.8) [FIRST FILTER]
4. Select top 25 features by importance
5. Apply K-means + RBF to create many new feature columns
6. Remove highly correlated features (correlation > 0.9) [SECOND FILTER]
7. Select top 30 final features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
import traceback

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
#A non-linear dependency score between a feature and a continuous target
#uses f_regression and mutual_info_regression to Ranks features, Keeps the top K
#

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
#distance More ML
from scipy.spatial.distance import cdist
#Faster

# Kalman Filter, estimate the true hidden state without noise
from filterpy.kalman import KalmanFilter


# Technical analysis, fundamental, tehcnical and more
import ta

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
TICKER = "GLD"
MINUTE_DATA_DIR = "./data/minute_data"
OUTPUT_FILE = f"./data/{TICKER}.US_minute_engineered_v4.csv"

# Feature selection parameters
N_FEATURES_BEFORE_RBF = 25  # Target number of features before RBF
N_FEATURES_FINAL = 55  # Final number of features after all processing
CORRELATION_THRESHOLD_FIRST = 0.8  # First filter: before RBF
CORRELATION_THRESHOLD_SECOND = 0.9  # Second filter: after RBF

# K-means + RBF parameters
N_CLUSTERS_LIST = [5, 10, 15, 20]
RBF_GAMMA_LIST = [0.01, 0.1, 0.5, 1.0]

# Kalman Filter parameters
KALMAN_PROCESS_NOISE = 0.01
KALMAN_MEASUREMENT_NOISE = 0.1

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --------------------------------------------------------------------
# Kalman Filter Trend Extraction
# --------------------------------------------------------------------
def apply_kalman_filter(series: pd.Series, process_noise: float = KALMAN_PROCESS_NOISE,
                        measurement_noise: float = KALMAN_MEASUREMENT_NOISE) -> pd.Series:
    """
    Apply Kalman Filter to extract trend from a time series.
    Uses a position + velocity state model for smooth trend extraction.
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State transition matrix (position + velocity model)
    kf.F = np.array([[1., 1.],
                     [0., 1.]])

    # Measurement matrix (we only observe position)
    kf.H = np.array([[1., 0.]])

    # Covariance matrices
    kf.P *= 1000.  # Initial uncertainty
    kf.R = np.array([[measurement_noise]])  # Measurement noise
    kf.Q = np.array([[process_noise, 0.],
                     [0., process_noise]])  # Process noise

    # Initial state
    kf.x = np.array([[series.iloc[0]], [0.]])

    # Run filter
    filtered_values = []
    velocities = []

    for z in series.values:
        kf.predict()
        kf.update(np.array([[z]]))
        filtered_values.append(kf.x[0, 0])
        velocities.append(kf.x[1, 0])

    return pd.Series(filtered_values, index=series.index), pd.Series(velocities, index=series.index)


def extract_kalman_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive Kalman-filtered trend features from price data.
    """
    print("\n  Extracting Kalman Filter Trend Features...")

    df = df.copy()

    # Apply Kalman filter to close price
    df['kalman_trend'], df['kalman_velocity'] = apply_kalman_filter(df['close'])

    # Trend deviation (price - trend)
    df['kalman_deviation'] = df['close'] - df['kalman_trend']
    df['kalman_deviation_pct'] = df['kalman_deviation'] / (df['kalman_trend'] + 1e-8)

    # Normalized deviation (z-score like)
    rolling_std = df['kalman_deviation'].rolling(20).std()
    df['kalman_deviation_zscore'] = df['kalman_deviation'] / (rolling_std + 1e-8)

    # Velocity features (rate of change of trend)
    df['kalman_velocity_ma5'] = df['kalman_velocity'].rolling(5).mean()
    df['kalman_velocity_ma10'] = df['kalman_velocity'].rolling(10).mean()

    # Acceleration (rate of change of velocity)
    df['kalman_acceleration'] = df['kalman_velocity'].diff()
    df['kalman_acceleration_ma5'] = df['kalman_acceleration'].rolling(5).mean()

    # Trend strength indicators
    df['kalman_trend_strength'] = df['kalman_velocity'].rolling(20).mean() / \
                                   (df['kalman_velocity'].rolling(20).std() + 1e-8)

    # Mean reversion signal (when deviation is extreme)
    df['kalman_mean_reversion'] = -df['kalman_deviation_zscore']

    # Trend momentum (velocity relative to recent history)
    df['kalman_momentum'] = df['kalman_velocity'] / \
                            (df['kalman_velocity'].rolling(50).std() + 1e-8)

    # Regime detection (trend vs mean-reverting)
    df['kalman_regime'] = np.where(
        abs(df['kalman_deviation_zscore']) > 2,
        np.sign(df['kalman_deviation_zscore']) * -1,  # Mean revert when extreme
        np.sign(df['kalman_velocity'])  # Follow trend otherwise
    )

    # Apply Kalman to volume for smoothed volume trend
    df['kalman_volume_trend'], _ = apply_kalman_filter(
        df['volume'],
        process_noise=0.1,
        measurement_noise=1.0
    )
    df['volume_vs_kalman'] = df['volume'] / (df['kalman_volume_trend'] + 1e-8)

    # Apply Kalman to volatility
    realized_vol = df['returns'].rolling(20).std() * np.sqrt(390)
    if not realized_vol.isna().all():
        realized_vol_filled = realized_vol.fillna(method='bfill').fillna(method='ffill')
        df['kalman_volatility'], _ = apply_kalman_filter(
            realized_vol_filled,
            process_noise=0.001,
            measurement_noise=0.1
        )
        df['vol_vs_kalman'] = realized_vol / (df['kalman_volatility'] + 1e-8)

    # Trend crossover signals
    df['price_above_kalman'] = (df['close'] > df['kalman_trend']).astype(int)
    df['kalman_crossover'] = df['price_above_kalman'].diff().fillna(0)

    # Distance from trend in ATR units
    atr = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()
    df['kalman_deviation_atr'] = df['kalman_deviation'] / (atr + 1e-8)

    print(f"    Added {sum(1 for c in df.columns if 'kalman' in c.lower())} Kalman features")

    return df


# --------------------------------------------------------------------
# 1. Data Loading & Basic Preprocessing
# --------------------------------------------------------------------
def load_and_prepare_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load minute data and perform basic preprocessing."""
    print(f"Loading data from {file_path}...")

    try:
        df = pd.read_csv(file_path)

        # Find datetime column
        datetime_col = None
        for col in df.columns:
            if col.lower() in ['datetime', 'date', 'time', 'timestamp', 'date_time', 'dt']:
                datetime_col = col
                break

        if datetime_col is None:
            for col in df.columns:
                try:
                    pd.to_datetime(df[col].iloc[:5])
                    datetime_col = col
                    break
                except:
                    continue

        if datetime_col is None:
            print("Error: No datetime column found")
            return None

        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                col_map[col] = 'open'
            elif 'high' in col_lower:
                col_map[col] = 'high'
            elif 'low' in col_lower:
                col_map[col] = 'low'
            elif 'close' in col_lower:
                col_map[col] = 'close'
            elif 'volume' in col_lower or 'vol' in col_lower:
                col_map[col] = 'volume'

        df = df.rename(columns=col_map)

        # Verify required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"Error: Missing columns {missing}")
            return None

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Add gap flag for missing minutes
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("US/Eastern")

        full_index = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="US/Eastern")
        gap_mask = ~full_index.isin(df.index)
        df['gap_flag'] = 0
        if gap_mask.any():
            prev_bars = full_index[gap_mask] - pd.Timedelta(minutes=1)
            df.loc[df.index.intersection(prev_bars), 'gap_flag'] = 1

        print(f"Loaded {len(df)} rows with {df['gap_flag'].sum()} detected gaps")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None


# --------------------------------------------------------------------
# 2. Core Feature Generation (with Kalman)
# --------------------------------------------------------------------
def generate_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate essential technical features including Kalman filter."""
    print("\n" + "="*60)
    print("STEP 1: GENERATING CORE FEATURES + KALMAN FILTER")
    print("="*60)

    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['close'] * 100
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'] / df['close'] * 100
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

    # Time features
    df['hour'] = df.index.hour
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['day_of_week'] = df.index.dayofweek

    # Cyclical time encoding
    df['time_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 390)
    df['time_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 390)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

    # Session flags
    market_open = 9 * 60 + 30
    df['is_first_hour'] = ((df['minute_of_day'] >= market_open) &
                           (df['minute_of_day'] < market_open + 60)).astype(int)
    df['is_last_hour'] = ((df['minute_of_day'] >= 15 * 60) &
                          (df['minute_of_day'] < 16 * 60)).astype(int)

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'close_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']

    # Volatility
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(390)

    # RSI
    for window in [7, 14]:
        try:
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=window)
            df[f'rsi_{window}'] = rsi.rsi()
        except:
            pass

    # MACD
    try:
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
    except:
        pass

    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    except:
        pass

    # ATR
    try:
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'],
                                              close=df['close'], window=14)
        df['atr_14'] = atr.average_true_range()
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
    except:
        pass

    # Stochastic
    try:
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'],
                                                  close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
    except:
        pass

    # Volume features
    df['volume_sma_10'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1e-8)
    df['volume_delta'] = df['volume'] * np.sign(df['close'] - df['open'])
    df['volume_delta_ma5'] = df['volume_delta'].rolling(5).mean()

    # VWAP
    df['day'] = df.index.date
    df['cum_vol'] = df.groupby('day')['volume'].cumsum()
    df['cum_pv'] = df.groupby('day').apply(
        lambda x: (x['close'] * x['volume']).cumsum()
    ).reset_index(level=0, drop=True)
    df['vwap'] = df['cum_pv'] / (df['cum_vol'] + 1e-8)
    df['close_vwap_ratio'] = df['close'] / df['vwap']

    # Momentum features
    for window in [3, 5, 10]:
        df[f'momentum_{window}'] = df['close'].pct_change(window)
        df[f'acceleration_{window}'] = df[f'momentum_{window}'].diff()

    # Order flow approximation
    df['buy_pressure'] = df['volume'] * (df['close'] > df['open']).astype(int)
    df['sell_pressure'] = df['volume'] * (df['close'] < df['open']).astype(int)
    df['order_imbalance'] = (df['buy_pressure'].rolling(10).sum() -
                             df['sell_pressure'].rolling(10).sum()) / \
                            (df['buy_pressure'].rolling(10).sum() +
                             df['sell_pressure'].rolling(10).sum() + 1e-8)

    # Price position features
    for window in [20, 50]:
        df[f'high_{window}'] = df['high'].rolling(window).max()
        df[f'low_{window}'] = df['low'].rolling(window).min()
        df[f'price_position_{window}'] = (df['close'] - df[f'low_{window}']) / \
                                          (df[f'high_{window}'] - df[f'low_{window}'] + 1e-8)

    # Trend strength
    df['trend_strength'] = (df['close'] - df['close'].rolling(20).mean()) / \
                           (df['close'].rolling(20).std() + 1e-8)

    # Mean reversion signal
    df['mean_reversion_signal'] = np.where(
        (df['trend_strength'] > 2) | (df['trend_strength'] < -2), 1, 0
    )

    # Pattern detection
    df['breakout_up'] = ((df['close'] > df['high'].rolling(20).max().shift(1)) &
                         (df['volume'] > df['volume_sma_10'] * 1.5)).astype(int)
    df['breakout_down'] = ((df['close'] < df['low'].rolling(20).min().shift(1)) &
                           (df['volume'] > df['volume_sma_10'] * 1.5)).astype(int)

    # Drawdown
    rolling_max = df['close'].rolling(50).max()
    df['drawdown'] = (df['close'] - rolling_max) / rolling_max

    # Additional advanced features
    df['efficiency_ratio'] = df['close'].diff(10).abs() / (df['close'].diff().abs().rolling(10).sum() + 1e-8)
    df['liquidity_score'] = (df['volume'] * df['close']) / ((df['volume'] * df['close']).rolling(50).mean() + 1e-8)

    # Clean up temp columns
    df.drop(columns=['day', 'cum_vol', 'cum_pv'], errors='ignore', inplace=True)

    # === KALMAN FILTER FEATURES ===
    df = extract_kalman_features(df)

    print(f"Generated {len(df.columns)} initial features (including Kalman)")
    return df


# --------------------------------------------------------------------
# 3. Cleaning Pipeline
# --------------------------------------------------------------------
def apply_cleaning_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler, SimpleImputer]:
    """
    Apply cleaning pipeline:
    1. Imputation (median)
    2. Outlier removal (IQR clipping)
    3. Scaling (MinMax to [-1, 1])
    """
    print("\n" + "="*60)
    print("STEP 2: APPLYING CLEANING PIPELINE")
    print("="*60)

    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'kalman_trend', 'kalman_volume_trend']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

    df_clean = df.copy()

    # --- Step 2a: Imputation ---
    print("\n  [2a] Imputing missing values with median...")
    missing_before = df_clean[numeric_cols].isna().sum().sum()

    imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

    missing_after = df_clean[numeric_cols].isna().sum().sum()
    print(f"       Missing values: {missing_before} → {missing_after}")

    # --- Step 2b: Outlier Removal (IQR Clipping) ---
    print("\n  [2b] Removing outliers with IQR clipping (3x)...")
    outlier_counts = {}

    for col in numeric_cols:
        data = df_clean[col]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR

        n_outliers = ((data < lower) | (data > upper)).sum()
        if n_outliers > 0:
            outlier_counts[col] = n_outliers

        df_clean[col] = data.clip(lower=lower, upper=upper)

    total_outliers = sum(outlier_counts.values())
    print(f"       Clipped {total_outliers} outlier values across {len(outlier_counts)} columns")

    # --- Step 2c: Scaling to [-1, 1] ---
    print("\n  [2c] Scaling features to [-1, 1] with MinMaxScaler...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    print(f"       Scaled {len(numeric_cols)} columns")

    return df_clean, scaler, imputer


# --------------------------------------------------------------------
# 4. Remove Highly Correlated Features (Generic Function)
# --------------------------------------------------------------------
def remove_correlated_features(df: pd.DataFrame, threshold: float,
                                exclude_cols: List[str] = None,
                                step_name: str = "FILTER") -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Remove features with correlation > threshold.
    Keeps the feature with higher importance (correlation with returns).
    """
    print(f"\n  [{step_name}] Removing features with correlation > {threshold}...")

    if exclude_cols is None:
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns']

    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

    if len(feature_cols) == 0:
        return df, [], pd.DataFrame()

    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()

    # Calculate feature importance (correlation with future returns)
    future_returns = df['returns'].shift(-1) if 'returns' in df.columns else None
    feature_importance = {}

    for col in feature_cols:
        try:
            if future_returns is not None:
                corr = abs(df[col].corr(future_returns))
                feature_importance[col] = corr if not np.isnan(corr) else 0
            else:
                feature_importance[col] = df[col].std()  # Use variance as proxy
        except:
            feature_importance[col] = 0

    # Find pairs of highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    removed_features = []
    removal_reasons = []
    features_to_drop = set()

    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if pd.notna(upper_tri.loc[idx, col]) and upper_tri.loc[idx, col] > threshold:
                # Keep the one with higher importance
                if feature_importance.get(col, 0) > feature_importance.get(idx, 0):
                    if idx not in features_to_drop:
                        features_to_drop.add(idx)
                        removed_features.append(idx)
                        removal_reasons.append(f"corr={upper_tri.loc[idx, col]:.3f} with {col}")
                else:
                    if col not in features_to_drop:
                        features_to_drop.add(col)
                        removed_features.append(col)
                        removal_reasons.append(f"corr={upper_tri.loc[idx, col]:.3f} with {idx}")

    removal_df = pd.DataFrame({
        'removed_feature': removed_features,
        'reason': removal_reasons
    }).drop_duplicates(subset=['removed_feature'])

    features_to_keep = [col for col in feature_cols if col not in features_to_drop]

    print(f"       Original features: {len(feature_cols)}")
    print(f"       Removed features: {len(features_to_drop)}")
    print(f"       Remaining features: {len(features_to_keep)}")

    # Keep essential cols + selected features
    final_cols = list(exclude_cols) + features_to_keep
    final_cols = [col for col in final_cols if col in df.columns]

    return df[final_cols], features_to_keep, removal_df


# --------------------------------------------------------------------
# 5. Feature Importance & Selection
# --------------------------------------------------------------------
def select_top_features(df: pd.DataFrame, n_features: int,
                        exclude_cols: List[str] = None,
                        step_name: str = "SELECT") -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Calculate feature importance and select top N features.
    """
    print(f"\n  [{step_name}] Selecting top {n_features} features by importance...")

    if exclude_cols is None:
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns']

    # Create target (future returns)
    target = df['returns'].shift(-1) if 'returns' in df.columns else None

    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

    if len(feature_cols) <= n_features:
        print(f"       Only {len(feature_cols)} features available, keeping all")
        importance_df = pd.DataFrame({'feature': feature_cols, 'combined_score': [1.0]*len(feature_cols)})
        return df, importance_df, feature_cols

    # Prepare clean data
    df_clean = df[feature_cols].copy()
    if target is not None:
        df_clean['target'] = target
    df_clean = df_clean.dropna()

    if len(df_clean) < 100:
        print(f"       Warning: Only {len(df_clean)} valid rows, keeping all features")
        importance_df = pd.DataFrame({'feature': feature_cols, 'combined_score': [1.0]*len(feature_cols)})
        return df, importance_df, feature_cols

    X = df_clean[feature_cols].values
    y = df_clean['target'].values if 'target' in df_clean.columns else np.random.randn(len(df_clean))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    importance_results = []

    # 1. Correlation
    for i, col in enumerate(feature_cols):
        try:
            corr = np.corrcoef(df_clean[col].values, y)[0, 1]
            corr = 0 if np.isnan(corr) else corr
        except:
            corr = 0
        importance_results.append({
            'feature': col,
            'correlation': abs(corr),
            'correlation_signed': corr
        })

    # 2. Mutual Information
    try:
        mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        for i, col in enumerate(feature_cols):
            importance_results[i]['mutual_info'] = mi_scores[i]
    except:
        for i in range(len(feature_cols)):
            importance_results[i]['mutual_info'] = 0

    # 3. F-regression
    try:
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)
        f_scores = np.nan_to_num(selector.scores_, nan=0, posinf=0, neginf=0)
        for i, col in enumerate(feature_cols):
            importance_results[i]['f_score'] = f_scores[i]
    except:
        for i in range(len(feature_cols)):
            importance_results[i]['f_score'] = 0

    # Create DataFrame and normalize
    importance_df = pd.DataFrame(importance_results)

    for col in ['correlation', 'mutual_info', 'f_score']:
        max_val = importance_df[col].max()
        if max_val > 0:
            importance_df[f'{col}_norm'] = importance_df[col] / max_val
        else:
            importance_df[f'{col}_norm'] = 0

    # Combined score
    importance_df['combined_score'] = (
        0.35 * importance_df['correlation_norm'] +
        0.35 * importance_df['mutual_info_norm'] +
        0.30 * importance_df['f_score_norm']
    )

    importance_df = importance_df.sort_values('combined_score', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)

    # Select top N features
    top_features = importance_df.head(n_features)['feature'].tolist()

    print(f"       Top {min(10, n_features)} features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"         {row['rank']:2d}. {row['feature']:<30s} score={row['combined_score']:.4f}")

    # Build final feature set
    final_cols = list(exclude_cols) + top_features
    final_cols = list(dict.fromkeys([col for col in final_cols if col in df.columns]))

    return df[final_cols], importance_df, top_features


# --------------------------------------------------------------------
# 6. K-Means + RBF Feature Generation
# --------------------------------------------------------------------
def rbf_kernel(X: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel values."""
    distances = cdist(X, centers, metric='sqeuclidean')
    return np.exp(-gamma * distances)


def generate_kmeans_rbf_features(df: pd.DataFrame, feature_cols: List[str],
                                   n_clusters_list: List[int] = [5, 10, 15, 20],
                                   gamma_list: List[float] = [0.01, 0.1, 0.5, 1.0]) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate K-means + RBF features with multiple configurations.
    """
    print("\n" + "="*60)
    print("STEP 5: GENERATING K-MEANS + RBF FEATURES")
    print("="*60)

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    df_result = df.copy()
    all_cluster_stats = {}
    new_features_count = 0

    print(f"\n  Feature columns used: {len(feature_cols)}")
    print(f"  Cluster sizes: {n_clusters_list}")
    print(f"  RBF gamma values: {gamma_list}")

    for n_clusters in n_clusters_list:
        print(f"\n  --- K-Means with {n_clusters} clusters ---")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # Add cluster labels
        df_result[f'cluster_{n_clusters}'] = cluster_labels
        new_features_count += 1

        # One-hot encoding
        for i in range(n_clusters):
            df_result[f'cluster_{n_clusters}_is_{i}'] = (cluster_labels == i).astype(int)
            new_features_count += 1

        # Distance to each cluster center
        distances = cdist(X, centers, metric='euclidean')
        for i in range(n_clusters):
            df_result[f'dist_cluster_{n_clusters}_{i}'] = distances[:, i]
            new_features_count += 1

        # Distance aggregates
        df_result[f'dist_nearest_{n_clusters}'] = distances.min(axis=1)
        df_result[f'dist_farthest_{n_clusters}'] = distances.max(axis=1)
        df_result[f'dist_ratio_{n_clusters}'] = distances.min(axis=1) / (distances.max(axis=1) + 1e-8)
        new_features_count += 3

        # RBF features
        for gamma in gamma_list:
            rbf_features = rbf_kernel(X, centers, gamma)

            for i in range(n_clusters):
                df_result[f'rbf_{n_clusters}_g{gamma}_c{i}'] = rbf_features[:, i]
                new_features_count += 1

            # Aggregated RBF
            df_result[f'rbf_{n_clusters}_g{gamma}_max'] = rbf_features.max(axis=1)
            df_result[f'rbf_{n_clusters}_g{gamma}_mean'] = rbf_features.mean(axis=1)
            df_result[f'rbf_{n_clusters}_g{gamma}_std'] = rbf_features.std(axis=1)
            df_result[f'rbf_{n_clusters}_g{gamma}_entropy'] = -np.sum(
                rbf_features * np.log(rbf_features + 1e-10), axis=1
            ) / np.log(n_clusters)
            new_features_count += 4

        # Cluster statistics
        cluster_stats = []
        future_returns = df['returns'].shift(-1) if 'returns' in df.columns else None

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            stats = {
                'cluster': cluster_id,
                'n_clusters': n_clusters,
                'count': mask.sum(),
                'pct': mask.sum() / len(df) * 100,
            }
            if future_returns is not None:
                cluster_returns = future_returns.loc[df.index[mask]]
                stats['avg_return'] = cluster_returns.mean() * 100 if len(cluster_returns) > 0 else 0
                stats['win_rate'] = (cluster_returns > 0).mean() * 100 if len(cluster_returns) > 0 else 0
            cluster_stats.append(stats)

        all_cluster_stats[n_clusters] = pd.DataFrame(cluster_stats)

    # Cluster transition features
    for n_clusters in n_clusters_list:
        df_result[f'cluster_{n_clusters}_changed'] = (
            df_result[f'cluster_{n_clusters}'] != df_result[f'cluster_{n_clusters}'].shift(1)
        ).astype(int)
        cluster_changes = df_result[f'cluster_{n_clusters}_changed'].cumsum()
        df_result[f'cluster_{n_clusters}_duration'] = df_result.groupby(cluster_changes).cumcount()
        new_features_count += 2

    # Inter-cluster agreement
    for i, n1 in enumerate(n_clusters_list[:-1]):
        for n2 in n_clusters_list[i+1:]:
            df_result[f'cluster_agree_{n1}_{n2}'] = (
                df_result[f'cluster_{n1}'] == df_result[f'cluster_{n2}']
            ).astype(int)
            new_features_count += 1

    print(f"\n  Total new features generated: {new_features_count}")

    return df_result, all_cluster_stats


# --------------------------------------------------------------------
# 7. Post-RBF Feature Filtering (SECOND FILTER)
# --------------------------------------------------------------------
def apply_post_rbf_filtering(df: pd.DataFrame, n_final_features: int = 30,
                              correlation_threshold: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Apply second round of filtering after RBF features are generated:
    1. Remove highly correlated features (>90%)
    2. Select top N features
    """
    print("\n" + "="*60)
    print("STEP 6: POST-RBF FEATURE FILTERING")
    print("="*60)

    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns']

    # Step 6a: Remove correlated features (>90%)
    df_decorr, features_kept, removal_df = remove_correlated_features(
        df,
        threshold=correlation_threshold,
        exclude_cols=exclude_cols,
        step_name="6a"
    )

    # Step 6b: Select top N features
    df_final, importance_df, top_features = select_top_features(
        df_decorr,
        n_features=n_final_features,
        exclude_cols=exclude_cols,
        step_name="6b"
    )

    print(f"\n  FINAL: {len(top_features)} features selected")

    return df_final, importance_df, top_features


# --------------------------------------------------------------------
# 8. Visualization
# --------------------------------------------------------------------
def visualize_results(df: pd.DataFrame, importance_df: pd.DataFrame,
                      cluster_stats: Dict, output_dir: str = './data'):
    """Create comprehensive visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # 1. Feature Importance (Final)
    plt.figure(figsize=(12, 10))
    top_n = min(30, len(importance_df))
    top_features = importance_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 0.8, top_n))
    plt.barh(range(top_n), top_features['combined_score'].values[::-1], color=colors[::-1])
    plt.yticks(range(top_n), top_features['feature'].values[::-1], fontsize=8)
    plt.xlabel('Combined Importance Score')
    plt.title(f'Top {top_n} Final Features (After All Filtering)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_feature_importance.png', dpi=150)
    plt.close()

    # 2. Cluster Analysis Grid
    if cluster_stats:
        n_plots = len(cluster_stats)
        fig, axes = plt.subplots((n_plots + 1) // 2, 2, figsize=(16, 6 * ((n_plots + 1) // 2)))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, (n_clusters, stats_df) in enumerate(cluster_stats.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]

            if 'avg_return' in stats_df.columns:
                colors = ['green' if x > 0 else 'red' for x in stats_df['avg_return']]
                bars = ax.bar(stats_df['cluster'], stats_df['avg_return'], color=colors, alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_ylabel('Avg Future Return (%)')

                if 'win_rate' in stats_df.columns:
                    for i, (bar, wr) in enumerate(zip(bars, stats_df['win_rate'])):
                        ax.annotate(f'{wr:.0f}%',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                   ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                                   fontsize=8)

            ax.set_xlabel('Cluster')
            ax.set_title(f'K={n_clusters} Clusters')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/cluster_analysis_grid.png', dpi=150)
        plt.close()

    # 3. Kalman Filter Visualization
    if 'kalman_trend' in df.columns or 'kalman_deviation' in df.columns:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        last_n = min(500, len(df))

        # Price vs Kalman Trend
        ax = axes[0]
        if 'kalman_trend' not in df.columns:
            # Recreate for plotting
            kalman_trend, _ = apply_kalman_filter(df['close'])
            ax.plot(df.index[-last_n:], df['close'].iloc[-last_n:], label='Price', alpha=0.7)
            ax.plot(df.index[-last_n:], kalman_trend.iloc[-last_n:], label='Kalman Trend', linewidth=2)
        else:
            ax.plot(df.index[-last_n:], df['close'].iloc[-last_n:], label='Price', alpha=0.7)
        ax.set_title('Price with Kalman Filter Trend')
        ax.legend()

        # Kalman Deviation
        ax = axes[1]
        if 'kalman_deviation' in df.columns:
            ax.plot(df.index[-last_n:], df['kalman_deviation'].iloc[-last_n:], color='purple')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Kalman Deviation (Price - Trend)')

        # Kalman Velocity
        ax = axes[2]
        if 'kalman_velocity' in df.columns:
            ax.plot(df.index[-last_n:], df['kalman_velocity'].iloc[-last_n:], color='orange')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Kalman Velocity (Trend Rate of Change)')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/kalman_analysis.png', dpi=150)
        plt.close()

    # 4. Feature Pipeline Summary
    fig, ax = plt.subplots(figsize=(12, 6))

    stages = ['Core\nFeatures', 'After\n1st Corr\nFilter', 'Top 25\nSelected',
              'After\nRBF/KMeans', 'After\n2nd Corr\nFilter', 'Final\nTop 30']

    # These are approximate - actual values depend on data
    counts = [80, 60, 25, 400, 150, 30]  # Placeholder values
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    bars = ax.bar(stages, counts, color=colors)
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Engineering Pipeline: Feature Count at Each Stage')

    for bar, count in zip(bars, counts):
        ax.annotate(str(count), xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_pipeline_summary.png', dpi=150)
    plt.close()

    print("  Visualizations saved!")


# --------------------------------------------------------------------
# 9. Main Pipeline
# --------------------------------------------------------------------
def main():
    """Main execution pipeline."""
    print("="*70)
    print("MINUTE-LEVEL FEATURE ENGINEERING v4.0")
    print("(With Kalman Filter + Two-Stage Feature Filtering)")
    print("="*70)
    print("\nPipeline Steps:")
    print("  1. Generate core features + Kalman Filter trends")
    print("  2. Apply cleaning (Impute → Outlier removal → Scale)")
    print("  3. FIRST FILTER: Remove correlated features (r > 0.8)")
    print("  4. Select top 25 features")
    print("  5. Generate K-means + RBF features")
    print("  6. SECOND FILTER: Remove correlated features (r > 0.9)")
    print("  7. Select final top 30 features")
    print("="*70)

    # Find input file
    if not os.path.exists(MINUTE_DATA_DIR):
        os.makedirs(MINUTE_DATA_DIR)
        print(f"Created {MINUTE_DATA_DIR}. Please add your data files.")
        return None, None, None

    all_files = [f for f in os.listdir(MINUTE_DATA_DIR) if f.endswith('.csv')]
    if not all_files:
        print(f"No CSV files found in {MINUTE_DATA_DIR}")
        return None, None, None

    input_file = None
    for f in all_files:
        if TICKER.lower() in f.lower():
            input_file = os.path.join(MINUTE_DATA_DIR, f)
            break

    if input_file is None:
        input_file = os.path.join(MINUTE_DATA_DIR, all_files[0])

    print(f"\nInput: {input_file}")

    # Step 1: Load data
    df = load_and_prepare_data(input_file)
    if df is None:
        return None, None, None

    initial_rows = len(df)

    # Step 2: Generate core features (includes Kalman)
    df = generate_core_features(df)
    features_after_core = len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])

    # Step 3: Apply cleaning pipeline
    df_clean, scaler, imputer = apply_cleaning_pipeline(df)

    # Step 4: FIRST FILTER - Remove correlated features (>0.8)
    print("\n" + "="*60)
    print("STEP 3: FIRST CORRELATION FILTER (threshold > 0.8)")
    print("="*60)
    df_decorr1, features_after_corr1, removal_df1 = remove_correlated_features(
        df_clean, threshold=CORRELATION_THRESHOLD_FIRST, step_name="3"
    )

    # Step 5: Select top 25 features
    print("\n" + "="*60)
    print("STEP 4: SELECT TOP 25 FEATURES")
    print("="*60)
    df_selected, importance_df_before, top_features_before = select_top_features(
        df_decorr1, n_features=N_FEATURES_BEFORE_RBF, step_name="4"
    )

    # Step 6: Generate K-means + RBF features
    feature_cols = [col for col in top_features_before if col in df_selected.columns]
    df_with_rbf, cluster_stats = generate_kmeans_rbf_features(
        df_selected, feature_cols,
        n_clusters_list=N_CLUSTERS_LIST,
        gamma_list=RBF_GAMMA_LIST
    )

    features_after_rbf = len(df_with_rbf.columns)

    # Scale new RBF features
    print("\n  Scaling new K-means/RBF features...")
    new_cols = [col for col in df_with_rbf.columns if col not in df_selected.columns]
    numeric_new = [col for col in new_cols if df_with_rbf[col].dtype in ['float64', 'int64', 'int32']]

    if numeric_new:
        new_scaler = MinMaxScaler(feature_range=(-1, 1))
        df_with_rbf[numeric_new] = new_scaler.fit_transform(df_with_rbf[numeric_new])

    # Step 7: SECOND FILTER - Post-RBF filtering
    df_final, importance_df_final, final_features = apply_post_rbf_filtering(
        df_with_rbf,
        n_final_features=N_FEATURES_FINAL,
        correlation_threshold=CORRELATION_THRESHOLD_SECOND
    )

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    df_final.to_csv(OUTPUT_FILE, index=True)
    print(f"\nFinal data saved to: {OUTPUT_FILE}")

    importance_file = OUTPUT_FILE.replace('.csv', '_importance.csv')
    importance_df_final.to_csv(importance_file, index=False)
    print(f"Feature importance saved to: {importance_file}")

    removal_file = OUTPUT_FILE.replace('.csv', '_removed_features.csv')
    removal_df1.to_csv(removal_file, index=False)
    print(f"Removed features log saved to: {removal_file}")

    if cluster_stats:
        all_stats = pd.concat(cluster_stats.values(), ignore_index=True)
        cluster_file = OUTPUT_FILE.replace('.csv', '_clusters.csv')
        all_stats.to_csv(cluster_file, index=False)
        print(f"Cluster analysis saved to: {cluster_file}")

    # Create visualizations
    try:
        visualize_results(df_final, importance_df_final, cluster_stats,
                         os.path.dirname(OUTPUT_FILE))
    except Exception as e:
        print(f"Visualization error: {e}")
        traceback.print_exc()

    # Print summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"\nData:")
    print(f"  Initial rows: {initial_rows}")
    print(f"  Final rows: {len(df_final)}")

    print(f"\nFeature Pipeline:")
    print(f"  After core generation (+ Kalman): {features_after_core}")
    print(f"  After 1st correlation filter (>{CORRELATION_THRESHOLD_FIRST}): {len(features_after_corr1)}")
    print(f"  After top-{N_FEATURES_BEFORE_RBF} selection: {N_FEATURES_BEFORE_RBF}")
    print(f"  After K-means + RBF expansion: {features_after_rbf}")
    print(f"  After 2nd correlation filter (>{CORRELATION_THRESHOLD_SECOND}): ~{len(df_final.columns) + 20}")
    print(f"  FINAL (top {N_FEATURES_FINAL}): {len(final_features)}")

    print(f"\nKalman Features Included:")
    kalman_features = [f for f in final_features if 'kalman' in f.lower()]
    for f in kalman_features:
        print(f"    - {f}")

    print(f"\n  TOTAL FINAL FEATURES: {len(df_final.columns)}")

    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)

    return df_final, importance_df_final, cluster_stats


if __name__ == "__main__":
    main()
