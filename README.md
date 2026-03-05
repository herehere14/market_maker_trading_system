# market_maker_trading_system

please be cautious! this is a simplified version and just for testing, the real version is much more complicated and involving sensitive data. But this is great for testing and simulation.

# market_maker_trading_system

A minute-level research pipeline for **market/medium-frequency trading**:  
1) fetch 1-minute OHLCV data (Polygon),  
2) engineer features (technical indicators + Kalman trend + filtering + KMeans/RBF expansion),  
3) train/evaluate an ML model with **realistic transaction cost simulation** (IBKR commissions + slippage) and “sniper mode” confidence thresholding.

> ⚠️ This repository is currently best viewed as a **research/backtesting pipeline**, not a production trading bot.

---

## What’s inside

### 1) `data_fetch.py` — Fetch 1-minute bars from Polygon
- Pulls 1-minute aggregates from Polygon’s API.
- Saves/updates a CSV to: `./data/minute_data/{TICKER}.US_minute.csv`
- If the file already exists, it appends only new rows after the last timestamp. :contentReference[oaicite:1]{index=1}

### 2) `data_engi.py` — Feature engineering pipeline (v4)
Produces an engineered dataset from minute OHLCV with:

- Core return/range features
- Technical indicators (via `ta`)
- **Kalman Filter trend extraction**
- Cleaning pipeline: impute → outlier clipping → scaling
- Two-stage correlation filtering (pre/post RBF)
- Feature selection (top-k)
- **KMeans + RBF feature expansion**
- Outputs to: `./data/{TICKER}.US_minute_engineered_v4.csv` :contentReference[oaicite:2]{index=2}

### 3) `Market_maker_ML.py` — Model training + evaluation + net-of-costs backtest
A minute-level model training & evaluation pipeline with:
- “**Sniper mode**”: only trade when predicted probability exceeds a confidence threshold
- Anti-overfitting regularization focus
- Net returns simulated **after** estimated IBKR commissions + slippage
- Saves plots/results into an output directory :contentReference[oaicite:3]{index=3}

### 4) `medium_frequency_trading_system.ipynb`
Notebook version / experiments (large). :contentReference[oaicite:4]{index=4}

---

## Repo structure
