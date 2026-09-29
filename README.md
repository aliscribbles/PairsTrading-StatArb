# PairsTrading-StatArb: Quantitative Statistical Arbitrage Engine

An end-to-end pairs trading and statistical arbitrage backtesting engine built in Python. This repository implements a complete quantitative pipeline: from automated universe screening and cointegration testing to walk-forward validation, state-machine signal generation, and dynamic Kelly Criterion position sizing.

## System Architecture

The engine is highly modular, allowing for independent testing of signal generators, risk overlays, and execution logic. 

*   **`data.py`**: Automated ingestion and preprocessing of adjusted daily closing prices.
*   **`cointegration.py`**: Universe screening using the Engle-Granger two-step method, expanding-window OLS regressions, and rolling Z-score normalisations.
*   **`signals.py`**: Core state-machine execution logic handling threshold crossovers (\(\pm1.75\sigma\) entry, \(0.5\sigma\) exit), including implementations for both Static OLS and 2D Kalman Filter state-space models.
*   **`backtest.py`**: Event-driven vectorised backtesting engine with granular transaction cost modelling, turnover tracking, and comprehensive tear-sheet performance metrics (Sharpe, Sortino, Max Drawdown).
*   **`kelly.py`**: Dynamic position sizing overlay calculating rolling win rates and risk/reward ratios to scale capital allocation continuously.

## The Research Journey & Key Findings

This project was developed iteratively, documenting a rigorous empirical evolution from a naive statistical arbitrage model to a fully scaled, risk-managed quantitative portfolio. The research ultimately exposed a critical paradox in mean-reversion trading: the hidden dangers of over-engineering adaptive models.

### Phase 1: Foundation & The Dynamic Beta Promise (`scripts/01` – `03`)
The initial pipeline was built to compare a standard Ordinary Least Squares (OLS) cointegration baseline against an adaptive state-space model. 
*   **The Early Win:** Early testing (`03_kalman_vs_static.py`) demonstrated that a 1D Kalman Filter—dynamically tracking the hedge ratio (\(\beta\)) successfully adapted to market drift and outperformed the naive, unadjusted Static OLS baseline. This early success validated the premise that dynamic models could isolate spreads more effectively over time.

### Phase 2: Validation & Risk Infrastructure (`scripts/04` – `07`)
To translate theoretical edge into a production-ready system, the architecture was systematically fortified against curve-fitting and structural decay:
*   **Walk-Forward Validation:** Engineered strict out-of-sample testing loops (504-day training / 252-day testing folds) to completely eliminate look-ahead bias.
*   **Macro Regime Detection:** Built a rolling Augmented Dickey-Fuller (ADF) state machine. When a pair's stationarity breaks (p-value > 0.10), the strategy acts as a kill-switch, flattening positions to prevent trending drawdowns.
*   **Parameter Optimisation:** Conducted sensitivity sweeps across entry/exit thresholds and lookback windows, explicitly targeting performance plateaus (settling on \(\pm1.75\sigma\) entry, \(0.5\sigma\) exit, 60-day window) rather than isolated, over-fit spikes.

### Phase 3: The Architecture Showdown (`scripts/08` & `09`)
The final phase tested two diverging philosophies for the production portfolio: scaling via mathematical capital allocation versus scaling via hyper-adaptive signal generation.

**1. The Production Champion: Regime-Aware OLS + Kelly Sizing (`08_kelly_dashboard.py`)**
Applying a 30%-capped Fractional Kelly sizing matrix to the regime-filtered Static OLS baseline produced exceptional out-of-sample performance. The Kelly overlay optimised capital efficiency by mathematically starving deteriorating relationships (allocating 0% to META-GS) and heavily weighting high-conviction cointegrated pairs (AAPL-MSFT, KO-PEP). 
*   **Result:** The Kelly-sized portfolio generated a **25.70%** cumulative return, nearly doubling the 13.20% return of the equal-weight baseline.

**2. The Adaptive Trap: Advanced 2D Kalman Hybrid (`09_advanced_hybrid_signals.py`)**
Attempting to build upon the Phase 1 success, the engine was upgraded to a full 2D Kalman Filter dynamically tracking both \(\alpha\) and \(\beta\), coupled with the ADF regime filter and a tightly tuned process noise covariance (\(\delta = 10^{-6}\)). 
*   **Result:** The model completely collapsed, yielding a **-14.08%** cumulative loss and a **-16.60%** maximum drawdown[cite: 4]. 
*   **The Post-Mortem:** By dynamically adjusting the intercept (\(\alpha\)) alongside the slope (\(\beta\)), the 2D filter became too smart. As an optimal state estimator, it perfectly absorbed price shocks, moving the baseline to minimize residual errors. It effectively "ate the spread," leaving no divergence stretch to trade. The engine was starved of valid entry signals and forced to trade on microscopic noise.

### Final Conclusion
The empirical evidence from Project Vectis proves that in daily equity statistical arbitrage, signal adaptivity is a double-edged sword. While a 1D dynamic beta showed early promise, fully adaptive state-space models mathematically destroy the exact structural mispricings required for mean-reversion. 

A rigidly anchored baseline—which allows spreads the breathing room to stretch and revert—paired with advanced, dynamic risk management (Regime Filtering + Kelly Sizing) is the vastly superior quantitative architecture.

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/PairsTrading-StatArb.git](https://github.com/yourusername/PairsTrading-StatArb.git)
   cd PairsTrading-StatArb

2. **Install dependencies:**
   ```bash
   pip install pandas numpy statsmodels matplotlib seaborn


3. **Run the Production Dashboard**
   ```bash
   python scripts/main.py


## Repository Structure

```text

PairsTrading-StatArb/
│
├── src/                 # Core engine modules
│    ├── stat_arb/
│       ├── __init__.py
│       ├── data.py               # Data engineering
│       ├── cointegration.py      # Math & stat-tests
│       ├── signals.py            # Signal generation & state machines
│       ├── backtest.py           # P&L and tear-sheet metrics
│       └── plotting.py           # Plotting
│
├── scripts/                  # Research checkpoints & execution files
│   ├── 01_pairs_selection.py
│   ├── 02_run_strategy.py
│   ├── 03_kalman_vs_static.py
│   ├── 04_walk_forward.py
│   ├── 05_regime_detection.py
│   ├── 06_parameter_sweep.py
│   ├── 07_portfolio_backtest.py
│   ├── 08_kelly_dashboard.py
│   ├── 09_advanced_hybrid_signals.py
│   └── main.py               # Final production Kelly pipeline
├── src/ 
│   └── stat_arb/
├── results/                  # Generated equity curves and heatmaps
└── README.md
   
