import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from stat_arb.data import load_and_prep_data
from stat_arb.cointegration import zscore_normalise
from stat_arb.signals import zscore_signals, kalman_hedge_ratio
from stat_arb.backtest import run_backtest
from stat_arb.cointegration import engle_granger_test


def rolling_adf_pvalue(spread: pd.Series, window: int = 120) -> pd.Series:
    """Runs a rolling Augmented Dickey-Fuller test and returns p-values."""
    p_values = pd.Series(index=spread.index, dtype=float)

    # Start loop after the initial window
    for i in range(window, len(spread)):
        window_data = spread.iloc[i - window : i]
        try:
            # We only need the p-value (index 1 of adfuller output)
            stat, p_val, _, _, _, _ = adfuller(window_data, maxlag=1, regression="c")
            p_values.iloc[i] = p_val
        except:
            p_values.iloc[i] = np.nan

    return p_values


def main():
    print("Loading data...")
    prices, returns, log_prices = load_and_prep_data()

    s1, s2 = "AAPL", "MSFT"
    print(f"Executing Advanced Hybrid Signal on {s1} - {s2}...")

    # 1. THE PRIOR (Run Static OLS first to avoid the zero burn-in)
    eg_res = engle_granger_test(log_prices[s1], log_prices[s2])
    static_beta = eg_res["hedge_ratio"]

    # 2. THE 2D KALMAN FILTER (Inject the prior and tighten the delta)
    kalman_res = kalman_hedge_ratio(
        log_prices[s1], log_prices[s2], initial_beta=static_beta, delta=1e-6
    )

    # 3. Extract and shift alpha and beta to prevent look-ahead bias
    kalman_alpha = kalman_res["alpha"].shift(1)
    kalman_beta = kalman_res["beta"].shift(1)

    # 4. THE MICRO EXECUTION (Dynamic Kalman)
    dynamic_spread = (
        log_prices[s1] - (kalman_alpha + kalman_beta * log_prices[s2])
    ).dropna()

    # 4. THE MACRO REGIME (Rolling ADF on the Kalman Spread)
    print("Calculating rolling ADF p-values on the DYNAMIC baseline...")
    adf_pvalues = rolling_adf_pvalue(dynamic_spread, window=120)
    regime_mask = adf_pvalues <= 0.10

    # 5. THE HYBRID MERGE (Proper State Machine)
    normalised_spread = zscore_normalise(dynamic_spread, window=60)

    # We must iterate sequentially so the engine is forced to wait
    # for a fresh entry crossover after a regime blackout.
    positions = pd.Series(0, index=normalised_spread.index)
    current_pos = 0

    for date, z in normalised_spread.items():
        # If regime is broken (False or NaN), flatten and reset state
        if not regime_mask.get(date, False):
            current_pos = 0
        else:
            # Normal Z-score state machine logic
            if current_pos == 0:
                if z > 1.75:
                    current_pos = -1
                elif z < -1.75:
                    current_pos = 1
            elif current_pos == 1 and z > -0.5:
                current_pos = 0
            elif current_pos == -1 and z < 0.5:
                current_pos = 0

        positions[date] = current_pos

    hybrid_signals = pd.DataFrame({"signal": positions})

    # 6. Execute Backtest
    backtest_res = run_backtest(
        prices,
        s1,
        s2,
        signals=hybrid_signals,
        hedge_ratios=kalman_beta.dropna(),
        mode="dynamic",
    )

    daily_ret = backtest_res["daily_return"].fillna(0)
    cum_ret = (1 + daily_ret).cumprod() - 1

    rolling_max = (1 + cum_ret).cummax()
    drawdown = (1 + cum_ret) / rolling_max - 1
    max_dd = drawdown.min()

    print("-" * 30)
    print("HYBRID ENGINE PERFORMANCE")
    print("-" * 30)
    print(f"Total Cumulative Return: {cum_ret.iloc[-1]:.2%}")
    print(f"Maximum Drawdown:      {max_dd:.2%}")
    print("-" * 30)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        cum_ret.index,
        cum_ret,
        color="blue",
        linewidth=1.5,
        label="Kalman + Regime Hybrid",
    )
    ax.set_title(
        f"{s1}-{s2} Hybrid Engine Equity Curve", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    ax.legend()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(BASE_DIR, "..", "results", "09_hybrid_signals.png")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Equity curve saved to: {save_path}")


if __name__ == "__main__":
    main()
