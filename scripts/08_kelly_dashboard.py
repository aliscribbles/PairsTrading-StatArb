import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stat_arb.data import load_and_prep_data
from stat_arb.cointegration import screen_pairs, engle_granger_test, zscore_normalise
from stat_arb.signals import zscore_signals
from stat_arb.backtest import run_backtest


def calculate_kelly(returns_series):
    """Calculates Full and Half Kelly fractions from a return series."""
    # Assuming 252 trading days for annualized metrics, but we calculate based on daily variance
    mean_ret = returns_series.mean()
    var_ret = returns_series.var()

    if var_ret == 0 or mean_ret <= 0:
        return 0.0, 0.0

    full_kelly = mean_ret / var_ret
    half_kelly = full_kelly / 2.0
    return full_kelly, half_kelly


def main():
    print("Loading data & screening universe...")
    prices, returns, log_prices = load_and_prep_data()
    pairs_df = screen_pairs(log_prices, corr_threshold=0.85).head(5)

    OPTIMISED_WINDOW = 60
    OPTIMISED_ENTRY = 1.75
    OPTIMISED_EXIT = 0.5

    portfolio_daily_returns = pd.DataFrame(index=prices.index)
    kelly_weights = {}

    print("\nExecuting Pipeline for Checkpoint 11 Dashboard...")

    for index, row in pairs_df.iterrows():
        s1, s2 = row["stock1"], row["stock2"]

        # 1. Math & Signals
        eg = engle_granger_test(log_prices[s1], log_prices[s2])
        normalised_spread = zscore_normalise(eg["spread"], window=OPTIMISED_WINDOW)
        zscore_signals_df = zscore_signals(
            normalised_spread, entry=OPTIMISED_ENTRY, exit_=OPTIMISED_EXIT
        )

        # 2. Backtest
        backtest = run_backtest(
            prices,
            s1,
            s2,
            signals=zscore_signals_df,
            hedge_ratio=eg["hedge_ratio"],
            mode="static",
        )
        daily_ret = backtest["daily_return"].fillna(0)

        # 3. Kelly Calculation
        full_k, half_k = calculate_kelly(daily_ret)
        # Cap max allocation to 30% per pair to prevent absurd leverage
        safe_weight = min(half_k, 0.30)
        kelly_weights[f"{s1}_{s2}"] = safe_weight

        portfolio_daily_returns[f"{s1}_{s2}"] = daily_ret

    portfolio_daily_returns.dropna(inplace=True)

    # Normalize weights so they sum to 1.0 (100% of capital)
    total_weight = sum(kelly_weights.values())
    if total_weight > 0:
        normalized_weights = {k: v / total_weight for k, v in kelly_weights.items()}
    else:
        normalized_weights = {
            k: 0.20 for k in kelly_weights.keys()
        }  # Fallback to equal weight if zero edge

    # --- AGGREGATION: EQUAL VS KELLY ---
    portfolio_daily_returns["Equal_Return"] = portfolio_daily_returns[
        [c for c in portfolio_daily_returns.columns if "_" in c]
    ].mean(axis=1)

    portfolio_daily_returns["Kelly_Return"] = 0.0
    for col in normalized_weights.keys():
        portfolio_daily_returns["Kelly_Return"] += (
            portfolio_daily_returns[col] * normalized_weights[col]
        )

    # Cumulative Returns
    equal_cum = (1 + portfolio_daily_returns["Equal_Return"]).cumprod() - 1
    kelly_cum = (1 + portfolio_daily_returns["Kelly_Return"]).cumprod() - 1

    # Print Final Dashboard Output
    print("-" * 40)
    print("CHECKPOINT 11: FINAL SIZING DASHBOARD")
    print("-" * 40)
    print("Optimal Target Allocations (Normalized Half-Kelly):")
    for pair, weight in normalized_weights.items():
        print(f"  {pair}: {weight:.1%}")

    print("\nPERFORMANCE COMPARISON:")
    print(f"Equal Weight Total Return: {equal_cum.iloc[-1]:.2%}")
    print(f"Kelly Weight Total Return: {kelly_cum.iloc[-1]:.2%}")
    print("-" * 40)

    # Plotting the Final Dashboard
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equal_cum.index, equal_cum, color="grey", alpha=0.7, label="Equal Weight")
    ax.plot(
        kelly_cum.index, kelly_cum, color="green", linewidth=2, label="Kelly Weight"
    )
    ax.set_title(
        "Final Dashboard: Kelly Sizing vs Equal Weight", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(BASE_DIR, "..", "results", "08_final_dashboard.png")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFinal Dashboard saved to: {save_path}")


if __name__ == "__main__":
    main()
