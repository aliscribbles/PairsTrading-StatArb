import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stat_arb.cointegration import (
    screen_pairs,
    engle_granger_test,
    zscore_normalise,
    expanding_ols_beta,
)
from stat_arb.backtest import run_backtest
from stat_arb.signals import zscore_signals, kalman_hedge_ratio

# import data
from stat_arb.data import load_and_prep_data


def main():

    # 1. Load Data
    prices, returns, log_prices = load_and_prep_data()

    # 2. Select Pair (Screen the universe and take the top result)
    pairs_df = screen_pairs(log_prices, corr_threshold=0.85)

    row = pairs_df[
        (pairs_df["stock1"] == "AAPL") & (pairs_df["stock2"] == "MSFT")
    ].iloc[0]
    s1 = row["stock1"]
    s2 = row["stock2"]
    print(f"Executing Strategy for Pair: {s1} - {s2} using static Beta")

    # 3. Get Hedge Ratio
    hedge_ratio = row["hedge_ratio"]

    # 4. Construct Spread
    eg = engle_granger_test(log_prices[s1], log_prices[s2])
    static_spread = eg["spread"]

    # 5. Normalize to Z-Score
    static_normalise = zscore_normalise(static_spread)

    # 6. Generate Signals
    static_signals = zscore_signals(static_normalise)

    # 7. Run static and dynamic Backtests and save results
    static_res = run_backtest(
        prices, s1, s2, static_signals, hedge_ratio=hedge_ratio, mode="static"
    )

    # --- The Dynamic Pipeline ---
    print(f"Executing Strategy for Pair: {s1} - {s2} using 2D Kalman Filter")

    # A. Get the dynamic states (returns a DataFrame of alpha and beta)
    kalman_states = kalman_hedge_ratio(log_prices[s1], log_prices[s2], delta=1e-5)

    # Walk-forward OLS: the honest static baseline (Kalman with delta = 0)
    wf = expanding_ols_beta(log_prices[s1], log_prices[s2], min_periods=252)
    wf = expanding_ols_beta(log_prices[s1], log_prices[s2], min_periods=252)
    print(f"log_prices length: {len(log_prices)}")
    print(f"wf length: {len(wf)}, non-NaN betas: {wf['beta'].notna().sum()}")
    print(
        f"first valid: {wf['beta'].first_valid_index()}, last: {wf['beta'].last_valid_index()}"
    )

    print(wf["beta"].tail())
    print(
        f"final expanding beta: {wf['beta'].iloc[-1]:.6f} "
        f"vs full-sample {hedge_ratio:.6f}"
    )

    # --- Burn-in diagnostic: inspect BEFORE building signals ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(
        kalman_states["beta"], color="tab:blue", lw=1, label="Kalman beta (diffuse)"
    )
    axes[0].plot(
        wf["beta"], color="green", lw=2.5, alpha=0.6, label="Expanding-window OLS"
    )
    axes[0].axhline(
        hedge_ratio, color="red", ls="--", lw=1.2, label="Full-sample OLS beta"
    )
    axes[0].set_ylim(0.55, 1.05)  # crop the diffuse spike to zero
    axes[0].set_title("Hedge ratio estimators")
    axes[0].legend(loc="lower right")

    axes[1].semilogy(kalman_states["P_beta"], color="tab:blue")
    axes[1].set_title("P_beta (log scale)")

    plt.tight_layout()
    plt.show()

    # Extract the beta column to pass to the backtester
    kalman_beta_series = kalman_states["beta"]

    # B. Construct the dynamic spread tracking BOTH alpha and beta
    # Spread = y - (alpha + beta * x) -> Shifted by 1 to prevent look-ahead!
    dynamic_spread = (
        log_prices[s1]
        - (
            kalman_states["alpha"].shift(1)
            + kalman_states["beta"].shift(1) * log_prices[s2]
        )
    ).dropna()

    dynamic_normalise = zscore_normalise(dynamic_spread)
    dynamic_signals = zscore_signals(dynamic_normalise)

    # C. Run Dynamic Backtest
    dynamic_res = run_backtest(
        prices,
        s1,
        s2,
        dynamic_signals,
        hedge_ratios=kalman_beta_series.shift(1).dropna(),
        mode="dynamic",
    )

    # print(f"Executing Strategy for Pair: {s1} - {s2} using Kalman Filter dynamic Beta")

    # # A. Get the dynamic beta series
    # kalman_beta_series = kalman_hedge_ratio(log_prices[s1], log_prices[s2],
    #                                         initial_beta=hedge_ratio, delta=1e-5
    #                                         )

    # # Construct the dynamic spread
    # dynamic_spread = (
    #     log_prices[s1] - kalman_beta_series.shift(1) * log_prices[s2]
    # ).dropna()
    # dynamic_normalise = zscore_normalise(dynamic_spread)
    # dynamic_signals = zscore_signals(dynamic_normalise)
    # dynamic_res = run_backtest(
    #     prices,
    #     s1,
    #     s2,
    #     dynamic_signals,
    #     hedge_ratios=kalman_beta_series.shift(1),
    #     mode="dynamic",
    # )

    # 8. Print Performance Metrics
    # static
    static_final_pnl = static_res["cumulative_return"].iloc[-1]
    static_max_dd = static_res["drawdown"].min()

    # dynamic
    dynamic_final_pnl = dynamic_res["cumulative_return"].iloc[-1]
    dynamic_max_dd = dynamic_res["drawdown"].min()

    print("-" * 30)
    print("STRATEGY PERFORMANCE")
    print("-" * 30)
    print("Static Result \n")

    print(f"Total Cumulative Return: {static_final_pnl:.2%}")
    print(f"Maximum Drawdown:      {static_max_dd:.2%}\n")

    print("Dynamic Result \n")
    print(f"Total Cumulative Return: {dynamic_final_pnl:.2%}")
    print(f"Maximum Drawdown:      {dynamic_max_dd:.2%}")
    print("-" * 30)

    print(
        f"beta range: {kalman_beta_series.min():.6f} to {kalman_beta_series.max():.6f}"
    )
    print(f"static turnover:  {static_res['turnover'].sum():.2f}")
    print(f"dynamic turnover: {dynamic_res['turnover'].sum():.2f}")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(kalman_beta_series, label="Kalman Beta")
    ax2.axhline(hedge_ratio, color="red", label="Static OLS Beta")
    ax2.set_title("Beta Comparison")
    ax2.legend()
    plt.show()

    # 9. Plot and Save Equity Curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        static_res.index,
        static_res["cumulative_return"],
        label="static OLS",
        color="gray",
        linewidth=1.5,
    )
    ax.plot(
        dynamic_res.index,
        dynamic_res["cumulative_return"],
        label="Dynamic Kalman",
        color="blue",
        linewidth=1.5,
    )
    ax.legend()
    ax.set_title(
        f"Statistical Arbitrage P&L: {s1} vs {s2}", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)

    # Create results directory string dynamically
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / f"03_{s1}_{s2}_kalman_vs_static_pnl.png"

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Equity curve saved to: {save_path}")


if __name__ == "__main__":
    main()
