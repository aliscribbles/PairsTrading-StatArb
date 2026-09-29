import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stat_arb.data import load_and_prep_data
from stat_arb.cointegration import regime_filter
from stat_arb.cointegration import screen_pairs
from stat_arb.signals import zscore_signals, regime_aware_signals
from stat_arb.backtest import run_backtest
from stat_arb.cointegration import zscore_normalise


def main():

    # 1. Load Data
    print("Loading and preparing data...")

    prices, returns, log_prices = load_and_prep_data()

    # 2. ynamically select your pair (e.g., AAPL and MSFT)
    print("Screening universe for cointegrated pairs...")
    pairs_df = screen_pairs(log_prices, corr_threshold=0.85)

    if pairs_df.empty:
        raise ValueError("No viable pairs found in the current universe.")

    # Extract the top row (assuming screen_pairs sorts by lowest ADF p-value)
    top_pair = pairs_df.iloc[0]
    s1 = top_pair["stock1"]
    s2 = top_pair["stock2"]
    hedge_ratio = top_pair["hedge_ratio"]

    print(f"Top Pair Selected: {s1} and {s2} (Hedge Ratio: {hedge_ratio:.2f})")

    # 3. Calculate the spread and Z-score using your existing functions.
    spread = log_prices[s1] - (hedge_ratio * log_prices[s2])
    std_zscore = zscore_normalise(spread)

    # --- The Blind Strategy ---

    # 4. Generate standard zscore_signals()
    blind_signals = zscore_signals(std_zscore)

    # 5. Run the static backtest and capture final PnL & Max Drawdown.
    blind_backtest = run_backtest(
        prices, s1, s2, hedge_ratio=hedge_ratio, mode="static", signals=blind_signals
    )

    blind_final_pnl = blind_backtest["cumulative_return"].iloc[-1]
    blind_max_drawdown = blind_backtest["drawdown"].min()

    # --- The Regime-Aware Strategy ---

    # 6. Calculate the binary mask using regime_filter(spread, window=120)
    binary_mask = regime_filter(spread)
    #    binary_mask = regime_filter(spread, threshold=0.25, window=90)

    # 7. Generate filtered signals using regime_aware_signals(blind_signals, binary_mask)
    filtered_signals = pd.DataFrame(
        {"signal": regime_aware_signals(blind_signals["signal"], binary_mask)}
    )

    # 8. Run the static backtest with the filtered signals and capture final PnL & Max Drawdown.

    filtered_backtest = run_backtest(
        prices, s1, s2, hedge_ratio=hedge_ratio, mode="static", signals=filtered_signals
    )
    filtered_final_pnl = filtered_backtest["cumulative_return"].iloc[-1]
    filtered_max_drawdown = filtered_backtest["drawdown"].min()

    # 9. Print the side-by-side performance metrics.
    print("\n=== Strategy Comparison ===")
    print(
        f"Blind Strategy  | PnL: {blind_final_pnl:.2%} | Max DD: {blind_max_drawdown:.2%}"
    )
    print(
        f"Filtered System | PnL: {filtered_final_pnl:.2%} | Max DD: {filtered_max_drawdown:.2%}\n"
    )

    # 10. (Optional) Plot the two equity curves against each other using matplotlib to visualize exactly where the kill switch saved the account.
    plt.figure(figsize=(12, 6))
    plt.plot(
        blind_backtest.index,
        blind_backtest["cumulative_return"],
        label="Blind Strategy",
        color="gray",
        alpha=0.7,
    )
    plt.plot(
        filtered_backtest.index,
        filtered_backtest["cumulative_return"],
        label="Regime-Aware Strategy",
        color="blue",
    )
    plt.title(
        f"Statistical Arbitrage P&L: {s1} vs {s2} (Regime Filtered)", fontweight="bold"
    )
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save and show the results
    from pathlib import Path

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / "05_regime_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved regime comparison plot to: {save_path}")


if __name__ == "__main__":
    main()
