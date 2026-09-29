import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stat_arb.data import load_and_prep_data
from stat_arb.cointegration import screen_pairs, engle_granger_test, zscore_normalise
from stat_arb.signals import zscore_signals
from stat_arb.backtest import run_backtest


def main():
    print("Loading data ...")
    prices, returns, log_prices = load_and_prep_data()

    print("Screening universe for top pairs")

    # Get all pairs, then slice exactly the top 5
    pairs_df = screen_pairs(log_prices, corr_threshold=0.85)
    top_5_pairs = pairs_df.head(5)

    # Optimised parameters from parameter sweep

    OPTIMISED_WINDOW = 60
    OPTIMISED_ENTRY = 1.75
    OPTIMISED_EXIT = 0.5

    # Store daily returnns of each pair here
    portfolio_daily_returns = pd.DataFrame(index=prices.index)

    print("\nExecuting Portfolio Backtest:")

    # --- THE PORTFOLIO LOOP ---

    for index, row in top_5_pairs.iterrows():
        s1 = row["stock1"]
        s2 = row["stock2"]
        hedge_ratio = row["hedge_ratio"]

        print(f"Processing {s1} - {s2} (Hedge Ratio: {hedge_ratio:.2f})")

        # 1. Calculate the spread using engle_granger_test
        eg = engle_granger_test(log_prices[s1], log_prices[s2])
        hedge_ratio = eg["hedge_ratio"]
        spread = eg["spread"]

        # 2. Normalise the spread (Use OPTIMIZED_WINDOW)
        normalised_spread = zscore_normalise(spread, window=OPTIMISED_WINDOW)

        # 3. Generate Signals (Use OPTIMIZED_ENTRY and OPTIMIZED_EXIT)
        zscore_signals_df = zscore_signals(
            normalised_spread, entry=OPTIMISED_ENTRY, exit_=OPTIMISED_EXIT
        )

        # 4. Run Backtest (Remember mode='static')
        backtest = run_backtest(
            prices,
            s1,
            s2,
            signals=zscore_signals_df,
            hedge_ratio=hedge_ratio,
            mode="static",
        )
        # 5. Extract the 'daily_return' column from the backtest result
        daily_return = backtest["daily_return"]

        # Assign it to the portfolio dataframe
        portfolio_daily_returns[f"{s1}_{s2}"] = daily_return

    # --- AGGREGATION ---
    # Drop any NaNs caused by the rolling window warmup
    portfolio_daily_returns.dropna(inplace=True)

    # Calculate Equal-Weighted Portfolio Daily Return
    # By taking the mean across the columns (axis=1)
    portfolio_daily_returns["Portfolio_Return"] = portfolio_daily_returns.mean(axis=1)

    # Calculate Cumulative Return
    portfolio_cum_return = (
        1 + portfolio_daily_returns["Portfolio_Return"]
    ).cumprod() - 1

    # Calculate Max Drawdown
    rolling_max = (1 + portfolio_cum_return).cummax()
    drawdown = (1 + portfolio_cum_return) / rolling_max - 1
    max_dd = drawdown.min()

    final_pnl = portfolio_cum_return.iloc[-1]

    print("-" * 30)
    print("PORTFOLIO PERFORMANCE (Top 5 Pairs)")
    print("-" * 30)
    print(f"Total Cumulative Return: {final_pnl:.2%}")
    print(f"Maximum Drawdown:      {max_dd:.2%}")
    print(f"Return / Drawdown:     {abs(final_pnl/max_dd):.2f}")
    print("-" * 30)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        portfolio_cum_return.index, portfolio_cum_return, color="purple", linewidth=1.5
    )
    ax.set_title(
        "Statistical Arbitrage Portfolio P&L (Top 5 Pairs)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(BASE_DIR, "..", "results", "07_portfolio_pnl.png")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPortfolio equity curve saved to: {save_path}")


if __name__ == "__main__":
    main()
