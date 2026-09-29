import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns  # Standard library for heatmaps

# Import your core engines
# Adjust these imports based on how you structured your data loader in previous scripts
from stat_arb.cointegration import engle_granger_test, zscore_normalise
from stat_arb.data import load_and_prep_data
from stat_arb.signals import zscore_signals
from stat_arb.backtest import run_backtest


def main():
    # 1. Load Data
    print("Loading data...")
    prices, returns, log_prices = load_and_prep_data()

    s1, s2 = "AAPL", "MSFT"  # Or whichever pair you prefer

    # Extract the static spread
    eg = engle_granger_test(log_prices[s1], log_prices[s2])
    hedge_ratio = eg["hedge_ratio"]
    spread = eg["spread"]

    # 2. Define Parameter Ranges
    windows = [20, 30, 40, 50, 60]
    entries = [1.5, 1.75, 2.0, 2.25, 2.5]
    exit_z = 0.5

    results = []

    print(f"Running Parameter Sweep for {s1}-{s2}...")

    # 3. The Grid Search Loop
    for w, entry in itertools.product(windows, entries):

        # A. Normalize spread with current window 'w'
        zscore_normalised_spread = zscore_normalise(spread, window=w)

        # B. Generate signals with current 'entry' (keep exit=exit_z)
        zscore_signals_df = zscore_signals(
            zscore_normalised_spread, entry=entry, exit_=exit_z
        )

        # C. Run Backtest (Remember to pass raw prices and the hedge_ratio)
        backtest = run_backtest(
            prices,
            s1,
            s2,
            signals=zscore_signals_df,
            hedge_ratio=hedge_ratio,
            mode="static",
        )

        # D. Extract Final PnL from the backtest results
        final_pnl = backtest["cumulative_return"].iloc[-1]

        # E. Append to results list
        results.append({"Window": w, "Entry": entry, "PnL": final_pnl})

    # 4. Convert to DataFrame
    results_df = pd.DataFrame(results)

    # 5. Pivot for the Heatmap (Rows = Entry, Columns = Window, Values = PnL)
    pivot_table = results_df.pivot(index="Entry", columns="Window", values="PnL")

    # 6. Plotting the Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="RdYlGn", ax=ax)
    ax.set_title(f"Parameter Sensitivity (PnL): {s1} vs {s2}")

    # 7. Save to results folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(BASE_DIR, "..", "results", f"06_{s1}_{s2}_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {save_path}")


if __name__ == "__main__":
    main()
