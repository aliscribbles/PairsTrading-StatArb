import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#import data
from stat_arb.data import load_and_prep_data


# Only import the core mathematical and backtesting engines
from stat_arb.cointegration import screen_pairs, engle_granger_test, zscore_normalise
from stat_arb.signals import zscore_signals
from stat_arb.backtest import run_backtest


def main():
    
    # 1. Load Data
    prices, returns, log_prices = load_and_prep_data()
    
    # 2. Select Pair (Screen the universe and take the top result)
    pairs_df = screen_pairs(log_prices, corr_threshold=0.85)
    s1, s2 = pairs_df.loc[0, ['stock1', 'stock2']]
    print(f"Executing Strategy for Pair: {s1} - {s2}")
    
    # 3. Get Hedge Ratio
    hedge_ratio = pairs_df.iloc[0]['hedge_ratio']
    
    # 4. Construct Spread
    eg = engle_granger_test(log_prices[s1], log_prices[s2])
    spread = eg['spread']
    
    # 5. Normalize to Z-Score
    normalise = zscore_normalise(spread)
    
    # 6. Generate Signals
    signals = zscore_signals(normalise)
    
    # 7. Run Backtest
    backtest = run_backtest(prices, s1, s2, signals, hedge_ratio)
    
    # 8. Print Performance Metrics
    final_pnl = backtest['cumulative_return'].iloc[-1]
    max_dd = backtest['drawdown'].min()
    
    print("-" * 30)
    print("STRATEGY PERFORMANCE")
    print("-" * 30)
    print(f"Total Cumulative Return: {final_pnl:.2%}")
    print(f"Maximum Drawdown:      {max_dd:.2%}")
    print("-" * 30)
    
    # 9. Plot and Save Equity Curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(backtest.index, backtest['cumulative_return'], color='teal', linewidth=1.5)
    ax.set_title(f'Statistical Arbitrage P&L: {s1} vs {s2}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)
    
    # Create results directory string dynamically
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / f"02_{s1}_{s2}_pnl.png"

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Equity curve saved to: {save_path}")

if __name__ == "__main__":
    main()

    