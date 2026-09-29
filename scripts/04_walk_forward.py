import pandas as pd
from stat_arb.data import load_and_prep_data
from stat_arb.cointegration import screen_pairs
from stat_arb.backtest import run_walk_forward


def main():
    # 1. Load Data
    prices, returns, log_prices = load_and_prep_data()

    # 2. Dynamic Pair Selection
    print("Screening universe for cointegrated pairs...")
    pairs_df = screen_pairs(log_prices, corr_threshold=0.85)

    if pairs_df.empty:
        raise ValueError("No viable pairs found in the current universe.")

    # Extract the top row (assuming screen_pairs sorts by lowest ADF p-value)
    top_pair = pairs_df.iloc[0]
    s1 = top_pair["stock1"]
    s2 = top_pair["stock2"]

    print(f"Selected Top Pair: {s1}-{s2} (ADF p-value: {top_pair['adf_pvalue']:.4f})")
    # 3.  Execute Walk-Forward Validation
    unified_oos_curve, overall_metrics, fold_metrics = run_walk_forward(
        prices, log_prices, s1, s2
    )

    # 4. Evaluate Overall Metrics
    print("Overall Performance Metrics:")
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value}")

    # 5. Evaluate Fold Metrics
    print("\nFold-wise Performance Metrics:")
    print(fold_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
