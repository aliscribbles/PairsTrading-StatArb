import numpy as np
import pandas as pd
import os

from stat_arb.cointegration import screen_pairs
from stat_arb.plotting import plot_correlation_matrix
import matplotlib.pyplot as plt

SECTORS = {'Technology': ['MSFT', 'AAPL', 'GOOGL', 'META'],
           'Financials': ['JPM', 'BAC', 'GS', 'MS'],
           'Energy': ['XOM', 'CVX', 'COP', 'EOG'],
           'Consumer_Staples': ['KO', 'PEP', 'MCD', 'YUM'],
           'Healthcare': ['JNJ', 'PFE', 'ABT', 'MDT'],
}


def main():
    
    print ("Loading Data ...")

    prices,returns, log_prices = load_and_prep_data()
    print ("Data Loaded Successfully ... \n")
    
    print("Generating Correlation Heatmap")
    
    fig = plot_correlation_matrix(returns, sector_groups=SECTORS)
    
    # Save the Output
    fig.savefig('../results/01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("Heatmap saved to results/01_correlation_heatmap.png\n")
    

    print("Screening for Correlated Pairs ...\n")
        
    correlated_pairs_df = screen_pairs(log_prices, corr_threshold=0.85)
    print(f"Found {len(correlated_pairs_df)} highly correlated pairs with correlation above 0.85\n")
    print(correlated_pairs_df.to_string(index=False))
    
    
if __name__ == "__main__":
    main()



