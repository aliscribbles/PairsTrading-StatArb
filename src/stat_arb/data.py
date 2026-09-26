import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prep_data(file_path=None):
    """
    Loads and prepares data

    Parameters
    ----------
    file_path (str): Path to the CSV file containing stock prices. If None, defaults to the standard data directory.
    Returns:
        prices (pd.DataFrame): DataFrame containing raw adjusted stock prices
        returns(pd.DataFrame): DataFrame containing daily returns
        log_prices(pd.DataFrame): DataFrame containing natural log adjusted stock prices

    """
    if file_path is None:
        BASE_DIR = Path(__file__).resolve().parents[2]
        csv_path = BASE_DIR / "data" / "stock_prices.csv"

    # Data Ingestion
    prices = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    returns = prices.pct_change().dropna()
    log_prices = np.log(prices)

    return prices, returns, log_prices
