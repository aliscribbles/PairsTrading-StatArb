import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant 


# def screen_pairs(log_prices: pd.DataFrame, corr_threshold:float = 0.85, 
#                  coint_threshold: float = 0.05) -> pd.DataFrame:
#     """    
#     Screens the universe for highly correlated pairs
    
#     Parameters:
#     log_prices (pd.DataFrame): Natural log of asset prices
#     corr_threshold (float): Minimum Pearson correlation coefficient threshold = 0.85
    
    
#     Returns:
#     pd.DataFrame: A dataframe containing 'pair', 'stock1', 'stock2', and 'correlation coefficient'
    
#     """
#     if log_prices.empty:
#         raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")
    
#     tickers = list(log_prices.columns)
#     rows = []
    
#     for i in range (len(tickers)):
#         for j in range (i+1, len(tickers)):
#             s1,s2= tickers[i],tickers[j]
#             corr= log_prices[s1].corr(log_prices[s2])
#             if corr < corr_threshold:continue
#             res = engle_granger_test(log_prices[s1], log_prices[s2])
#             if not res['cointegrated']: continue
#             hl = half_life(res['spread'])
#             if hl>60 or hl<=0: continue
#             rows.append({
#                 'pair':(f'{s1}-{s2}'),
#                 'stock1':s1,
#                 'stock2':s2,
#                 'correlation':round(corr,4),
#                 'adf_pvalue':res['p_value'],
#                 'hedge_ratio':res['hedge_ratio'],
#                 'half_life': hl,
#             })          
                       
    
    
#The Logic    
def screen_pairs(log_prices:pd.DataFrame, corr_threshold:float = 0.85) ->pd.DataFrame:
    """
    Screens the universe for highly correlated pairs.
    
    Parameters:
    log_prices (pd.DataFrame): Natural log of asset prices
    corr_threshold (float): Minimum Pearson correlation coefficient threshold = 0.85
    
    Returns:
    pd.DataFrame: A dataframe containing 'pair', 'stock1', 'stock2', and 'correlation coefficient'
    """
    if log_prices.empty:
        raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")
    
    tickers = list(log_prices.columns)
    rows= []
    
    for i in (range(len(tickers))):
        for j in range(i+1, len(tickers)):
            s1,s2=tickers[i],tickers[j]
            corr= log_prices[s1].corr(log_prices[s2])
            if corr< corr_threshold: continue
            res=engle_granger_test(log_prices[s1], log_prices[s2])
            if not res['cointegrated']: 
                continue
            hl = half_life(res['spread'])
            if hl>60 or hl<=0: 
                continue
            rows.append({
                'pair':            (f'{s1}-{s2}'),
                'stock1':          s1,
                'stock2':          s2,
                'correlation':     round(corr,4),
                'hedge_ratio':     round(res['hedge_ratio'],6),
                'adf_pvalue':      round(res['p_value'],4),
                'half_life':        hl,
                
            })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('adf_pvalue', ascending=True).reset_index(drop=True)

#The Engle-Granger Test
def engle_granger_test(y: pd.Series, x: pd.Series, significance: float = 0.05) -> dict:
    
    
    """
    Performs the Engle-Granger two step co-integration test
    
    Returns:
        Dict: Contains 'hedge_ratio', 'adf_stat', 'p_value', 'cointegrated' (bool), and 'spread'.
        
    """
    reg = OLS(y, add_constant(x)).fit()    
    hedge_ratio = reg.params.iloc[1]
    spread = y-hedge_ratio * x
    
    adf = adfuller(spread.dropna(), autolag='AIC')
    
    return{ 
        'hedge_ratio':      round(hedge_ratio,6),
        'adf_stat':         round(adf[0],4),
        'p_value':          round(adf[1],4),
        'critical_values':  adf[4],
        'cointegrated':     adf[1] < significance,
        'spread':           spread,
        'r_squared':        round(reg.rsquared,4)
    }


def expanding_ols_beta(y: pd.Series, x: pd.Series,
                       min_periods: int = 252) -> pd.DataFrame:
    """
    Walk-forward OLS: alpha and beta at time t fitted only on data
    strictly before t. Equivalent to the Kalman filter with delta = 0.
    Rows before min_periods are NaN.
    """
    n = len(y)
    alpha = np.full(n, np.nan)
    beta  = np.full(n, np.nan)

    for t in range(min_periods, n):
        params = OLS(y.iloc[:t], add_constant(x.iloc[:t])).fit().params
        alpha[t] = params.iloc[0]
        beta[t]  = params.iloc[1]

    return pd.DataFrame({"alpha": alpha, "beta": beta}, index=y.index)

#The Half-Life of Mean Reversion    
def half_life(spread: pd.Series) -> float:
    """
    Mean-reversion half-life via AR(1): HL = -ln(2) / phi.
    """
    sp = spread.dropna()
    lag = sp.shift(1).dropna()
    delta = sp.diff().dropna()
    
    common = lag.index.intersection(delta.index)
    
    reg= OLS(delta.loc[common], add_constant(lag.loc[common])).fit()
    phi = reg.params.iloc[1]
    
    if phi>=0:
        return float('inf')
    
    
    return round(-np.log(2)/phi, 2)

#The Normalizer
def zscore_normalise(spread:pd.Series, window: int = 60) -> pd.Series:
    """
    Normalizes the spread to a z-score using a rolling window.
    
    Parameters:
    spread (pd.Series): The spread series to normalize.
    window (int): The rolling window size for mean and std deviation.
    
    Returns:
    pd.Series: The z-score normalized spread.
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore