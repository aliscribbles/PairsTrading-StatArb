import pandas as pd
import numpy as np


def run_backtest(prices: pd.DataFrame, s1: str, s2:str,
                 signals: pd.DataFrame, 
                 hedge_ratio: float | None = None,
                 hedge_ratios: pd.Series | None = None,
                 mode: str = 'static', 
                 transaction_cost:float = 0.0005,
                 notional: float = 1_000_000,
                 normalise_by_gross: bool = False
                 ) -> pd.DataFrame:
    
    """
    Event-driven unified backtesting function supporting with explicit two-leg weights:
    - static hedge ratio (float)
    - dynamic hedge ratio (Series)
    
    weights at time t: w1 = signal, w2 = -signal* hedge_ratio
    Return earned t-1 -> t uses weights set at t-1.
    Turnover is measured on both legs, so dynamic-beta rebalancing is costed.
        
    Parameters:
        prices (pd.DataFrame): Raw adjusted close prices.
        s1 (str): Ticker for Asset 1 (y).
        s2 (str): Ticker for Asset 2 (x).
        signals (pd.DataFrame): The daily position signals (1, -1, 0).
        hedge_ratio (float or series): The static beta from the Engle-Granger test or dynamic beta from Kalman filter
        transaction_cost (float): 5bps cost per position change.
        
    Returns:
        pd.DataFrame: Contains 'signal', 'daily_return', 'cumulative_return', and 'drawdown'.
    """  
    #Hedge Ratio Handling
    if mode == 'static':
        
        if hedge_ratio is None:
            raise ValueError("Static mode requires hedge_ratio = (float).")
        hr =pd.Series(float(hedge_ratio), index=prices.index)
    
    elif mode == 'dynamic':
        if hedge_ratios is None:
            raise ValueError("Dynamic mode requires hedge_ratios = (Series).")
        hr = hedge_ratios.reindex(prices.index).ffill()
        
    else:
        raise ValueError("mode must be 'static' or 'dynamic'")
    
    #Signals
    sig=signals['signal'].reindex(prices.index)
    
    #Trade only where BOTH a signal and a hedge ratio exist.
    valid = hr.notna() & sig.notna()
    idx = prices.index[valid]
    if len(idx) ==0:
        raise ValueError("No overlapping signal / hedge-ratio dates")
    
    #Returns
    hr,sig = hr.loc[idx], sig.loc[idx]
    asset_return_r1 = prices[s1].loc[idx].pct_change().fillna(0.0)
    asset_return_r2 = prices[s2].loc[idx].pct_change().fillna(0.0)
    
    #Weights held into each day ( set at previous close)
    w1 = sig.shift(1).fillna(0.0)
    w2 = -(sig * hr).shift(1).fillna(0.0)
        
    #Gross return 
    gross_return = w1 * asset_return_r1 + w2 * asset_return_r2
    
    #Turnover on BOTH legs, charged the day the trade lands
    turnover = w1.diff().abs().fillna(w1.abs()) \
            + w2.diff().abs().fillna(w2.abs())
    tc = turnover * transaction_cost
            
    
    if normalise_by_gross: # return on capital employed
        gross_exposure = (w1.abs() + w2.abs()).replace(0.0, np.nan)
        net_return = (gross_return / gross_exposure).fillna(0.0) - tc
    else:
        net_return = gross_return - tc
        
    #Cumulative PnL    
    cum_pnl = ( 1+ net_return).cumprod()
    
    #Drawdown

    dd = cum_pnl / cum_pnl.cummax() - 1
    
    
    #Output
    return pd.DataFrame({
    'signal': sig,
    'daily_return': net_return,
    'cumulative_return': cum_pnl -1,
    'drawdown': dd,
    'daily_pnl_usd': net_return * notional,
    'cumulative_pnl_usd': (cum_pnl - 1) * notional,
    'hedge_ratio': hr,
    'w1': w1,
    'w2': w2,
    'gross_return': gross_return,
    'turnover': turnover,
    'transaction_cost': tc
    
    },index = idx
    )




# def run_backtest(prices: pd.DataFrame, s1: str, s2:str,
#                  signals: pd.DataFrame, 
#                  hedge_ratio: float | None = None,
#                  hedge_ratios: pd.Series | None = None,
#                  mode: str = 'static', 
#                  transaction_cost:float = 0.0005,
#                  notional: float = 1_000_000
#                  ) -> pd.DataFrame:
    
#     """
#     Event-driven unified backtesting function supporting:
#     - static hedge ratio (float)
#     - dynamic hedge ratio (Series)
        
#     Parameters:
#         prices (pd.DataFrame): Raw adjusted close prices.
#         s1 (str): Ticker for Asset 1 (y).
#         s2 (str): Ticker for Asset 2 (x).
#         signals (pd.DataFrame): The daily position signals (1, -1, 0).
#         hedge_ratio (float or series): The static beta from the Engle-Granger test or dynamic beta from Kalman filter
#         transaction_cost (float): 5bps cost per position change.
        
#     Returns:
#         pd.DataFrame: Contains 'signal', 'daily_return', 'cumulative_return', and 'drawdown'.
#     """  
#     #Hedge Ratio Handling
#     if mode == 'static':
        
#         if hedge_ratio is None:
#             raise ValueError("Static mode requires hedge_ratio = float.")
#         hr =pd.Series(hedge_ratio, index=prices.index)
    
#     elif mode == 'dynamic':
#         if hedge_ratios is None:
#             raise ValueError("Dynamic mode requires hedge_ratios = Series.")
#         hr = hedge_ratios.reindex(prices.index).ffill().fillna(0)
        
#     else:
#         raise ValueError("mode must be 'static' or 'dynamic'")
    
#     #Returns
#     asset_return_r1 = prices[s1].pct_change().fillna(0)
#     asset_return_r2 = prices[s2].pct_change().fillna(0)
    
#     #Spread Return
#     spread_ret = asset_return_r1 - (hr * asset_return_r2)
    
#     #Signals
#     sig=signals['signal'].reindex(prices.index).fillna(0)
    
#     #Transaction costs
#     position_change = sig.diff().abs()    
#     tc = position_change * transaction_cost
    
#     #Daily PnL
#     daily_pnl = (sig.shift(1).fillna(0) * spread_ret) - tc
    
#     #Cumulative PnL
#     cum_pnl = (1+ daily_pnl).cumprod()
    
#     #Drawdown
#     dd = (cum_pnl/cum_pnl.cummax()) -1
    
#     #Output
#     return pd.DataFrame({
#     'signal': sig,
#     'daily_return': daily_pnl,
#     'cumulative_return': cum_pnl -1,
#     'drawdown': dd,
#     'daily_pnl_usd': daily_pnl * notional,
#     'cumulative_pnl_usd': (cum_pnl - 1) * notional,
#     'hedge_ratio': hr,
    
#     },index = prices.index
#     )

    