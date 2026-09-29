import pandas as pd
import numpy as np


def run_backtest(
    prices: pd.DataFrame,
    s1: str,
    s2: str,
    signals: pd.DataFrame,
    hedge_ratio: float | None = None,
    hedge_ratios: pd.Series | None = None,
    mode: str = "static",
    transaction_cost: float = 0.0005,
    notional: float = 1_000_000,
    normalise_by_gross: bool = False,
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
    # Hedge Ratio Handling
    if mode == "static":

        if hedge_ratio is None:
            raise ValueError("Static mode requires hedge_ratio = (float).")
        hr = pd.Series(float(hedge_ratio), index=prices.index)

    elif mode == "dynamic":
        if hedge_ratios is None:
            raise ValueError("Dynamic mode requires hedge_ratios = (Series).")
        hr = hedge_ratios.reindex(prices.index).ffill()

    else:
        raise ValueError("mode must be 'static' or 'dynamic'")

    # Signals
    sig = signals["signal"].reindex(prices.index)

    # Trade only where BOTH a signal and a hedge ratio exist.
    valid = hr.notna() & sig.notna()
    idx = prices.index[valid]
    if len(idx) == 0:
        raise ValueError("No overlapping signal / hedge-ratio dates")

    # Returns
    hr, sig = hr.loc[idx], sig.loc[idx]
    asset_return_r1 = prices[s1].loc[idx].pct_change().fillna(0.0)
    asset_return_r2 = prices[s2].loc[idx].pct_change().fillna(0.0)

    # Weights held into each day ( set at previous close)
    w1 = sig.shift(1).fillna(0.0)
    w2 = -(sig * hr).shift(1).fillna(0.0)

    # Gross return
    gross_return = w1 * asset_return_r1 + w2 * asset_return_r2

    # Turnover on BOTH legs, charged the day the trade lands
    turnover = w1.diff().abs().fillna(w1.abs()) + w2.diff().abs().fillna(w2.abs())
    tc = turnover * transaction_cost

    if normalise_by_gross:  # return on capital employed
        gross_exposure = (w1.abs() + w2.abs()).replace(0.0, np.nan)
        net_return = (gross_return / gross_exposure).fillna(0.0) - tc
    else:
        net_return = gross_return - tc

    # Cumulative PnL
    cum_pnl = (1 + net_return).cumprod()

    # Drawdown

    dd = cum_pnl / cum_pnl.cummax() - 1

    # Output
    return pd.DataFrame(
        {
            "signal": sig,
            "daily_return": net_return,
            "cumulative_return": cum_pnl - 1,
            "drawdown": dd,
            "daily_pnl_usd": net_return * notional,
            "cumulative_pnl_usd": (cum_pnl - 1) * notional,
            "hedge_ratio": hr,
            "w1": w1,
            "w2": w2,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": tc,
        },
        index=idx,
    )


def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates a suite of annualized performance metrics from a daily return series.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)

    # Sharpe Ratio
    excess_ret = r - (risk_free_rate / 252)
    sharpe = (
        (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
        if excess_ret.std() > 0
        else 0.0
    )

    # Sortino Ratio (Downside deviation only)
    downside = r[r < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (excess_ret.mean() * 252) / downside_std if downside_std > 0 else 0.0

    # Maximum Drawdown
    cum_ret = (1 + r).cumprod()
    drawdown = (cum_ret / cum_ret.cummax()) - 1
    max_dd = drawdown.min()

    # Win Rate
    win_rate = (r > 0).mean()

    return {
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "n_days": len(r),
    }


import pandas as pd
import numpy as np
from stat_arb.cointegration import engle_granger_test, zscore_normalise
from stat_arb.signals import zscore_signals
from stat_arb.backtest import run_backtest, performance_metrics


def run_walk_forward(
    prices: pd.DataFrame,
    log_prices: pd.DataFrame,
    s1: str,
    s2: str,
    train_days: int = 504,
    test_days: int = 252,
    zscore_window: int = 60,
):
    """
    Executes walk-forward out-of-sample validation for a statistical arbitrage pairs trading strategy.

    Parameters:
        prices (pd.DataFrame): DataFrame containing raw adjusted stock prices for PnL calculation.
        log_prices (pd.DataFrame): DataFrame containing natural log stock prices for cointegration testing.
        s1 (str): Ticker symbol for the primary asset.
        s2 (str): Ticker symbol for the secondary asset.
        train_days (int): Number of days in the rolling training window.
        test_days (int): Number of days in the forward testing window.
        zscore_window (int): Lookback period for calculating the rolling Z-score.

    Returns:
        unified_oos_curve (pd.Series): The out-of-sample cumulative return curve across all folds.
        overall_metrics (dict): Annualized performance metrics for the unified curve.
        fold_metrics (pd.DataFrame): Chronological log of performance metrics for each test fold.
    """

    from stat_arb.cointegration import engle_granger_test, zscore_normalise
    from stat_arb.signals import zscore_signals

    # Store out-of-sample (OOS) daily returns
    oos_returns = []

    # Track performance metrics per fold
    fold_metrics = []

    total_days = len(prices)
    start_idx = 0

    while start_idx + train_days < total_days:

        # 1. Define window boundaries
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, total_days)

        # 2. Slice the Data
        train_prices = log_prices.iloc[start_idx:train_end_idx]
        test_prices = prices.iloc[train_end_idx:test_end_idx]

        # Engle-Granger and cointegration test and spread construction are strictly performed in log-space to ensure the spread represents a proportional (percentage) relationship rather than an absolute dollar difference
        # calculating the hedge ratio and spread using raw dollars, the spread would artificially widen over the years simply because the underlying stock prices appreciated, completely breaking the mean-reversion mechanics.

        # 3. Train the Model (Calculate Beta on log prices)
        train = engle_granger_test(train_prices[s1], train_prices[s2])
        train_beta = train["hedge_ratio"]

        # 4. The "Warm Start" Spread Construction
        warm_start_idx = max(0, train_end_idx - zscore_window)
        warm_price = log_prices.iloc[warm_start_idx:test_end_idx]
        spread = warm_price[s1] - (train_beta * warm_price[s2])
        zscore = zscore_normalise(spread, window=zscore_window)

        # 5. Generate OOS signals
        test_zscore = zscore.iloc[
            zscore_window:
        ]  # Drop the warm-up period for OOS evaluation, what remains perfectly matches the test_prices index
        test_signals = zscore_signals(test_zscore)

        # 6. Run OOS backtest
        backtest = run_backtest(
            test_prices,
            s1,
            s2,
            signals=test_signals,
            hedge_ratio=train_beta,
            mode="static",
        )
        oos_returns.append(backtest["daily_return"])

        # Track performance metrics specific to this fold
        fold_pm = performance_metrics(backtest["daily_return"])
        fold_metrics.append(
            {
                "test_start": test_prices.index[0].date(),
                "test_end": test_prices.index[-1].date(),
                "train_beta": round(train_beta, 4),
                "oos_sharpe": fold_pm.get("sharpe_ratio", 0.0),
                "oos_return_pct": fold_pm.get("ann_return_pct", 0.0),
            }
        )

        # 7. Shift the window forward
        start_idx += test_days

    # 8. Aggregate and Return
    unified_oos_returns = pd.concat(oos_returns)
    unified_oos_curve = (1 + unified_oos_returns).cumprod() - 1

    # Run performance metrics on the unified curve
    overall_metrics = performance_metrics(unified_oos_returns, risk_free_rate=0.0)

    return unified_oos_curve, overall_metrics, pd.DataFrame(fold_metrics)


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
