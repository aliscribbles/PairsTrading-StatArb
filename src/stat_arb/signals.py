import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def zscore_signals(zscore: pd.Series, entry=2, exit_=0.5, stop_loss=4) -> pd.DataFrame:
    """
    Generate long/short/exit signals from Z-score series.
    +1 = long spread, -1 = short spread, 0 = flat.

    Parameters:
        zscore (pd.Series): The rolling Z-score of the spread.
        entry (float): The Z-score threshold to enter a trade.
        exit_ (float): The Z-score threshold to exit a trade.
        stop_loss (float): The Z-score threshold to cut losses.

    Returns:
        pd.DataFrame: DataFrame with zscore and signal columns.
    """
    # 1. Initialize a DataFrame to store the daily signals
    sig = pd.DataFrame({"zscore": zscore, "signal": 0})
    position = 0

    # 2. Iterate through the time series step-by-step
    for i in range(1, len(sig)):
        z = sig["zscore"].iloc[i]

        # 3. Handle the NaN warmup period
        if np.isnan(z):
            sig.iloc[i, sig.columns.get_loc("signal")] = 0
            position = 0
            continue
        # 4. The State Machine
        if position == 0:
            if z < -entry:
                position = 1
            elif z > entry:
                position = -1
        elif position == 1:
            if z > -exit_ or abs(z) > stop_loss:
                position = 0
        elif position == -1 and (z < exit_ or abs(z) > stop_loss):
            position = 0

        sig.iloc[i, sig.columns.get_loc("signal")] = position

    return sig


# def kalman_hedge_ratio( y: pd.Series, x: pd.Series,
#                        delta: float = 1e-4, v_e: float = 1e-3,
#                        initial_beta: float = 0.0) ->pd.Series:
#     """
#     Calculates a dynamic hedge ratio using a 1D Kalman Filter.

#     Parameters:
#         y (pd.Series): Asset 1 log prices (the dependent variable).
#         x (pd.Series): Asset 2 log prices (the independent variable).
#         delta (float): Process noise variance.
#         v_e (float): Measurement noise variance.

#     Returns:
#         pd.Series: The rolling, dynamic hedge ratio.
#     """
#     #store index
#     index = y.index

#     #convert to numpy arrays
#     y= y.to_numpy()
#     x= x.to_numpy()

#     #Initializing arrays
#     beta = np.zeros(len(y))
#     P = np.zeros(len(y))

#     ## Start the filter at our known OLS baseline instead of 0
#     beta[0] = initial_beta
#     P[0] = 0.0

#     #Recursive Kalman Filter
#     for t in range(1,len(y)):
#         #Predict
#         beta_pred =beta[t-1]
#         P_pred = P[t-1] + delta

#         #Update
#         y_pred = x[t] * beta_pred
#         error = y[t] - y_pred
#         S = (x[t]**2) *P_pred +v_e
#         K = (P_pred * x[t]) / S

#         beta[t] = beta_pred + K * error
#         P[t] = (1- K * x[t]) * P_pred

#     #Wrap back into pandas Series
#     beta_series = pd.Series(beta,index = index,name= 'dynamic_hedge_ratio')
#     return beta_series

            
def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    v_e: float | None = None,
    diffuse: bool = True,
    P0: float = 1e4,
    train_window: int = 252,
    initial_beta: float = 0.0,
    initial_alpha: float = 0.0,
) -> pd.DataFrame:
    """
    Calculates dynamic alpha and beta using a 2D Kalman Filter.

    Parameters:
        y (pd.Series): Asset 1 log prices (the dependent variable).
        x (pd.Series): Asset 2 log prices (the independent variable).
        delta (float): Process noise variance, used directly as V_w = delta * I.
        (Chan uses V_w = delta/(1-delta) * I; values are not comparable.)
        v_e (float | None): Measurement noise variance. If None, estimated as the residual variance of an OLS fit on the first `train_window` observations.
        diffuse (bool): If True, P0 * I initialisation with theta = [0, 0] — no future information. If False, seeds theta from initial_alpha/initial_beta with P0 = 0 (dogmatic prior).
        P0 (float): Initial state covariance. Large (~1e4) = diffuse.

    Returns:
        pd.DataFrame: A dataframe containing 'alpha' and 'beta' over time.
    """
    # store index
    index = y.index

    if v_e is None:
        n_train = min(train_window, len(y))
        reg = OLS(y.iloc[:n_train], add_constant(x.iloc[:n_train])).fit()
        v_e = float(reg.resid.var())

    # convert to numpy arrays
    y_arr = y.to_numpy()
    x_arr = x.to_numpy()
    n = len(y)

    # 1. State vector [alpha, beta] (2 x N matrix)
    theta = np.zeros((2, n))

    # 2. State covariance matrix P (2 x 2 x N tensor)
    P = np.zeros((2, 2, n))

    if diffuse:
        P[:, :, 0] = P0 * np.eye(2)

    else:
        theta[:, 0] = [initial_alpha, initial_beta]

    # 3. Process noise covariance (V_w) and Identity Matrix (I)
    V_w = delta * np.eye(2)
    I = np.eye(2)

    e = np.full(n, np.nan)  # innovation
    Q = np.full(n, np.nan)  # innovation variance
    P_alpha = np.full(n, np.nan)
    P_beta = np.full(n, np.nan)

    # Recursive 2D Kalman Filter
    for t in range(1, n):
        # F is the observation matrix [1,x[t]]
        F = np.array([[1.0, x_arr[t]]])

        # Predict Step
        theta_pred = theta[:, t - 1 : t]  # Extract as 2x1 column vector
        P_pred = P[:, :, t - 1] + V_w

        # Update Step
        y_pred = F @ theta_pred
        error = y_arr[t] - y_pred[0, 0]

        # Innovation covariance S
        S = F @ P_pred @ F.T + v_e

        e[t] = error
        Q[t] = S[0, 0]

        # Kalman Gain K
        K = (P_pred @ F.T) / S[0, 0]

        # Update state and covariance matrices
        theta[:, t : t + 1] = theta_pred + K * error
        P[:, :, t] = (I - K @ F) @ P_pred

        P_alpha[t] = P[0, 0, t]
        P_beta[t] = P[1, 1, t]

    # Wrap back into pandas DataFrame
    return pd.DataFrame(
        {
            "alpha": theta[0, :],
            "beta": theta[1, :],
            "e": e,
            "Q": Q,
            "P_alpha": P_alpha,
            "P_beta": P_beta,
        },
        index=index,
    )


# def kalman_hedge_ratio(
#     y: pd.Series,
#     x: pd.Series,
#     delta: float = 1e-4,
#     v_e: float = 1e-3,
#     initial_beta: float = 0.0,
#     initial_alpha: float = 0.0,
# ) -> pd.DataFrame:
#     """
#     Calculates dynamic alpha and beta using a 2D Kalman Filter.

#     Parameters:
#         y (pd.Series): Asset 1 log prices (the dependent variable).
#         x (pd.Series): Asset 2 log prices (the independent variable).
#         delta (float): Process noise variance.
#         v_e (float): Measurement noise variance.

#     Returns:
#         pd.DataFrame: A dataframe containing 'alpha' and 'beta' over time.
#     """
#     # store index
#     index = y.index

#     # convert to numpy arrays
#     y = y.to_numpy()
#     x = x.to_numpy()
#     n = len(y)

#     # 1. State vector [alpha, beta] (2 x N matrix)
#     theta = np.zeros((2, n))
#     theta[:, 0] = [initial_alpha, initial_beta]

#     # 2. State covariance matrix P (2 x 2 x N tensor)
#     P = np.zeros((2, 2, n))

#     # 3. Process noise covariance (V_w) and Identity Matrix (I)
#     V_w = delta * np.eye(2)
#     I = np.eye(2)

#     # Recursive 2D Kalman Filter
#     for t in range(1, n):
#         # F is the observation matrix [1,x[t]]
#         F = np.array([[1.0, x[t]]])

#         # Predict Step
#         theta_pred = theta[:, t - 1 : t]  # Extract as 2x1 column vector
#         P_pred = P[:, :, t - 1] + V_w

#         # Update Step
#         y_pred = F @ theta_pred
#         error = y[t] - y_pred[0, 0]

#         # Innovation covariance S
#         S = F @ P_pred @ F.T + v_e

#         # Kalman Gain K
#         K = (P_pred @ F.T) / S[0, 0]

#         # Update state and covariance matrices
#         theta[:, t : t + 1] = theta_pred + K * error
#         P[:, :, t] = (I - K @ F) @ P_pred

#     # Wrap back into pandas DataFrame
#     return pd.DataFrame({"alpha": theta[0, :], "beta": theta[1, :]}, index=index)
