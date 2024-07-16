import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    # return np.mean(200 * np.abs(pred - true) / (pred + true + 1e-8))

def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))

def r_squared(pred, true):
    """
    Calculate R-squared score.
    
    Parameters:
    - y_true: numpy array of actual values
    - y_pred: numpy array of predicted values
    
    Returns:
    - r2: R-squared score
    """
    # Total sum of squares
    ss_total = np.sum((true - np.mean(true))**2)
    # Residual sum of squares
    ss_res = np.sum((true - pred)**2)
    # R-squared
    r2 = 1 - (ss_res / ss_total)
    return r2

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    r2 = r_squared(pred, true)

    return mae, mse, rmse, mape, mspe, smape, r2
