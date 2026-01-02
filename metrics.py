import numpy as np

def mse(y_true, y_pred):
    """
    Mean Squared Error

    Measures average squared difference
    between actual and predicted values.

    Large mistakes are punished more.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error

    Same as MSE but easier to understand
    because it is in the same unit as target.
    """
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Mean Absolute Error

    Average of absolute differences.
    Treats all errors equally.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    R² Score

    Tells how much variation in the output
    is explained by the model.

    1.0 = perfect
    0.0 = no better than guessing mean
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)
