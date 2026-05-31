# 파일 이름: my_metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def normalized_rmse(y_true, y_pred):
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    range_y = np.max(y_true) - np.min(y_true)
    return rmse / range_y if range_y != 0 else rmse

def pearson_correlation(y_true, y_pred):
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
        
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return np.clip(corr, 0, 1)

def competition_score(y_true, y_pred):
    nrmse = min(normalized_rmse(y_true, y_pred), 1)
    pearson = pearson_correlation(y_true, y_pred)
    return 0.5 * (1 - nrmse) + 0.5 * pearson