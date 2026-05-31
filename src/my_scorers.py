# 파일 이름: my_scorers.py

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# NRMSE Scorer 클래스
class NRMSE_Scorer:
    def __init__(self, qt):
        self.qt = qt  # QuantileTransformer 객체를 클래스 내부에 저장

    def __call__(self, y_true, y_pred, **kwargs):
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            y_pred = y_pred.to_numpy()

        y_true_real = self.qt.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_real = self.qt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        if y_true_real.max() == y_true_real.min():
            return 0.0
        
        rmse = sqrt(mean_squared_error(y_true_real, y_pred_real))
        nrmse = rmse / (y_true_real.max() - y_true_real.min())
        return nrmse

# Pearson 상관계수 Scorer 클래스
class Pearsonr_Scorer:
    def __init__(self, qt):
        self.qt = qt

    def __call__(self, y_true, y_pred, **kwargs):
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            y_pred = y_pred.to_numpy()

        y_true_real = self.qt.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_real = self.qt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        if np.std(y_pred_real) < 1e-6 or np.std(y_true_real) < 1e-6:
            return 0.0
            
        r, _ = pearsonr(y_pred_real, y_true_real)
        return r