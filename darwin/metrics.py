import numpy as np
from sklearn import metrics


def log_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1).mean()


def roc_auc_score(y, pred):
    return metrics.roc_auc_score(y, pred)
