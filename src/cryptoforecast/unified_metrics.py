"""
Унифицированные метрики: регрессия и классификация.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Метрики регрессии: MAE, RMSE, R2"""
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2_score(y_true, y_pred))
    }


def _roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Безопасный ROC AUC (защита от одного класса)"""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> dict:
    """Метрики классификации: Accuracy, ROC AUC, Precision, Recall, F1"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    if y_score is not None:
        metrics['roc_auc'] = _roc_auc_safe(y_true, y_score)
    else:
        metrics['roc_auc'] = _roc_auc_safe(y_true, y_pred.astype(float))
    
    return metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str = 'regression', threshold: float = 0.5) -> dict:
    """
    Унифицированный расчет метрик в зависимости от типа задачи.
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения (вероятности для классификации)
        task_type: 'regression' или 'classification'
        threshold: Порог для бинарной классификации
    
    Returns:
        dict: Словарь с метриками
    """
    if task_type == 'classification':
        # Бинарная классификация
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_true_binary = y_true.astype(int) if y_true.max() <= 1 else y_true
        return calculate_classification_metrics(y_true_binary, y_pred_binary, y_pred)
    else:
        # Регрессия
        return calculate_regression_metrics(y_true, y_pred)


def get_metrics_config(task_type: str = 'regression') -> dict:
    """Конфигурация метрик в зависимости от типа задачи"""
    if task_type == 'classification':
        return {
            'primary_metric': 'roc_auc',
            'metrics': ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        }
    else:
        return {
            'primary_metric': 'r2',
            'metrics': ['mae', 'rmse', 'r2']
        }


def get_primary_metric_value(metrics: dict, task_type: str = 'regression') -> float:
    """Получить значение основной метрики для сравнения"""
    config = get_metrics_config(task_type)
    primary = config['primary_metric']
    return metrics.get(primary, 0.0)


def is_better(current: float, new: float, task_type: str = 'regression') -> bool:
    """Сравнить метрики (больше = лучше для ROC AUC, R2; меньше = лучше для MAE, RMSE)"""
    if task_type == 'classification':
        return new > current  # ROC AUC: больше лучше
    else:
        return new > current  # R2: больше лучше