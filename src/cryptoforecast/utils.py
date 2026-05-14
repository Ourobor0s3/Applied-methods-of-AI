# ============== Импорты ==============
"""Метрики и утилиты для обучения моделей."""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# ============== Вспомогательные метрики ==============

def _roc_auc_safe(y_true, y_score):
    """ROC AUC с fallback на 0.5 для вырожденных случаев (только один класс)."""
    return 0.5 if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))


# ============== Вычисление метрик ==============

def calculate_regression_metrics(y_true, y_pred):
    """Метрики регрессии: RMSE, R² и MAE для оценки качества предсказания."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def calculate_classification_metrics(y_true, y_pred, y_score=None):
    """Метрики классификации: ROC AUC и Accuracy.
    y_score — вероятности (для ROC AUC), если None — берём y_pred.
    Threshold бинаризации: 0.5."""
    y_true_arr = np.asarray(y_true).astype(int) if np.asarray(y_true).max() > 1 else np.asarray(y_true).astype(int)
    y_pred_arr = (np.asarray(y_pred) >= 0.5).astype(int)
    score = y_score if y_score is not None else np.asarray(y_pred).astype(float)
    return {
        "roc_auc": _roc_auc_safe(y_true_arr, score),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
    }


def calculate_metrics(y_true, y_pred, task_type="regression"):
    """Универсальный диспетчер метрик по типу задачи."""
    return calculate_classification_metrics(y_true, y_pred) if task_type == "classification" else calculate_regression_metrics(y_true, y_pred)


def get_primary_metric_value(metrics, task_type="regression"):
    """Извлекает значение основной метрики из словаря метрик."""
    return metrics.get("r2" if task_type == "regression" else "roc_auc", 0.0)


def get_metrics_config(task_type="regression"):
    """Возвращает конфиг метрик: имя основной метрики и список всех."""
    if task_type == "classification":
        return {"primary_metric": "roc_auc", "metrics": ["roc_auc", "accuracy"]}
    return {"primary_metric": "r2", "metrics": ["rmse", "r2", "mae"]}


# ============== Ранняя остановка ==============

def should_stop_early(train_losses, val_losses, patience, delta):
    """Проверяет условие ранней остановки по dynamic patience.
    Возвращает (should_stop, best_epoch_idx).
    stop_epoch возвращается как 0, если ранняя остановка не сработала."""
    if len(val_losses) < patience + 1:
        return False, 0
    best_val = min(val_losses[-(patience + 1):])
    return val_losses[-1] > best_val + delta, val_losses.index(best_val)


# ============== Визуализация ==============
# Все функции принимают необязательный logger для загрузки графиков в ClearML.
# При logger=None графики сохраняются только в файл (save_path).


def plot_training_losses(train_losses, val_losses, model_name, save_path=None,
                         logger=None, title_suffix=""):
    """Кривые обучения и валидации с отметкой ранней остановки.
    Для нейросетей — ключевой индикатор переобучения."""
    if not train_losses:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    # Пытаемся найти эпоху ранней остановки в последних patience эпохах
    stop_epoch = 0
    for i in range(len(val_losses) - 1, max(0, len(val_losses) - 10), -1):
        if val_losses[i] > min(val_losses[:i + 1]) + 0.001:
            stop_epoch = i + 1
            ax.axvline(x=stop_epoch, color="orange", linestyle="--",
                       label=f"Early Stop (ep {stop_epoch})")
            break

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(f"Training Losses{suffix} — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("Training/Loss", model_name, fig, iteration=0)
    plt.close(fig)
    return fig


def plot_prediction_scatter(y_true, y_pred, model_name, task_type="regression",
                            save_path=None, logger=None, title_suffix=""):
    """Scatter-график предсказаний vs реальных значений + residuals.
    Для регрессии: линия y=x и R² на графике.
    Для классификации: порог 0.5."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Левый: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors="none", s=20)
    if task_type == "regression":
        mn, mx = y_true.min(), y_true.max()
        axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="y=x")
        if len(y_true) > 1:
            r2 = r2_score(y_true, y_pred)
            axes[0].text(0.05, 0.95, f"R² = {r2:.4f}", transform=axes[0].transAxes,
                        fontsize=12, va="top")
    else:
        axes[0].axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    suffix = f" — {title_suffix}" if title_suffix else ""
    axes[0].set_title(f"Actual vs Predicted{suffix}\n{model_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Правый: Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors="none", s=20)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"Residuals vs Predicted\n{model_name}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("Predictions", model_name, fig, iteration=0)
    plt.close(fig)
    return fig


def plot_roc_curve(y_true, y_score, model_name, save_path=None,
                   logger=None, title_suffix=""):
    """ROC AUC кривая — основная метрика качества классификации.
    AUC = 0.5 означает случайное угадывание, AUC = 1.0 — идеальное."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(f"ROC Curve{suffix}\n{model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("ROC Curve", model_name, fig, iteration=0)
    plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None,
                          logger=None, title_suffix=""):
    """Матрица ошибок для бинарной классификации.
    Показывает True Positive / False Positive / True Negative / False Negative."""
    y_true, y_pred = np.asarray(y_true).astype(int), (np.asarray(y_pred) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True,
                xticklabels=["Down/Flat", "Up"], yticklabels=["Down/Flat", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(f"Confusion Matrix{suffix}\n{model_name}")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("Confusion Matrix", model_name, fig, iteration=0)
    plt.close(fig)
    return fig


def plot_model_comparison(results, primary_metric, task_type, save_path=None,
                           logger=None, title_suffix=""):
    """Bar chart сравнения моделей по основной метрике (R² для регрессии, ROC AUC для классификации).
    Для регрессии: Y начинается от 0.
    Для классификации: Y в диапазоне [min-0.1, 1.0] для наглядности различий."""
    models = list(results.keys())
    primary_vals = [results[m]["metrics"].get(primary_metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, primary_vals, color=colors, edgecolor="black", linewidth=0.5)

    # Подписи значений над столбцами
    for bar, val in zip(bars, primary_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(primary_metric.upper())
    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(f"Model Comparison ({primary_metric.upper()}){suffix}")
    ax.grid(True, axis="y", alpha=0.3)

    # Масштаб оси Y зависит от типа задачи
    if task_type == "regression":
        ax.set_ylim(0, max(primary_vals) * 1.2 if max(primary_vals) > 0 else 1)
    else:
        ax.set_ylim(min(primary_vals) - 0.1, 1.0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("Model Comparison", primary_metric, fig, iteration=0)
    plt.close(fig)
    return fig


def plot_feature_importance(importance_df, model_name, save_path=None,
                            logger=None, title_suffix=""):
    """Горизонтальный bar chart важности признаков для XGBoost (top-20).
    Показывает, какие признаки больше всего влияют на предсказание."""
    if importance_df.empty:
        return None

    top = importance_df.head(20).copy()
    top["feature"] = top["feature"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"], color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    suffix = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(f"Feature Importance — Top 20{suffix}\n{model_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("Feature Importance", model_name, fig, iteration=0)
    plt.close(fig)
    return fig


def plot_prediction_distribution(y_true, y_pred, model_name,
                                  task_type="classification", save_path=None,
                                  logger=None, title_suffix=""):
    """Распределение предсказаний vs реальных значений.
    Для классификации: гистограммы вероятностей по классам + общая гистограмма.
    Для регрессии: overlay гистограмм + линейный график (sorted) для визуального сравнения."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if task_type == "classification":
        # Левый: предсказания по классам
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(y_pred[y_true == 0], bins=30, alpha=0.6,
                     label="Down/Flat (actual)", color="red")
        axes[0].hist(y_pred[y_true == 1], bins=30, alpha=0.6,
                     label="Up (actual)", color="green")
        axes[0].axvline(x=0.5, color="black", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Predicted Probability")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].set_title(f"Predictions by Class\n{model_name}")
        axes[0].grid(True, alpha=0.3)

        # Правый: общее распределение вероятностей
        axes[1].hist(y_pred, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        axes[1].axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold 0.5")
        axes[1].set_xlabel("Predicted Probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Probability Distribution\n{model_name}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Левый: overlay гистограмм
        axes[0].hist(y_true, bins=40, alpha=0.6, label="Actual", color="blue")
        axes[0].hist(y_pred, bins=40, alpha=0.6, label="Predicted", color="orange")
        axes[0].set_xlabel("Volatility (log)")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].set_title(f"Target Distribution\n{model_name}")
        axes[0].grid(True, alpha=0.3)

        # Правый: линейный график отсортированных значений
        sorted_idx = np.argsort(y_true)
        axes[1].plot(np.arange(len(y_true)), y_true[sorted_idx],
                     "b-", linewidth=1, label="Actual", alpha=0.8)
        axes[1].plot(np.arange(len(y_true)), y_pred[sorted_idx],
                     "r-", linewidth=1, label="Predicted", alpha=0.8)
        axes[1].set_xlabel("Sample (sorted by actual)")
        axes[1].set_ylabel("Volatility (log)")
        axes[1].legend()
        axes[1].set_title(f"Predictions vs Actual (sorted)\n{model_name}")
        axes[1].grid(True, alpha=0.3)

    suffix = f" — {title_suffix}" if title_suffix else ""
    fig.suptitle(suffix, fontsize=1, y=0.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if logger:
        logger.report_matplotlib_figure("Prediction Distribution", model_name, fig, iteration=0)
    plt.close(fig)
    return fig