"""
Сводный анализ результатов обоих скриптов с унифицированными метриками.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

from experiment_config import ARTIFACTS_DIR
from unified_metrics import get_metrics_config


def create_unified_analysis_report():
    """
    Создает сводный отчет по результатам обоих анализов.
    """
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "price_analysis": {},
        "volatility_analysis": {},
        "cross_task_comparison": {}
    }
    
    # Анализ результатов price скрипта
    price_comparison_path = os.path.join(ARTIFACTS_DIR, "price_model_comparison.csv")
    if os.path.exists(price_comparison_path):
        price_df = pd.read_csv(price_comparison_path)
        
        # Динамическое извлечение метрик (колонки зависят от TASK_TYPE)
        first_row = price_df.iloc[0].to_dict()
        metric_keys = [k for k in first_row.keys() if k != 'model_name']
        
        report["price_analysis"] = {
            "best_model": first_row.get("model_name", "unknown"),
            "best_metrics": {k: float(first_row[k]) for k in metric_keys if pd.notna(first_row.get(k))},
            "models_count": len(price_df),
            "all_models": price_df.to_dict("records")
        }
    
    # Анализ результатов volatility скрипта
    volatility_comparison_path = os.path.join(ARTIFACTS_DIR, "volatility_model_comparison.csv")
    if os.path.exists(volatility_comparison_path):
        volatility_df = pd.read_csv(volatility_comparison_path)
        
        # Динамическое извлечение метрик
        first_row = volatility_df.iloc[0].to_dict()
        metric_keys = [k for k in first_row.keys() if k != 'model_name']
        
        report["volatility_analysis"] = {
            "best_model": first_row.get("model_name", "unknown"),
            "best_metrics": {k: float(first_row[k]) for k in metric_keys if pd.notna(first_row.get(k))},
            "models_count": len(volatility_df),
            "all_models": volatility_df.to_dict("records")
        }
    
    # Сохранение отчета
    report_path = os.path.join(ARTIFACTS_DIR, "unified_analysis_report.json")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report_path


def generate_unified_metrics_summary():
    """
    Генерирует сводку по унифицированным метрикам для обоих скриптов.
    """
    summary = {
        "regression": get_metrics_config("regression"),
        "classification": get_metrics_config("classification"),
    }
    
    return summary


if __name__ == "__main__":
    # Создание сводного отчета
    report_path = create_unified_analysis_report()
    print(f"Unified analysis report saved to: {report_path}")
    
    # Вывод сводки по метрикам
    metrics_summary = generate_unified_metrics_summary()
    print("\nUnified metrics summary:")
    print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))