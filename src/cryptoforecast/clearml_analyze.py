import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt

from clearml import Task

from constants import DATA_BEAR_PATH, DATA_BTC_DAY_PATH, DATA_BULL_PATH

# Импорт вашего анализа
from analysis_script import analyze_correlation_advanced

# ==========================================
# 1. CLEARML TASK
# ==========================================

task = Task.init(
    project_name="CryptoForecast",
    task_name="btc_news_sentiment_analysis",
    task_type=Task.TaskTypes.training
)

logger = task.get_logger()

# ==========================================
# 2. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ==========================================

START_DATE = "2025-01-16"
END_DATE = "2025-08-16"

params = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "candle_interval_minutes": 60
}

task.connect(params)

# ==========================================
# 3. ЗАГРУЗКА ДАННЫХ
# ==========================================

print("📂 Загрузка данных...")

candles_df = pd.read_csv(DATA_BTC_DAY_PATH, parse_dates=['Open time'])
bull_df = pd.read_csv(DATA_BULL_PATH, parse_dates=['datetime'])
bear_df = pd.read_csv(DATA_BEAR_PATH, parse_dates=['datetime'])

# гарантируем наличие колонок
for col in ['positive_votes', 'negative_votes', 'important_votes']:
    if col not in bull_df:
        bull_df[col] = 0
    if col not in bear_df:
        bear_df[col] = 0

# корректируем голоса по типу новости
bull_df['negative_votes'] = 0
bear_df['positive_votes'] = 0

all_news = pd.concat([bull_df, bear_df], ignore_index=True)

print(f"✓ Загружено: {len(candles_df)} свечей, {len(all_news)} новостей")

# ==========================================
# 4. СОХРАНЕНИЕ DATASETS
# ==========================================

task.upload_artifact(
    name="candles_dataset",
    artifact_object=candles_df
)

task.upload_artifact(
    name="news_dataset",
    artifact_object=all_news
)

# ==========================================
# 5. ЗАПУСК АНАЛИЗА
# ==========================================

print("\n🚀 Запуск анализа...")

results, merged_data = analyze_correlation_advanced(
    candles_df,
    all_news,
    candle_interval_minutes=params["candle_interval_minutes"],
    start_date=params["start_date"],
    end_date=params["end_date"],
)

# ==========================================
# 6. ЛОГИРОВАНИЕ МЕТРИК
# ==========================================

if results and "base" in results:

    base = results["base"]

    logger.report_scalar(
        title="Correlation",
        series="pearson",
        value=base.get("corr", 0),
        iteration=0
    )

    logger.report_scalar(
        title="Correlation",
        series="r2",
        value=base.get("r2", 0),
        iteration=0
    )

    logger.report_scalar(
        title="Correlation",
        series="p_value",
        value=base.get("p_value", 0),
        iteration=0
    )

# ==========================================
# 7. RANDOM FOREST METRICS
# ==========================================

if results and "rf_validation" in results and results["rf_validation"]:

    rf = results["rf_validation"]

    logger.report_scalar(
        "RandomForest",
        "R2 Train",
        rf["r2_train"],
        iteration=0
    )

    logger.report_scalar(
        "RandomForest",
        "R2 Test",
        rf["r2_test"],
        iteration=0
    )

    logger.report_scalar(
        "RandomForest",
        "CV Mean",
        rf["cv_mean"],
        iteration=0
    )

# ==========================================
# 8. СОХРАНЕНИЕ MERGED DATASET
# ==========================================

task.upload_artifact(
    name="merged_dataset",
    artifact_object=merged_data
)

# ==========================================
# 9. СОХРАНЕНИЕ RESULTS JSON
# ==========================================

def convert_numpy(obj):

    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]

    if isinstance(obj, tuple):
        return [convert_numpy(v) for v in obj]

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    return obj


results_clean = convert_numpy(results)

task.upload_artifact(
    name="analysis_results",
    artifact_object=results_clean
)

# ==========================================
# 10. СОХРАНЕНИЕ ГРАФИКОВ
# ==========================================

figs = [plt.figure(i) for i in plt.get_fignums()]

for i, fig in enumerate(figs):

    logger.report_matplotlib_figure(
        title="analysis_plots",
        series=f"figure_{i}",
        figure=fig,
        iteration=0
    )

# ==========================================
# 11. ФИНАЛ
# ==========================================

logger.report_text("Analysis finished successfully")

print("\n✅ Анализ завершён")
print("📊 Результаты доступны в ClearML UI")

task.close()