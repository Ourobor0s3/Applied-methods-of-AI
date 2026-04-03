import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task

from constants import DATA_BEAR_PATH, DATA_BTC_DAY_PATH, DATA_BULL_PATH

# Импортируем функции из вашего скрипта анализа
from analysis_script import research_analysis, plot_research_results

# ==========================================
# 1. CLEARML TASK
# ==========================================

task = Task.init(
    project_name="CryptoForecast",
    task_name="btc_news_research_analysis_v2",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False  # Создаём новый эксперимент при каждом запуске
)

logger = task.get_logger()

# ==========================================
# 2. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# ==========================================

START_DATE = "2025-01-16"
END_DATE = "2025-12-16"
ARCHITECTURE = "densenet"  # 'resnet' или 'densenet'

params = {
    "start_date": START_DATE,
    "end_date": END_DATE,
    "architecture": ARCHITECTURE,
    "sentiment_model": "BERTweet (finiteautomata/bertweet-base-sentiment-analysis)",
    "volatility_window": 24,
    "candle_interval": "1H"
}

task.connect(params)

# ==========================================
# 3. ЗАГРУЗКА ДАННЫХ
# ==========================================

print("📂 Загрузка данных...")

candles_df = pd.read_csv(DATA_BTC_DAY_PATH, parse_dates=['Open time'])
bull_df = pd.read_csv(DATA_BULL_PATH, parse_dates=['datetime'])
bear_df = pd.read_csv(DATA_BEAR_PATH, parse_dates=['datetime'])

# Гарантируем наличие колонок
for col in ['positive_votes', 'negative_votes', 'important_votes']:
    if col not in bull_df: bull_df[col] = 0
    if col not in bear_df: bear_df[col] = 0

bull_df['negative_votes'] = 0
bear_df['positive_votes'] = 0

all_news = pd.concat([bull_df, bear_df], ignore_index=True)

print(f"✓ Загружено: {len(candles_df)} свечей, {len(all_news)} новостей")

# Сохраняем датасеты в ClearML
task.upload_artifact(name="candles_dataset", artifact_object=candles_df)
task.upload_artifact(name="news_dataset", artifact_object=all_news)

# ==========================================
# 4. ЗАПУСК АНАЛИЗА
# ==========================================

print("\n🚀 Запуск исследовательского анализа...")

results, merged_data = research_analysis(
    candles_df,
    all_news,
    start_date=params["start_date"],
    end_date=params["end_date"],
    arch=params["architecture"],
    export_path="clearml_research_output.json"
)

# ==========================================
# 5. ЛОГИРОВАНИЕ ГИПОТЕЗ И МЕТРИК
# ==========================================

if results:
    hypotheses = results.get("hypotheses", {})
    statistics = results.get("statistics", {})
    volatility_model = results.get("volatility_model", {})

    # 1. Реактивное голосование
    rv = hypotheses.get("reactive_voting", {})
    if rv:
        logger.report_scalar("Hypothesis 1: Reactive Voting", "Correlation", rv.get("corr", 0), iteration=0)
        logger.report_scalar("Hypothesis 1: Reactive Voting", "P-Value", rv.get("p_value", 1), iteration=0)
        logger.report_scalar("Hypothesis 1: Reactive Voting", "T-Test P-Value", rv.get("ttest_p", 1), iteration=0)

    # 2. Предсказание волатильности
    vp = hypotheses.get("volatility_prediction", {})
    if vp:
        logger.report_scalar("Hypothesis 2: Volatility Prediction", "Correlation 1h", vp.get("corr_1h", 0), iteration=0)
        logger.report_scalar("Hypothesis 2: Volatility Prediction", "P-Value 1h", vp.get("p_1h", 1), iteration=0)

    # 3. Временные паттерны
    tp = hypotheses.get("temporal_patterns", {})
    if tp:
        logger.report_scalar("Hypothesis 3: Temporal Patterns", "Peak Hour (UTC)", tp.get("peak_hour", 0), iteration=0)
        logger.report_scalar("Hypothesis 3: Temporal Patterns", "Time-Vol Correlation", tp.get("time_vol_corr", 0), iteration=0)

    # 4. Сентимент-анализ (BERT)
    sa = hypotheses.get("sentiment_analysis", {})
    if sa and sa.get("ttest_p") is not None:
        logger.report_scalar("Hypothesis 4: Sentiment (BERT)", "P-Value (Pos vs Neg)", sa.get("ttest_p", 1), iteration=0)
        logger.report_scalar("Hypothesis 4: Sentiment (BERT)", "Mean Pos Move", sa.get("pos_mean", 0), iteration=0)
        logger.report_scalar("Hypothesis 4: Sentiment (BERT)", "Mean Neg Move", sa.get("neg_mean", 0), iteration=0)
        logger.report_scalar("Hypothesis 4: Sentiment (BERT)", "N Positive", sa.get("n_pos", 0), iteration=0)
        logger.report_scalar("Hypothesis 4: Sentiment (BERT)", "N Negative", sa.get("n_neg", 0), iteration=0)

    # 5. ML-модель волатильности
    if volatility_model and "r2_test" in volatility_model:
        logger.report_scalar("ML Model: Volatility", "R2 Test", volatility_model.get("r2_test", 0), iteration=0)
        logger.report_scalar("ML Model: Volatility", "MAE Test", volatility_model.get("mae", 0), iteration=0)

    # 6. Важность признаков (Top-5)
    fi = statistics.get("feature_importance", {})
    if fi:
        sorted_imp = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, val in sorted_imp:
            logger.report_scalar("Feature Importance (Volatility)", feat, val, iteration=0)

# ==========================================
# 6. СОХРАНЕНИЕ ARTEFACTS & ГРАФИКОВ
# ==========================================

# 1. Merged Dataset
if merged_data is not None:
    task.upload_artifact(name="merged_dataset", artifact_object=merged_data)

# 2. Results JSON
def convert_numpy(obj):
    """Рекурсивно конвертирует numpy-типы в стандартные Python для JSON"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

if results:
    results_clean = convert_numpy(results)
    task.upload_artifact(name="analysis_results_json", artifact_object=results_clean)

# 3. Графики
if merged_data is not None and results is not None:
    print("📊 Генерация графиков для ClearML...")
    plt.ioff()  # Отключаем интерактивный режим для сервера
    plot_research_results(merged_data, results)
    
    figs = [plt.figure(i) for i in plt.get_fignums()]
    for i, fig in enumerate(figs):
        logger.report_matplotlib_figure(
            title="research_plots",
            series=f"figure_{i+1}",
            figure=fig,
            iteration=0
        )
        plt.close(fig)  # Освобождаем память

# ==========================================
# 7. ФИНАЛ
# ==========================================

logger.report_text("Research Analysis V2 finished successfully")
print("\n✅ Анализ завершён")
print("📊 Результаты доступны в ClearML UI")

task.close()