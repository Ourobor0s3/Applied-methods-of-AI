"""
Единые настройки: пути к данным, эксперименты, ClearML, артефакты.
"""

from __future__ import annotations

from pathlib import Path

# --- Файлы данных (лежат в DATA_DIR) ---
DATA_FILE_BTC_1H = "btc_1h_data_2018_to_2025.csv"
DATA_FILE_BTC_1D = "btc_1d_data_2018_to_2025.csv"
DATA_FILE_BTC_4H = "btc_4h_data_2018_to_2025.csv"
DATA_FILE_BTC_15M = "btc_15m_data.csv"
DATA_FILE_BEAR = "bear.csv"
DATA_FILE_BULL = "bull.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_BTC_PATH = DATA_DIR / DATA_FILE_BTC_1H
DATA_BTC_DAY_PATH = DATA_DIR / DATA_FILE_BTC_15M
DATA_BEAR_PATH = DATA_DIR / DATA_FILE_BEAR
DATA_BULL_PATH = DATA_DIR / DATA_FILE_BULL

# --- ClearML ---
CLEARML_PROJECT_NAME = "CryptoForecast"
CLEARML_TASK_NAME_PRICE = "price_comparison_v3"
CLEARML_TASK_NAME_VOLATILITY = "volatility_comparison_v3"
CLEARML_REUSE_LAST_TASK_ID = False  # передать в create_logger(..., reuse_last_task_id=...)

# --- Даты и окна ---
DATA_START_DATE = "2025-01-16"
DATA_END_DATE = "2025-12-16"
FORECAST_HORIZON = 3
VOLATILITY_WINDOW = 5

# --- NLP ---
USE_NLP = True
NLP_SENTENCE_MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# --- Модели для сравнения (имена совпадают с models_factory.create_model) ---
MODELS_TO_TEST = ["liquid", "densenet", "resnet", "xgboost"]

# --- Обучение (общие имена для логов и конфигов) ---
SEQUENCE_LENGTH = 25
TIME_SERIES_CV_SPLITS = 3
TRAINING_EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 1e-4
HIDDEN_DIM_DEFAULT = 64
HIDDEN_DIM_LIQUID = 48
HUBER_LOSS_DELTA = 0.5
XGBOOST_FLATTEN_BATCH_SIZE = 1024

# --- Технические индикаторы (свечи) ---
ROLLING_VOLATILITY_SHORT = 5
VOLUME_MA_WINDOW = 5
RSI_WINDOW = 14
EMA_FAST = 12
EMA_SLOW = 26

# --- Артефакты (относительно текущей рабочей директории при запуске) ---
ARTIFACTS_DIR = "artifacts"
BEST_MODEL_SELECTION_METRIC_PRICE = "roc_auc"
BEST_MODEL_SELECTION_METRIC_VOLATILITY = "r2"
CLASSIFICATION_THRESHOLD = 0.5

# --- Имена для отчётов ClearML (единый словарь, чтобы не разъезжались строки) ---
CLEARML_CONFIG_RUN = "run"
CLEARML_CONFIG_DATA = "data"
CLEARML_CONFIG_TRAINING_PREFIX = "training"
CLEARML_TITLE_DATASET = "dataset"
CLEARML_TITLE_METRICS_EPOCH = "metrics_epoch"
CLEARML_TITLE_METRICS_FINAL = "metrics_final"
CLEARML_TITLE_MODEL = "model"
CLEARML_TITLE_COMPARISON = "model_comparison"
CLEARML_TITLE_FEATURE_IMPORTANCE = "feature_importance"


def clearml_base_parameters() -> dict:
    """Параметры задачи ClearML: одни и те же ключи для price и volatility, где возможно."""
    return {
        "models_to_test": MODELS_TO_TEST,
        "use_nlp": USE_NLP,
        "nlp_sentence_model_name": NLP_SENTENCE_MODEL_NAME,
        "data_start_date": DATA_START_DATE,
        "data_end_date": DATA_END_DATE,
        "forecast_horizon": FORECAST_HORIZON,
        "volatility_window": VOLATILITY_WINDOW,
        "sequence_length": SEQUENCE_LENGTH,
        "time_series_cv_splits": TIME_SERIES_CV_SPLITS,
        "training_epochs": TRAINING_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_dim_default": HIDDEN_DIM_DEFAULT,
        "hidden_dim_liquid": HIDDEN_DIM_LIQUID,
        "huber_loss_delta": HUBER_LOSS_DELTA,
        "artifacts_dir": ARTIFACTS_DIR,
    }


def training_config_dict(model_name: str) -> dict:
    """Конфиг одного прогона обучения (для connect_configuration)."""
    return {
        "model_name": model_name,
        "sequence_length": SEQUENCE_LENGTH,
        "training_epochs": TRAINING_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_dim_default": HIDDEN_DIM_DEFAULT,
        "hidden_dim_liquid": HIDDEN_DIM_LIQUID,
    }


def comparison_scalar_series(metric_key: str, model_name: str) -> str:
    """Единый формат series для сравнения моделей: metric_key.model_name."""
    return f"{metric_key}.{model_name}"
