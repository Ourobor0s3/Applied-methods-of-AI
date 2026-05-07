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
CLEARML_TASK_VERSION = "v5"
CLEARML_REUSE_LAST_TASK_ID = False

# --- Модели для сравнения и тип задачи (должны быть до функций) ---
# Изменить здесь:
#   TASK_TYPE: "regression" (MAE, RMSE, R2) или "classification" (ROC AUC, Accuracy, F1)
#   MODELS_TO_TEST: какие модели тестировать ["liquid", "densenet", "resnet", "xgboost"]
TASK_TYPE_PRICE = "classification"
TASK_TYPE_VOLATILITY = "regression"
MODELS_TO_TEST = ["densenet"]
CLASSIFICATION_THRESHOLD = 0.5

# Динамические имена задач (формируются из MODELS_TO_TEST + TASK_TYPE)
def _make_task_name(prefix: str, models: list, task_type: str, version: str) -> str:
    """Создает имя задачи: price_[models]_[task_type]_vX"""
    models_str = "_".join(sorted(models))  # liquid_densenet_resnet_xgboost
    return f"{prefix}_{models_str}_{task_type}_{version}"

CLEARML_TASK_NAME_PRICE = _make_task_name("price", MODELS_TO_TEST, TASK_TYPE_PRICE, CLEARML_TASK_VERSION)
CLEARML_TASK_NAME_VOLATILITY = _make_task_name("volatility", MODELS_TO_TEST, TASK_TYPE_VOLATILITY, CLEARML_TASK_VERSION)

# --- Даты и окна ---
DATA_START_DATE = "2025-01-16"
DATA_END_DATE = "2025-12-16"
FORECAST_HORIZON = 3
VOLATILITY_WINDOW = 5

# --- NLP ---
USE_NLP = True
USE_NEWS_FILTER = True

# Для Ollama: "qwen3-embedding:0.6b" или "mxbai-embed-large"
# Для установки olama нужно запустить скрипты (для макича):
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull qwen3-embedding:0.6b
# Для sentence-transformers: "Qwen/Qwen3-Embedding-0.5B" или "paraphrase-MiniLM-L6-v2"
NLP_SENTENCE_MODEL_NAME = "qwen3-embedding:0.6b" # paraphrase-MiniLM-L6-v2
# Максимальная размерность эмбеддингов после PCA (для избежания переполнения памяти в XGBoost)
NLP_MAX_EMBEDDING_DIM = 64

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

# --- Стабильность XGBoost/OpenMP (кроссплатформенно) ---
# Можно менять под ОС/окружение без правок в коде моделей.
# macOS (особенно Apple Silicon, если были segfault): держите 1 поток.
# Windows: обычно можно повысить до 2-4 (или до числа физических ядер), если стабильно.
XGBOOST_N_JOBS = 1
XGBOOST_NTHREAD = 1
# macOS: обычно "1" для стабильности OpenMP/libomp.
# Windows: можно пробовать "2"/"4" и выше при стабильной работе.
OMP_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"

# --- Технические индикаторы (свечи) ---
ROLLING_VOLATILITY_SHORT = 5
VOLUME_MA_WINDOW = 5
RSI_WINDOW = 14
EMA_FAST = 12
EMA_SLOW = 26

# --- Артефакты (относительно текущей рабочей директории при запуске) ---
ARTIFACTS_DIR = "artifacts"

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
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "use_nlp": USE_NLP,
        "use_news_filter": USE_NEWS_FILTER,
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
        "xgboost_n_jobs": XGBOOST_N_JOBS,
        "xgboost_nthread": XGBOOST_NTHREAD,
        "omp_num_threads": OMP_NUM_THREADS,
        "openblas_num_threads": OPENBLAS_NUM_THREADS,
        "mkl_num_threads": MKL_NUM_THREADS,
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
