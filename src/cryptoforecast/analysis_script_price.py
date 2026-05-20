"""
================================================================================
  Скрипт обучения: прогнозирование направления цены BTC (классификация)
================================================================================
  Задача: бинарная классификация — растёт ли цена через horizon периодов?
  Таргет: 1 если future_close > close, иначе 0

  Данные: OHLCV 15-минутные свечи + новости bull/bear
  Период: DATA_START_DATE .. DATA_END_DATE (experiment_config.py)
  Модели: liquid, densenet, resnet, xgboost

  Конфигурация запуска: MODELS_TO_TEST, TASK_TYPE_PRICE, TRAINING_EPOCHS и пр.
  Все параметры — в experiment_config.py (единый источник правды).
================================================================================
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

from clearml_logger import create_logger
from experiment_config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CLEARML_CONFIG_DATA,
    CLEARML_CONFIG_RUN,
    CLEARML_CONFIG_TRAINING_PREFIX,
    CLEARML_PROJECT_NAME,
    CLEARML_REUSE_LAST_TASK_ID,
    CLEARML_TASK_NAME_PRICE,
    CLEARML_TITLE_COMPARISON,
    CLEARML_TITLE_DATASET,
    CLEARML_TITLE_METRICS_EPOCH,
    CLEARML_TITLE_METRICS_FINAL,
    CLEARML_TITLE_MODEL,
    CLEARML_TITLE_TRAINING_LOSS,
    DATA_BEAR_PATH,
    DATA_BTC_DAY_PATH,
    DATA_BULL_PATH,
    DATA_END_DATE,
    DATA_START_DATE,
    EMA_FAST,
    EMA_SLOW,
    FORECAST_HORIZON,
    HIDDEN_DIM_DEFAULT,
    HIDDEN_DIM_LIQUID,
    CLASSIFICATION_THRESHOLD,
    ROLLING_VOLATILITY_SHORT,
    TASK_TYPE_PRICE,
    LEARNING_RATE,
    MODELS_TO_TEST,
    NLP_SENTENCE_MODEL_NAME,
    RSI_WINDOW,
    SEQUENCE_LENGTH,
    TIME_SERIES_CV_SPLITS,
    TRAINING_EPOCHS,
    USE_NLP,
    USE_NEWS_FILTER,
    NEWS_FILTER_MIN_TOTAL_VOTES,
    NEWS_FILTER_MIN_POSITIVE_VOTES,
    NEWS_FILTER_MIN_REACTION_INTENSITY,
    NEWS_FILTER_MAX_TITLE_LENGTH,
    NEWS_FILTER_MIN_TITLE_LENGTH,
    NEWS_FILTER_SPAM_KEYWORDS,
    NEWS_FILTER_QUALITY_THRESHOLD,
    VOLATILITY_WINDOW,
    VOLUME_MA_WINDOW,
    WEIGHT_DECAY,
    XGBOOST_FLATTEN_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_DELTA,
    GRADIENT_CLIP_VALUE,
    clearml_base_parameters,
    comparison_scalar_series,
    training_config_dict,
)
from models_factory import create_model
from nlp_features import NewsTitleEncoder, aggregate_news_embeddings_with_votes
from news_filters import aggregate_news_weighted, filter_news
from utils import (
    should_stop_early,
    plot_training_losses,
    plot_prediction_scatter,
    plot_model_comparison,
    plot_prediction_distribution,
    plot_roc_curve,
    plot_confusion_matrix,
    calculate_metrics,
    get_primary_metric_value,
    get_metrics_config,
)

warnings.filterwarnings("ignore")

logger = create_logger(
    project_name=CLEARML_PROJECT_NAME,
    task_name=CLEARML_TASK_NAME_PRICE,
    task_type="training",
    reuse_last_task_id=CLEARML_REUSE_LAST_TASK_ID,
    **clearml_base_parameters(),
)


# =============================================================================
#  Загрузка и подготовка данных
# =============================================================================

def load_and_prepare_data(candles_path, bull_path, bear_path,
                            start_date=DATA_START_DATE, end_date=DATA_END_DATE,
                            forecast_horizon=FORECAST_HORIZON,
                            use_nlp=USE_NLP,
                            use_news_filter=USE_NEWS_FILTER):
    """Загружает свечи и новости, создаёт признаки и таргет.

    Признаки (ценовые):
      - returns: дневная доходность
      - volatility: скользящее стандартное отклонение доходности (ROLLING_VOLATILITY_SHORT)
      - volume_ratio: объём / скользящее среднее объёма
      - price_range: (High - Low) / Close
      - macd: EMA(12) - EMA(26)
      - rsi: Relative Strength Index

    Признаки (новостные):
      - sentiment_sum / sentiment_mean / sentiment_std: взвешенный сентимент по голосам
      - news_count: количество новостей за день

    Таргет: 1 если Close через forecast_horizon выше текущего, иначе 0.
    """
    logger.connect_configuration(
        {"start_date": start_date, "end_date": end_date,
         "forecast_horizon": forecast_horizon, "use_nlp": use_nlp,
         "nlp_sentence_model_name": NLP_SENTENCE_MODEL_NAME,
         "use_news_filter": use_news_filter},
        name=CLEARML_CONFIG_DATA,
    )

    # Загрузка данных
    candles = pd.read_csv(candles_path, parse_dates=["Open time"])
    bull = pd.read_csv(bull_path, parse_dates=["datetime"])
    bear = pd.read_csv(bear_path, parse_dates=["datetime"])

    # Bulls: нет negative_votes, Bears: нет positive_votes
    for news_df in (bull, bear):
        for col in ("positive_votes", "negative_votes", "important_votes"):
            if col not in news_df.columns:
                news_df[col] = 0
    bull["negative_votes"] = 0
    bear["positive_votes"] = 0
    all_news = pd.concat([bull, bear], ignore_index=True)

    if use_news_filter:
        all_news, filter_stats = filter_news(
            all_news,
            use_filter=True,
            min_total_votes=NEWS_FILTER_MIN_TOTAL_VOTES,
            min_positive_votes=NEWS_FILTER_MIN_POSITIVE_VOTES,
            min_reaction_intensity=NEWS_FILTER_MIN_REACTION_INTENSITY,
            min_title_length=NEWS_FILTER_MIN_TITLE_LENGTH,
            max_title_length=NEWS_FILTER_MAX_TITLE_LENGTH,
            spam_keywords=NEWS_FILTER_SPAM_KEYWORDS,
            min_quality=NEWS_FILTER_QUALITY_THRESHOLD,
        )
        logger.report_scalar("News Filtering", "initial_count", float(filter_stats["total_initial"]), 0)
        logger.report_scalar("News Filtering", "remaining_count", float(filter_stats["total_remaining"]), 0)
        logger.report_scalar("News Filtering", "credibility_dropped", float(filter_stats["credibility_dropped"]), 0)
        logger.report_scalar("News Filtering", "relevance_dropped", float(filter_stats["relevance_dropped"]), 0)
        logger.report_scalar("News Filtering", "quality_dropped", float(filter_stats["quality_dropped"]), 0)
        logger.report_scalar("News Filtering", "total_dropped", float(filter_stats["total_dropped"]), 0)
        print(f"[Filter] Initial: {filter_stats['total_initial']} | "
              f"Remaining: {filter_stats['total_remaining']} | "
              f"Dropped: {filter_stats['total_dropped']} "
              f"(cred={filter_stats['credibility_dropped']}, "
              f"rel={filter_stats['relevance_dropped']}, "
              f"qual={filter_stats['quality_dropped']})")

    # Фильтрация по дате
    candles = candles[(candles["Open time"] >= start_date) & (candles["Open time"] <= end_date)].copy()
    all_news = all_news[(all_news["datetime"] >= start_date) & (all_news["datetime"] <= end_date)].copy()

    # Ценовые признаки
    candles["returns"] = candles["Close"].pct_change()
    candles["volatility"] = candles["returns"].rolling(ROLLING_VOLATILITY_SHORT).std()
    candles["price_range"] = (candles["High"] - candles["Low"]) / candles["Close"]
    candles["volume_ma"] = candles["Volume"].rolling(VOLUME_MA_WINDOW).mean()
    candles["volume_ratio"] = candles["Volume"] / (candles["volume_ma"] + 1e-8)
    candles["ema_12"] = candles["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    candles["ema_26"] = candles["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    candles["macd"] = candles["ema_12"] - candles["ema_26"]

    # RSI
    delta = candles["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_WINDOW).mean()
    candles["rsi"] = 100 - (100 / (1 + loss.replace(0, np.nan)))

    # Сентимент новостей: weighted sum по типам голосов
    all_news["sentiment_score"] = (
        all_news["positive_votes"] * 1.0
        + all_news["important_votes"] * 0.5
        - all_news["negative_votes"] * 0.8
    )

    # NLP-эмбеддинги (если включено)
    nlp_cols: list[str] = []
    if use_nlp:
        encoder = NewsTitleEncoder(model_name=NLP_SENTENCE_MODEL_NAME)
        news_with_emb = aggregate_news_embeddings_with_votes(
            all_news, encoder)
        nlp_cols = [c for c in news_with_emb.columns if c != "date"]
        news_daily = aggregate_news_weighted(
            all_news, date_col="datetime", sentiment_col="sentiment_score")
        for col in nlp_cols:
            news_daily[col] = news_with_emb.set_index("date")[col].reindex(news_daily["date"]).values
    else:
        news_daily = aggregate_news_weighted(
            all_news, date_col="datetime", sentiment_col="sentiment_score")

    # Объединение свечей с новостями
    candles["date"] = pd.to_datetime(candles["Open time"].dt.date)
    df = candles.merge(news_daily, on="date", how="left")

    # Заполняем пропуски нулями (дни без новостей)
    news_cols = [
        "sentiment_sum", "sentiment_mean", "sentiment_std", "news_count",
        "weighted_sentiment", "max_quality_sentiment",
        "total_votes_sum", "total_votes_mean", "total_votes_max",
        "reaction_sum", "reaction_mean", "reaction_max",
        "consensus_mean", "consensus_max",
        "quality_mean", "quality_std", "quality_weighted_count",
    ]
    for col in news_cols + nlp_cols:
        df[col] = df[col].fillna(0)

    # Таргет: направление цены через forecast_horizon свечей
    df["future_close"] = df["Close"].shift(-forecast_horizon)
    df["returns_future"] = (df["future_close"] - df["Close"]) / df["Close"]

    if TASK_TYPE_PRICE == "classification":
        df["target"] = (df["future_close"] > df["Close"]).astype(int)
    else:
        df["target"] = df["returns_future"].clip(-1, 1)

    # Отбрасываем строки с NaN в признаках или без future_close
    feature_cols = ["returns", "volatility", "volume_ratio", "price_range", "macd", "rsi"]
    df = df.dropna(subset=feature_cols + news_cols + nlp_cols + ["future_close"]).reset_index(drop=True)

    # Логируем статистику датасета в ClearML
    if TASK_TYPE_PRICE == "classification":
        positive_ratio_pct = float(df["target"].mean() * 100)
        logger.report_scalar(CLEARML_TITLE_DATASET, "sample_count", float(len(df)), 0)
        logger.report_scalar(CLEARML_TITLE_DATASET, "positive_class_ratio_pct", positive_ratio_pct, 0)
    else:
        logger.report_scalar(CLEARML_TITLE_DATASET, "sample_count", float(len(df)), 0)
        logger.report_scalar(CLEARML_TITLE_DATASET, "target_mean", float(df["target"].mean()), 0)
        logger.report_scalar(CLEARML_TITLE_DATASET, "target_std", float(df["target"].std()), 0)

    logger.report_scalar(CLEARML_TITLE_DATASET, "nlp_feature_count", float(len(nlp_cols)), 0)

    return df, feature_cols, news_cols, nlp_cols


class PriceSequenceDataset(Dataset):
    """PyTorch Dataset для последовательностей фиксированной длины (sequence_length).
    Каждый элемент: (ценовые_признаки[seq_len, n_feat],
                     новостные_признаки[seq_len, n_news],
                     таргет_на_последнем_шаге)

    RobustScaler fit на первых 80% train-части — валидация не должна "видеть"
    тестовые данные через скейлеры (предотвращает leakage).
    """

    def __init__(self, df: pd.DataFrame, sequence_length: int,
                 target_col: str = "target",
                 feature_cols: list = None, news_cols: list = None, nlp_cols: list = None):
        self.sequence_length = sequence_length
        feature_cols = feature_cols or ["returns", "volatility", "volume_ratio",
                                         "price_range", "macd", "rsi"]
        news_cols = news_cols or ["sentiment_sum", "sentiment_mean",
                                    "sentiment_std", "news_count"]
        nlp_cols = nlp_cols or []

        self.scaler_price = RobustScaler()
        self.scaler_news = RobustScaler()

        price_data = df[feature_cols].values
        news_block = df[news_cols + nlp_cols].values if nlp_cols else df[news_cols].values

        # Fit скейлеров только на train-части (первые 80%)
        split_idx = max(1, int(len(df) * 0.8))
        self.scaler_price.fit(price_data[:split_idx])
        self.scaler_news.fit(news_block[:split_idx])

        self.price_data = torch.tensor(self.scaler_price.transform(price_data),
                                       dtype=torch.float32)
        self.news_data = torch.tensor(self.scaler_news.transform(news_block),
                                      dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.price_data) - self.sequence_length)

    def __getitem__(self, idx: int):
        end = idx + self.sequence_length
        return self.price_data[idx:end], self.news_data[idx:end], self.targets[end - 1]


def _flatten(ds):
    """Превращает PyTorch Dataset в плоские массивы (X, y) для XGBoost."""
    loader = DataLoader(ds, batch_size=XGBOOST_FLATTEN_BATCH_SIZE, shuffle=False)
    xs, ys = [], []
    for p_b, n_b, t_b in loader:
        xs.append(np.hstack([
            p_b.numpy().reshape(p_b.shape[0], -1),
            n_b.numpy().reshape(n_b.shape[0], -1)
        ]))
        ys.append(t_b.numpy())
    return np.vstack(xs), np.concatenate(ys)


def train_one_model(df: pd.DataFrame, model_name: str,
                    feature_cols: list, news_cols: list, nlp_cols: list,
                    *, training_epochs: int, batch_size: int,
                    learning_rate: float, sequence_length: int):
    """Обучает одну модель на данных df.

    Для нейросетей (liquid, resnet, densenet):
      - AdamW + CosineAnnealingLR
      - BCEWithLogitsLoss (классификация) или MSELoss (регрессия)
      - Gradient clipping + early stopping
      - Логирует loss и метрики в ClearML каждую эпоху

    Для XGBoost:
      - Flatten данных (последовательность -> вектор)
      - Early stopping через eval_set
      - Возвращает метрики сразу (без цикла эпох)
    """
    logger.connect_configuration(
        training_config_dict(model_name),
        name=f"{CLEARML_CONFIG_TRAINING_PREFIX}_{model_name}"
    )

    # TimeSeriesSplit: используем последний сплит как train/test
    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    train_idx, test_idx = list(tscv.split(df))[-1]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_ds = PriceSequenceDataset(train_df, sequence_length,
                                     feature_cols=feature_cols,
                                     news_cols=news_cols, nlp_cols=nlp_cols)
    test_ds = PriceSequenceDataset(test_df, sequence_length,
                                    feature_cols=feature_cols,
                                    news_cols=news_cols, nlp_cols=nlp_cols)

    # XGBoost: flatten + обучение
    if model_name == "xgboost":
        x_train, y_train = _flatten(train_ds)
        x_val, y_val = _flatten(test_ds)
        model = create_model("xgboost", num_features=x_train.shape[1], news_dim=0)
        model.fit(x_train, y_train, x_val, y_val,
                  feature_names=[f"f{i}" for i in range(x_train.shape[1])])
        probs = np.clip(model.predict(x_val), 0, 1) \
            if TASK_TYPE_PRICE == "classification" else model.predict(x_val)
        metrics = calculate_metrics(y_val, probs, TASK_TYPE_PRICE)
        print(f"price | {model_name} | ROC={metrics['roc_auc']:.4f} ACC={metrics['accuracy']:.4f}")
        return model, probs, y_val, metrics, train_ds, [], []

    # Нейросети: PyTorch training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = HIDDEN_DIM_LIQUID if model_name == "liquid" else HIDDEN_DIM_DEFAULT
    model = create_model(
        model_name, num_features=len(feature_cols),
        news_dim=len(news_cols) + len(nlp_cols), hidden_dim=hidden
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    enable_clearml = logger._ENABLE_CLEARML if hasattr(logger, "_ENABLE_CLEARML") else True

    # Функция потерь зависит от типа задачи
    loss_fn = nn.BCEWithLogitsLoss() if TASK_TYPE_PRICE == "classification" else nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=training_epochs)

    train_losses, val_losses, best_val_loss, best_model_state = [], [], float("inf"), None

    # Логируем число параметров модели
    logger.report_scalar(CLEARML_TITLE_MODEL, f"{model_name}.param_count",
                        sum(p.numel() for p in model.parameters()), 0)

    for epoch in range(training_epochs):
        # --- Train ---
        model.train()
        train_loss_list = []
        for p_b, n_b, t_b in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(p_b.to(device), n_b.to(device)).squeeze(), t_b.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            opt.step()
            train_loss_list.append(loss.item())

        sched.step()

        # --- Validation ---
        model.eval()
        loss_list, probs_list, y_list = [], [], []

        with torch.no_grad():
            for p_b, n_b, t_b in test_loader:
                out = model(p_b.to(device), n_b.to(device)).squeeze()
                loss_list.append(loss_fn(out, t_b.to(device)).item())
                probs_list.extend(
                    torch.sigmoid(out).cpu().numpy()
                    if TASK_TYPE_PRICE == "classification" else out.cpu().numpy()
                )
                y_list.extend(t_b.numpy())

        t_loss, v_loss = np.mean(train_loss_list), np.mean(loss_list)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Логируем loss в ClearML
        if enable_clearml:
            logger.report_scalar(CLEARML_TITLE_TRAINING_LOSS,
                                 f"{model_name}.train_loss", t_loss, epoch + 1)
            logger.report_scalar(CLEARML_TITLE_TRAINING_LOSS,
                                 f"{model_name}.val_loss", v_loss, epoch + 1)

        # Проверка ранней остановки
        should_stop, _ = should_stop_early(train_losses, val_losses,
                                           EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA)
        if should_stop:
            print(f"Ранняя остановка на эпохе {epoch + 1}")
            if best_model_state:
                model.load_state_dict(best_model_state)
            break

        # Запоминаем лучшее состояние модели
        if v_loss < best_val_loss:
            best_val_loss, best_model_state = v_loss, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Печать метрик (для отслеживания в консоли)
        metrics = calculate_metrics(np.asarray(y_list), np.asarray(probs_list), TASK_TYPE_PRICE)
        print(f"price | {model_name} | ep={epoch+1:3d} | "
              f"loss={t_loss:.4f}/{v_loss:.4f} | "
              f"ROC={metrics['roc_auc']:.4f} ACC={metrics['accuracy']:.4f}")

    # Сохраняем лучшее состояние и массивы потерь
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, f"{ARTIFACTS_DIR}/{model_name}_best_final.pt")

    np.save(f"{ARTIFACTS_DIR}/{model_name}_train_losses.npy", train_losses)
    np.save(f"{ARTIFACTS_DIR}/{model_name}_val_losses.npy", val_losses)

    # Графики обучения — только если ClearML выключен
    # (если включен — графики загружаются после цикла в main)
    if not enable_clearml:
        plot_training_losses(train_losses, val_losses, model_name,
                             f"{ARTIFACTS_DIR}/{model_name}_training_losses.png")

    return model, np.asarray(probs_list), np.asarray(y_list), metrics, \
        train_ds, train_losses, val_losses


# =============================================================================
#  Сохранение лучшей модели
# =============================================================================

def save_best_model_bundle(best: dict, feature_cols: list, news_cols: list, nlp_cols: list,
                            output_dir: str = ARTIFACTS_DIR):
    """Сохраняет артефакты лучшей модели: скейлеры, веса, предсказания, метаданные.
    Все файлы загружаются в ClearML как артефакты."""
    os.makedirs(output_dir, exist_ok=True)
    name = best["model_name"]
    model = best["model"]
    train_ds: PriceSequenceDataset = best["train_ds"]

    # Скейлеры для обратного преобразования при инференсе
    scalers_path = os.path.join(output_dir, "best_price_scalers.joblib")
    joblib.dump({"price_scaler": train_ds.scaler_price,
                 "news_scaler": train_ds.scaler_news}, scalers_path)

    # Модель: XGBoost -> joblib, PyTorch -> state_dict
    if name == "xgboost":
        model_path = os.path.join(output_dir, "best_price_xgboost_model.joblib")
        joblib.dump(model.model, model_path)
    else:
        model_path = os.path.join(output_dir, "best_price_torch_model.pt")
        torch.save(model.state_dict(), model_path)

    # CSV с предсказаниями и реальными значениями
    probs = np.asarray(best["preds"])
    targets = np.asarray(best["targets"])
    preds_path = os.path.join(output_dir, "best_price_predictions.csv")
    pd.DataFrame({"target": targets, "predicted_probability": probs}).to_csv(preds_path, index=False)

    # Метаданные: конфиг, метрики, пути к файлам
    meta = {
        "task": f"price_direction_{TASK_TYPE_PRICE}",
        "task_type": TASK_TYPE_PRICE,
        "best_model_name": name,
        "selection_metric": get_metrics_config(TASK_TYPE_PRICE)["primary_metric"],
        "best_metrics": best["metrics"],
        "feature_columns": feature_cols,
        "news_columns": news_cols,
        "nlp_columns": nlp_cols,
        "sequence_length": int(best["sequence_length"]),
        "model_path": model_path,
        "scalers_path": scalers_path,
        "predictions_path": preds_path,
    }
    meta_path = os.path.join(output_dir, "best_price_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.upload_artifact("best_price_model_meta", meta_path)
    logger.upload_artifact("best_price_scalers", scalers_path)
    logger.upload_artifact("best_price_model", model_path)
    logger.upload_artifact("best_price_predictions", preds_path)
    logger.report_text(f"Saved reusable bundle: {meta_path}", iteration=0)
    print(f"Saved best price model: {model_path}\nMetadata: {meta_path}")
    return meta_path


if __name__ == "__main__":
    # Параметры из experiment_config.py (или переопределённые через ClearML)
    models_to_test = logger.get_parameter("models_to_test", MODELS_TO_TEST)
    logger.connect_configuration({"models_to_test": models_to_test}, name=CLEARML_CONFIG_RUN)
    print(f"ClearML enabled={logger.is_enabled} | models_to_test={models_to_test}")

    # Загрузка и подготовка данных
    df, feat_cols, news_cols, nlp_cols = load_and_prepare_data(
        DATA_BTC_DAY_PATH, DATA_BULL_PATH, DATA_BEAR_PATH,
        start_date=logger.get_parameter("data_start_date", DATA_START_DATE),
        end_date=logger.get_parameter("data_end_date", DATA_END_DATE),
        forecast_horizon=logger.get_parameter("forecast_horizon", FORECAST_HORIZON),
        use_nlp=logger.get_parameter("use_nlp", USE_NLP),
        use_news_filter=logger.get_parameter("use_news_filter", USE_NEWS_FILTER),
    )

    epochs = logger.get_parameter("training_epochs", TRAINING_EPOCHS)
    batch = logger.get_parameter("batch_size", BATCH_SIZE)
    lr = logger.get_parameter("learning_rate", LEARNING_RATE)
    seq_len = logger.get_parameter("sequence_length", SEQUENCE_LENGTH)

    results: dict = {}
    best = None

    # Обучение всех моделей по очереди
    for model_name in models_to_test:
        print(f"\nTraining: {model_name}")
        result = train_one_model(
            df, model_name, feat_cols, news_cols, nlp_cols,
            training_epochs=epochs, batch_size=batch,
            learning_rate=lr, sequence_length=seq_len,
        )

        # Распаковка: torch-модели возвращают 7 элементов, XGBoost — 5
        if len(result) == 7:
            model, probs, targets, metrics, train_ds, train_losses, val_losses = result
            results[model_name] = {
                "metrics": metrics, "preds": probs, "targets": targets,
                "train_losses": train_losses, "val_losses": val_losses
            }
        else:
            model, probs, targets, metrics, train_ds = result
            results[model_name] = {"metrics": metrics, "preds": probs, "targets": targets}

        # Логируем все метрики в ClearML
        for metric_name, value in metrics.items():
            logger.report_scalar(
                CLEARML_TITLE_COMPARISON,
                comparison_scalar_series(metric_name, model_name),
                value, 0,
            )

        # Выбираем лучшую модель по основной метрике (ROC AUC)
        current_value = get_primary_metric_value(metrics, TASK_TYPE_PRICE)
        best_value = get_primary_metric_value(best["metrics"], TASK_TYPE_PRICE) if best else float("-inf")

        if best is None or current_value > best_value:
            best_result = result
            if len(best_result) == 7:
                _, _, _, _, _, best_train_losses, best_val_losses = best_result
            else:
                best_train_losses, best_val_losses = [], []
            best = {
                "model_name": model_name, "model": model, "metrics": metrics,
                "preds": probs, "y_score": probs, "targets": targets,
                "train_losses": best_train_losses, "val_losses": best_val_losses,
                "sequence_length": seq_len, "train_ds": train_ds,
            }

    # Таблица сравнения моделей
    primary_metric = get_metrics_config(TASK_TYPE_PRICE)["primary_metric"]
    comparison_df = (
        pd.DataFrame([results[m]["metrics"] for m in results])
        .sort_values(primary_metric, ascending=False)
        .reset_index(drop=True)
    )
    comparison_df.insert(0, "model_name", list(results.keys()))

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    comparison_path = os.path.join(ARTIFACTS_DIR, "price_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    logger.report_table("model_comparison_table", "summary", comparison_df.round(4), 0)
    logger.upload_artifact("price_model_comparison_csv", comparison_path)

    print(f"\nModel comparison (sorted by {primary_metric}):")
    print(comparison_df.to_string(index=False))

    best_metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in best["metrics"].items()])
    print(f"\nBest model: {best['model_name']} | {best_metrics_str}")

    # Сохранение артефактов лучшей модели
    save_best_model_bundle(best, feat_cols, news_cols, nlp_cols)

    # =============================================================================
    #  Построение и загрузка графиков в ClearML
    # =============================================================================
    enable_clearml = logger._ENABLE_CLEARML if hasattr(logger, "_ENABLE_CLEARML") else True

    # Кривые обучения для всех нейросетевых моделей
    for model_name in results:
        if results[model_name].get("train_losses"):
            tl = results[model_name]["train_losses"]
            vl = results[model_name]["val_losses"]
            plot_training_losses(tl, vl, model_name,
                                 save_path=f"{ARTIFACTS_DIR}/{model_name}_training_losses.png",
                                 logger=logger if enable_clearml else None,
                                 title_suffix="Price")

    # Сравнение моделей (bar chart)
    plot_model_comparison(results, primary_metric, TASK_TYPE_PRICE,
                          save_path=f"{ARTIFACTS_DIR}/price_model_comparison.png",
                          logger=logger if enable_clearml else None,
                          title_suffix="Price")

    # Предсказания лучшей модели: scatter + residuals
    plot_prediction_scatter(
        best["targets"], best["preds"], best["model_name"],
        task_type=TASK_TYPE_PRICE,
        save_path=f"{ARTIFACTS_DIR}/best_price_prediction_scatter.png",
        logger=logger if enable_clearml else None,
        title_suffix="Price",
    )

    # Распределение вероятностей предсказаний
    plot_prediction_distribution(
        best["targets"], best["preds"], best["model_name"],
        task_type=TASK_TYPE_PRICE,
        save_path=f"{ARTIFACTS_DIR}/best_price_prediction_distribution.png",
        logger=logger if enable_clearml else None,
        title_suffix="Price",
    )

    # ROC-кривая
    y_score = best.get("y_score", best["preds"])
    plot_roc_curve(
        best["targets"], y_score, best["model_name"],
        save_path=f"{ARTIFACTS_DIR}/best_price_roc_curve.png",
        logger=logger if enable_clearml else None,
        title_suffix="Price",
    )

    # Матрица ошибок
    plot_confusion_matrix(
        best["targets"], best["preds"], best["model_name"],
        save_path=f"{ARTIFACTS_DIR}/best_price_confusion_matrix.png",
        logger=logger if enable_clearml else None,
        title_suffix="Price",
    )

    logger.mark_completed()
    logger.close()
