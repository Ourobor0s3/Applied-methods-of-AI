"""
================================================================================
  Скрипт обучения: прогнозирование волатильности BTC (регрессия)
================================================================================
  Задача: регрессия — предсказать log-волатильность через horizon периодов.
  Таргет: log1p(|future_volatility| * 100), где future_volatility — std доходностей

  Данные: OHLCV 15-минутные свечи + новости bull/bear
  Период: DATA_START_DATE .. DATA_END_DATE (experiment_config.py)
  Модели: liquid, densenet, resnet, xgboost

  Конфигурация запуска: MODELS_TO_TEST, TASK_TYPE_VOLATILITY, TRAINING_EPOCHS и пр.
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
    CLEARML_TASK_NAME_VOLATILITY,
    CLEARML_TITLE_COMPARISON,
    CLEARML_TITLE_DATASET,
    CLEARML_TITLE_FEATURE_IMPORTANCE,
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
    HUBER_LOSS_DELTA,
    TASK_TYPE_VOLATILITY,
    CLASSIFICATION_THRESHOLD,
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
    plot_feature_importance,
    plot_prediction_distribution,
    calculate_metrics,
    get_primary_metric_value,
    get_metrics_config,
)

warnings.filterwarnings("ignore")

logger = create_logger(
    project_name=CLEARML_PROJECT_NAME,
    task_name=CLEARML_TASK_NAME_VOLATILITY,
    task_type="training",
    reuse_last_task_id=CLEARML_REUSE_LAST_TASK_ID,
    **clearml_base_parameters(),
)

TARGET_COLUMN = "target_volatility_log"


# =============================================================================
#  Обучение нейросетевой модели с early stopping
# =============================================================================

def train_model_with_early_stopping(model, train_loader, test_loader, model_name,
                                     device, training_epochs, logger,
                                     enable_clearml=True, train_dataset=None):
    """Цикл обучения PyTorch-модели с логированием в ClearML.

    - AdamW optimizer + CosineAnnealingLR scheduler
    - HuberLoss (устойчив к выбросам в данных волатильности)
    - Gradient clipping для стабильности
    - Ранняя остановка при отсутствии улучшения val_loss
    - Логирование train/val loss в ClearML каждую эпоху
    - Сохранение лучшего состояния модели (best_model_state)
    """
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=training_epochs)
    loss_fn = nn.HuberLoss(delta=HUBER_LOSS_DELTA)

    train_losses, val_losses, best_val_loss, best_model_state = [], [], float("inf"), None

    # Число параметров — в ClearML для сравнения сложности моделей
    logger.report_scalar(CLEARML_TITLE_MODEL, f"{model_name}.param_count",
                        sum(p.numel() for p in model.parameters()), 0)

    for epoch in range(training_epochs):
        # --- Train ---
        model.train()
        train_loss_list = []
        for p_b, n_b, t_sc, _ in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(p_b.to(device), n_b.to(device)).squeeze(), t_sc.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            opt.step()
            train_loss_list.append(loss.item())

        sched.step()

        # --- Validation ---
        model.eval()
        loss_list, preds_list, targets_list = [], [], []

        with torch.no_grad():
            for p_b, n_b, _, t_raw in test_loader:
                out = model(p_b.to(device), n_b.to(device)).squeeze()
                loss_list.append(loss_fn(out, t_raw.to(device)).item())
                preds_list.extend(out.cpu().numpy())
                targets_list.extend(t_raw.numpy())

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

        # Запоминаем лучшее состояние
        if v_loss < best_val_loss:
            best_val_loss, best_model_state = v_loss, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Печать метрик в консоль
        metrics = calculate_metrics(np.asarray(targets_list), np.asarray(preds_list),
                                     TASK_TYPE_VOLATILITY)
        print(f"vol | {model_name} | ep={epoch+1:3d} | "
              f"loss={t_loss:.4f}/{v_loss:.4f} | "
              f"R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")

    # Восстанавливаем лучшее состояние и сохраняем
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, f"{ARTIFACTS_DIR}/{model_name}_best_final.pt")

    np.save(f"{ARTIFACTS_DIR}/{model_name}_train_losses.npy", train_losses)
    np.save(f"{ARTIFACTS_DIR}/{model_name}_val_losses.npy", val_losses)

    # Графики обучения — только если ClearML выключен
    if not enable_clearml:
        plot_training_losses(train_losses, val_losses, model_name,
                             f"{ARTIFACTS_DIR}/{model_name}_training_losses.png")

    return model, np.asarray(preds_list), np.asarray(targets_list), \
        metrics, train_dataset, train_losses, val_losses


def load_and_prepare_data(candles_path, bull_path, bear_path,
                            start_date=DATA_START_DATE, end_date=DATA_END_DATE,
                            forecast_horizon=FORECAST_HORIZON,
                            volatility_window=VOLATILITY_WINDOW,
                            use_nlp=USE_NLP,
                            use_news_filter=USE_NEWS_FILTER):
    """Загружает свечи и новости, создаёт признаки и таргет.

    Признаки (ценовые):
      - returns: доходность (pct_change)
      - volatility: скользящее std доходности (volatility_window)
      - volume_ratio: объём / MA объёма
      - price_range: (High - Low) / Close
      - macd: EMA(12) - EMA(26)
      - rsi: Relative Strength Index

    Признаки (новостные, расширенные при USE_NLP=True):
      - sentiment_sum / sentiment_mean / sentiment_std
      - news_count
      - Расширенные голоса: sentiment_weighted, total_votes, positive/negative_ratio,
        reaction_intensity, consensus_score

    Таргет: log1p(|future_volatility| * 100), где future_volatility —
    std доходностей через forecast_horizon периодов.

    Данные агрегируются по дням для объединения с новостным сентиментом.
    Выбросы (>99.9% квантиль) отбрасываются.
    """
    logger.connect_configuration(
        {"start_date": start_date, "end_date": end_date,
         "forecast_horizon": forecast_horizon, "volatility_window": volatility_window,
         "use_nlp": use_nlp, "nlp_sentence_model_name": NLP_SENTENCE_MODEL_NAME,
         "use_news_filter": use_news_filter},
        name=CLEARML_CONFIG_DATA,
    )

    # Загрузка данных из CSV
    candles = pd.read_csv(candles_path, parse_dates=["Open time"])
    bull = pd.read_csv(bull_path, parse_dates=["datetime"])
    bear = pd.read_csv(bear_path, parse_dates=["datetime"])

    # Bulls: нет negative_votes, Bears: нет positive_votes — заполняем нулями
    for news_df in (bull, bear):
        for col in ("positive_votes", "negative_votes", "important_votes"):
            if col not in news_df.columns:
                news_df[col] = 0
    bull["negative_votes"] = 0
    bear["positive_votes"] = 0
    all_news = pd.concat([bull, bear], ignore_index=True)

    # Предварительная фильтрация новостей
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

    # Фильтрация по периоду данных
    candles = candles[
        (candles["Open time"] >= start_date) & (candles["Open time"] <= end_date)
    ].copy()
    all_news = all_news[
        (all_news["datetime"] >= start_date) & (all_news["datetime"] <= end_date)
    ].copy()

    # Ценовые признаки (технические индикаторы)
    candles["returns"] = candles["Close"].pct_change()
    candles["volatility"] = candles["returns"].rolling(volatility_window).std()
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

    # Сентимент: weighted sum по типам голосов
    all_news["sentiment_score"] = (
        all_news["positive_votes"] * 1.0
        + all_news["important_votes"] * 0.5
        - all_news["negative_votes"] * 0.8
    )

    # NLP-эмбеддинги заголовков (опционально)
    nlp_cols: list[str] = []
    if use_nlp:
        encoder = NewsTitleEncoder(model_name=NLP_SENTENCE_MODEL_NAME)
        news_with_emb = aggregate_news_embeddings_with_votes(
            all_news, encoder)
        news_daily = aggregate_news_weighted(all_news, date_col="datetime", sentiment_col="sentiment_score")
        news_daily = news_daily.merge(news_with_emb, on="date", how="left")
        nlp_cols = [c for c in news_daily.columns if "nlp_emb" in c]
    else:
        news_daily = aggregate_news_weighted(all_news, date_col="datetime", sentiment_col="sentiment_score")

    # Объединение свечей с новостями по дате
    candles["date"] = pd.to_datetime(candles["Open time"].dt.date)
    df = candles.merge(news_daily, on="date", how="left")

    # Список новостных колонок (расширенный при USE_NLP=False)
    expected_news_cols = [
        "sentiment_sum", "sentiment_mean", "sentiment_std", "news_count",
        "weighted_sentiment", "max_quality_sentiment",
        "total_votes_sum", "total_votes_mean", "total_votes_max",
        "reaction_sum", "reaction_mean", "reaction_max",
        "consensus_mean", "consensus_max",
        "quality_mean", "quality_std", "quality_weighted_count",
    ]
    news_cols = [col for col in expected_news_cols if col in df.columns]
    for col in news_cols + nlp_cols:
        df[col] = df[col].fillna(0)

    # Логируем статистику голосов в ClearML
    logger.report_scalar("User Votes Analysis", "Total News Items", len(all_news), 0)
    logger.report_scalar("User Votes Analysis", "Average Total Votes per News",
                         all_news["total_votes"].mean(), 0)
    logger.report_scalar("User Votes Analysis", "Average Reaction Intensity",
                         all_news["reaction_intensity"].mean(), 0)
    logger.report_scalar("User Votes Analysis", "Average Consensus Score",
                         all_news["consensus_score"].mean(), 0)

    # Таргет: будущая волатильность
    if "returns" not in df.columns:
        df["returns"] = df["Close"].pct_change()

    df["target_volatility"] = df["returns"].shift(-forecast_horizon).rolling(volatility_window).std()
    df[TARGET_COLUMN] = np.log1p(df["target_volatility"].abs() * 100)

    # Удаляем строки с NaN
    feature_cols = ["returns", "volatility", "volume_ratio", "price_range", "macd", "rsi"]
    df = df.dropna(subset=feature_cols + news_cols + nlp_cols + [TARGET_COLUMN]).reset_index(drop=True)

    # Отбрасываем выбросы (>99.9% квантиль)
    if len(df) > 100:
        df = df[df["target_volatility"] <= df["target_volatility"].quantile(0.999)]

    df = df[np.isfinite(df[TARGET_COLUMN])].reset_index(drop=True)

    # Логируем характеристики датасета
    logger.report_scalar(CLEARML_TITLE_DATASET, "sample_count", float(len(df)), 0)
    logger.report_scalar(CLEARML_TITLE_DATASET, "nlp_feature_count", float(len(nlp_cols)), 0)

    return df, feature_cols, news_cols, nlp_cols


class VolatilitySequenceDataset(Dataset):
    """PyTorch Dataset для последовательностей волатильности.

    Каждый элемент: (ценовые_признаки[seq_len, n_feat],
                     новостные_признаки[seq_len, n_news],
                     scaled_target[последний шаг],
                     raw_target[последний шаг])

    Три скейлера: price (ценовые), news (новостные), target (таргет).
    Все скейлеры fit на первых 80% train-части (утечка данных исключена).

    Особенность: храним и scaled, и raw target — для обучения на scaled,
    а для метрик используем raw (правильная оценка).
    """

    def __init__(self, df: pd.DataFrame, sequence_length: int,
                 target_col: str = TARGET_COLUMN,
                 feature_cols: list = None, news_cols: list = None, nlp_cols: list = None):
        self.sequence_length = sequence_length
        feature_cols = feature_cols or ["returns", "volatility", "volume_ratio",
                                         "price_range", "macd", "rsi"]
        news_cols = news_cols or [
            "sentiment_sum", "sentiment_mean", "sentiment_std", "news_count",
            "sentiment_weighted_sum", "sentiment_weighted_mean", "total_votes_mean",
            "positive_ratio_mean", "negative_ratio_mean",
            "reaction_intensity_mean", "consensus_score_mean"
        ]
        nlp_cols = nlp_cols or []

        self.scaler_price = RobustScaler()
        self.scaler_news = RobustScaler()
        self.scaler_target = RobustScaler()

        price_data = df[feature_cols].values
        news_block = df[news_cols + nlp_cols].values if nlp_cols else df[news_cols].values
        target_data = df[[target_col]].values

        split_idx = max(1, int(len(df) * 0.8))
        self.scaler_price.fit(price_data[:split_idx])
        self.scaler_news.fit(news_block[:split_idx])
        self.scaler_target.fit(target_data[:split_idx])

        self.price_data = torch.tensor(self.scaler_price.transform(price_data),
                                       dtype=torch.float32)
        self.news_data = torch.tensor(self.scaler_news.transform(news_block),
                                      dtype=torch.float32)
        self.targets_scaled = torch.tensor(
            self.scaler_target.transform(target_data).flatten(), dtype=torch.float32)
        self.targets_raw = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.price_data) - self.sequence_length)

    def __getitem__(self, idx: int):
        end = idx + self.sequence_length
        return (self.price_data[idx:end], self.news_data[idx:end],
                self.targets_scaled[end - 1], self.targets_raw[end - 1])


def _flatten(ds):
    loader = DataLoader(ds, batch_size=XGBOOST_FLATTEN_BATCH_SIZE, shuffle=False)
    xs, ys = [], []
    for p_b, n_b, _, t_raw in loader:
        xs.append(np.hstack([p_b.numpy().reshape(p_b.shape[0], -1), n_b.numpy().reshape(n_b.shape[0], -1)]))
        ys.append(t_raw.numpy())
    return np.vstack(xs), np.concatenate(ys)


def train_one_model(
    df: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
    news_cols: list[str],
    nlp_cols: list[str],
    *,
    training_epochs: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
):
    logger.connect_configuration(training_config_dict(model_name), name=f"{CLEARML_CONFIG_TRAINING_PREFIX}_{model_name}")

    tscv = TimeSeriesSplit(n_splits=TIME_SERIES_CV_SPLITS)
    train_idx, test_idx = list(tscv.split(df))[-1]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_ds = VolatilitySequenceDataset(
        train_df, sequence_length, feature_cols=feature_cols, news_cols=news_cols, nlp_cols=nlp_cols
    )
    test_ds = VolatilitySequenceDataset(
        test_df, sequence_length, feature_cols=feature_cols, news_cols=news_cols, nlp_cols=nlp_cols
    )

    if model_name == "xgboost":
        x_train, y_train = _flatten(train_ds)
        x_val, y_val = _flatten(test_ds)
        model = create_model("xgboost", num_features=x_train.shape[1], news_dim=0)
        model.fit(x_train, y_train, x_val, y_val, feature_names=[f"f{i}" for i in range(x_train.shape[1])])
        preds = model.predict(x_val)
        metrics = calculate_metrics(y_val, preds, TASK_TYPE_VOLATILITY)
        imp = model.get_feature_importance(top_n=20)
        if not imp.empty:
            logger.report_table(CLEARML_TITLE_FEATURE_IMPORTANCE, model_name, imp.round(4), 0)
        print(f"vol | {model_name} | R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")
        return model, preds, y_val, metrics, train_ds, [], []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_model_with_early_stopping(
        model=create_model(model_name, num_features=len(feature_cols), news_dim=len(news_cols) + len(nlp_cols), 
                          hidden_dim=HIDDEN_DIM_LIQUID if model_name == "liquid" else HIDDEN_DIM_DEFAULT).to(device),
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        model_name=model_name, device=device, training_epochs=training_epochs, logger=logger,
        enable_clearml=logger._ENABLE_CLEARML if hasattr(logger, '_ENABLE_CLEARML') else True,
        train_dataset=train_ds
    )


def save_best_model_bundle(
    best: dict,
    feature_cols: list[str],
    news_cols: list[str],
    nlp_cols: list[str],
    output_dir: str = ARTIFACTS_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    name = best["model_name"]
    model = best["model"]
    train_ds: VolatilitySequenceDataset = best["train_ds"]

    scalers_path = os.path.join(output_dir, "best_volatility_scalers.joblib")
    joblib.dump(
        {
            "price_scaler": train_ds.scaler_price,
            "news_scaler": train_ds.scaler_news,
            "target_scaler": train_ds.scaler_target,
        },
        scalers_path,
    )

    if name == "xgboost":
        model_path = os.path.join(output_dir, "best_volatility_xgboost_model.joblib")
        joblib.dump(model.model, model_path)
    else:
        model_path = os.path.join(output_dir, "best_volatility_torch_model.pt")
        torch.save(model.state_dict(), model_path)

    preds_path = os.path.join(output_dir, "best_volatility_predictions.csv")
    pd.DataFrame(
        {
            "target": best["targets"],
            "prediction": best["preds"],
            "absolute_error": np.abs(best["targets"] - best["preds"]),
        }
    ).to_csv(preds_path, index=False)

    meta = {
        "task": f"volatility_{TASK_TYPE_VOLATILITY}",
        "task_type": TASK_TYPE_VOLATILITY,
        "best_model_name": name,
        "selection_metric": get_metrics_config(TASK_TYPE_VOLATILITY)['primary_metric'],
        "target_column": TARGET_COLUMN,
        "best_metrics": metrics,
        "feature_columns": feature_cols,
        "news_columns": news_cols,
        "nlp_columns": nlp_cols,
        "sequence_length": int(best["sequence_length"]),
        "model_path": model_path,
        "scalers_path": scalers_path,
        "predictions_path": preds_path,
    }
    meta_path = os.path.join(output_dir, "best_volatility_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.upload_artifact("best_volatility_model_meta", meta_path)
    logger.upload_artifact("best_volatility_scalers", scalers_path)
    logger.upload_artifact("best_volatility_model", model_path)
    logger.upload_artifact("best_volatility_predictions", preds_path)
    logger.report_text(f"Saved reusable bundle: {meta_path}", iteration=0)
    print(f"Saved best volatility model: {model_path}\nMetadata: {meta_path}")
    return meta_path


if __name__ == "__main__":
    models_to_test = logger.get_parameter("models_to_test", MODELS_TO_TEST)
    logger.connect_configuration({"models_to_test": models_to_test}, name=CLEARML_CONFIG_RUN)
    print(f"ClearML enabled={logger.is_enabled} | models_to_test={models_to_test}")

    df, feat_cols, news_cols, nlp_cols = load_and_prepare_data(
        DATA_BTC_DAY_PATH,
        DATA_BULL_PATH,
        DATA_BEAR_PATH,
        start_date=logger.get_parameter("data_start_date", DATA_START_DATE),
        end_date=logger.get_parameter("data_end_date", DATA_END_DATE),
        forecast_horizon=logger.get_parameter("forecast_horizon", FORECAST_HORIZON),
        volatility_window=logger.get_parameter("volatility_window", VOLATILITY_WINDOW),
        use_nlp=logger.get_parameter("use_nlp", USE_NLP),
        use_news_filter=logger.get_parameter("use_news_filter", USE_NEWS_FILTER),
    )

    epochs = logger.get_parameter("training_epochs", TRAINING_EPOCHS)
    batch = logger.get_parameter("batch_size", BATCH_SIZE)
    lr = logger.get_parameter("learning_rate", LEARNING_RATE)
    seq_len = logger.get_parameter("sequence_length", SEQUENCE_LENGTH)

    results: dict = {}
    best = None

    for model_name in models_to_test:
        print(f"\nTraining: {model_name}")
        result = train_one_model(
            df,
            model_name,
            feat_cols,
            news_cols,
            nlp_cols,
            training_epochs=epochs,
            batch_size=batch,
            learning_rate=lr,
            sequence_length=seq_len,
        )
        
        # Распаковка результата (может быть 5 или 7 элементов)
        if len(result) == 7:
            model, preds, targets, metrics, train_ds, train_losses, val_losses = result
            results[model_name] = {
                "metrics": metrics,
                "preds": preds,
                "targets": targets,
                "train_losses": train_losses,
                "val_losses": val_losses
            }
        else:
            model, preds, targets, metrics, train_ds = result
            results[model_name] = {
                "metrics": metrics,
                "preds": preds,
                "targets": targets
            }

        # Логирование всех метрик для сравнения
        for metric_name, value in metrics.items():
            logger.report_scalar(
                CLEARML_TITLE_COMPARISON,
                comparison_scalar_series(metric_name, model_name),
                value,
                0,
            )

        # Выбор лучшей модели по основной метрике
        current_value = get_primary_metric_value(metrics, TASK_TYPE_VOLATILITY)
        best_value = get_primary_metric_value(best["metrics"], TASK_TYPE_VOLATILITY) if best else float('-inf')
        
        if best is None or current_value > best_value:
            best_result = result
            if len(best_result) == 7:
                _, _, _, _, _, best_train_losses, best_val_losses = best_result
            else:
                best_train_losses, best_val_losses = [], []
            best = {
                "model_name": model_name,
                "model": model,
                "metrics": metrics,
                "preds": preds,
                "targets": targets,
                "train_losses": best_train_losses,
                "val_losses": best_val_losses,
                "sequence_length": seq_len,
                "train_ds": train_ds,
            }

    comparison_df = (
        pd.DataFrame(
            [
                results[m]["metrics"]
                for m in results
            ]
        )
        .sort_values(get_metrics_config(TASK_TYPE_VOLATILITY)['primary_metric'], ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    comparison_path = os.path.join(ARTIFACTS_DIR, "volatility_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    logger.report_table("model_comparison_table", "summary", comparison_df.round(4), 0)
    logger.upload_artifact("volatility_model_comparison_csv", comparison_path)

    print(f"\nModel comparison (sorted by {get_metrics_config(TASK_TYPE_VOLATILITY)['primary_metric']}):")
    print(comparison_df.to_string(index=False))
    
    best_metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in best["metrics"].items()])
    print(f"\nBest model: {best['model_name']} | {best_metrics_str}")

    save_best_model_bundle(best, feat_cols, news_cols, nlp_cols)

    enable_clearml = logger._ENABLE_CLEARML if hasattr(logger, '_ENABLE_CLEARML') else True

    for model_name in results:
        if "train_losses" in results[model_name] and results[model_name]["train_losses"]:
            train_losses = results[model_name]["train_losses"]
            val_losses = results[model_name]["val_losses"]
            plot_training_losses(
                train_losses, val_losses, model_name,
                save_path=f"{ARTIFACTS_DIR}/{model_name}_training_losses.png",
                logger=logger if enable_clearml else None,
                title_suffix="Volatility",
            )

    plot_model_comparison(
        results, get_metrics_config(TASK_TYPE_VOLATILITY)["primary_metric"],
        TASK_TYPE_VOLATILITY,
        save_path=f"{ARTIFACTS_DIR}/volatility_model_comparison.png",
        logger=logger if enable_clearml else None,
        title_suffix="Volatility",
    )

    plot_prediction_scatter(
        best["targets"], best["preds"], best["model_name"],
        task_type=TASK_TYPE_VOLATILITY,
        save_path=f"{ARTIFACTS_DIR}/best_volatility_prediction_scatter.png",
        logger=logger if enable_clearml else None,
        title_suffix="Volatility",
    )

    plot_prediction_distribution(
        best["targets"], best["preds"], best["model_name"],
        task_type=TASK_TYPE_VOLATILITY,
        save_path=f"{ARTIFACTS_DIR}/best_volatility_prediction_distribution.png",
        logger=logger if enable_clearml else None,
        title_suffix="Volatility",
    )

    if best["model_name"] == "xgboost" and "model" in best:
        imp = best["model"].get_feature_importance(top_n=20)
        if not imp.empty:
            plot_feature_importance(
                imp, best["model_name"],
                save_path=f"{ARTIFACTS_DIR}/best_volatility_feature_importance.png",
                logger=logger if enable_clearml else None,
                title_suffix="Volatility",
            )

    logger.mark_completed()
    logger.close()
