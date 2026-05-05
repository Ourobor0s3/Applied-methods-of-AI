"""
Сравнение моделей на регрессии лог-волатильности (таргет: target_volatility_log).
Конфигурация и пути к данным: experiment_config.py.
"""

from __future__ import annotations

import json
import os
import warnings

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
    TASK_TYPE,
    CLASSIFICATION_THRESHOLD,
    LEARNING_RATE,
    MODELS_TO_TEST,
    NLP_SENTENCE_MODEL_NAME,
    NLP_MAX_EMBEDDING_DIM,
    RSI_WINDOW,
    SEQUENCE_LENGTH,
    TIME_SERIES_CV_SPLITS,
    TRAINING_EPOCHS,
    USE_NLP,
    VOLATILITY_WINDOW,
    VOLUME_MA_WINDOW,
    WEIGHT_DECAY,
    XGBOOST_FLATTEN_BATCH_SIZE,
    clearml_base_parameters,
    comparison_scalar_series,
    training_config_dict,
)
from models_factory import create_model
from nlp_features import NewsTitleEncoder, aggregate_news_embeddings_with_votes, create_enhanced_vote_features
from unified_metrics import calculate_metrics, get_metrics_config, get_primary_metric_value

warnings.filterwarnings("ignore")

logger = create_logger(
    project_name=CLEARML_PROJECT_NAME,
    task_name=CLEARML_TASK_NAME_VOLATILITY,
    task_type="training",
    reuse_last_task_id=CLEARML_REUSE_LAST_TASK_ID,
    **clearml_base_parameters(),
)

TARGET_COLUMN = "target_volatility_log"


def load_and_prepare_data(
    candles_path,
    bull_path,
    bear_path,
    start_date=DATA_START_DATE,
    end_date=DATA_END_DATE,
    forecast_horizon=FORECAST_HORIZON,
    volatility_window=VOLATILITY_WINDOW,
    use_nlp=USE_NLP,
):
    """Свечи, новости, опционально NLP; таргет — log1p(|future rolling vol| * 100)."""
    logger.connect_configuration(
        {
            "start_date": start_date,
            "end_date": end_date,
            "forecast_horizon": forecast_horizon,
            "volatility_window": volatility_window,
            "use_nlp": use_nlp,
            "nlp_sentence_model_name": NLP_SENTENCE_MODEL_NAME,
        },
        name=CLEARML_CONFIG_DATA,
    )

    candles = pd.read_csv(candles_path, parse_dates=["Open time"])
    bull = pd.read_csv(bull_path, parse_dates=["datetime"])
    bear = pd.read_csv(bear_path, parse_dates=["datetime"])

    for news_df in (bull, bear):
        for col in ("positive_votes", "negative_votes", "important_votes"):
            if col not in news_df.columns:
                news_df[col] = 0
    bull["negative_votes"] = 0
    bear["positive_votes"] = 0
    all_news = pd.concat([bull, bear], ignore_index=True)

    candles = candles[(candles["Open time"] >= start_date) & (candles["Open time"] <= end_date)].copy()
    all_news = all_news[(all_news["datetime"] >= start_date) & (all_news["datetime"] <= end_date)].copy()

    candles["returns"] = candles["Close"].pct_change()
    candles["volatility"] = candles["returns"].rolling(volatility_window).std()
    candles["price_range"] = (candles["High"] - candles["Low"]) / candles["Close"]
    candles["volume_ma"] = candles["Volume"].rolling(VOLUME_MA_WINDOW).mean()
    candles["volume_ratio"] = candles["Volume"] / (candles["volume_ma"] + 1e-8)
    candles["ema_12"] = candles["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    candles["ema_26"] = candles["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    candles["macd"] = candles["ema_12"] - candles["ema_26"]

    delta = candles["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_WINDOW).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_WINDOW).mean()
    candles["rsi"] = 100 - (100 / (1 + loss.replace(0, np.nan)))

    all_news["sentiment_score"] = (
        all_news["positive_votes"] * 1.0
        + all_news["important_votes"] * 0.5
        - all_news["negative_votes"] * 0.8
    )
    all_news["sentiment_abs"] = all_news["sentiment_score"].abs()
    
    # Расширенные признаки голосов
    all_news = create_enhanced_vote_features(all_news)

    nlp_cols: list[str] = []
    if use_nlp:
        encoder = NewsTitleEncoder(model_name=NLP_SENTENCE_MODEL_NAME)
        news_with_emb = aggregate_news_embeddings_with_votes(all_news, encoder, max_emb_dim=NLP_MAX_EMBEDDING_DIM)
        news_agg = (
            all_news.groupby(all_news["datetime"].dt.date)
            .agg({
                "sentiment_score": ["sum", "mean", "std"], 
                "title": "count",
                "sentiment_weighted": ["sum", "mean", "std"],
                "total_votes": ["sum", "mean"],
                "positive_ratio": ["mean"],
                "negative_ratio": ["mean"], 
                "reaction_intensity": ["mean", "std"],
                "consensus_score": ["mean", "std"]
            })
            .reset_index()
        )
        news_agg.columns = [
            "date", "sentiment_sum", "sentiment_mean", "sentiment_std", "news_count",
            "sentiment_weighted_sum", "sentiment_weighted_mean", "sentiment_weighted_std",
            "total_votes_sum", "total_votes_mean", "positive_ratio_mean", 
            "negative_ratio_mean", "reaction_intensity_mean", "reaction_intensity_std",
            "consensus_score_mean", "consensus_score_std"
        ]
        news_agg["date"] = pd.to_datetime(news_agg["date"])
        news_daily = news_agg.merge(news_with_emb, on="date", how="left")
        nlp_cols = [c for c in news_daily.columns if "nlp_emb" in c]
    else:
        all_news["date"] = all_news["datetime"].dt.date
        news_daily = (
            all_news.groupby("date")
            .agg({"sentiment_score": ["sum", "mean", "std"], "title": "count"})
            .reset_index()
        )
        news_daily.columns = ["date", "sentiment_sum", "sentiment_mean", "sentiment_std", "news_count"]
        news_daily["date"] = pd.to_datetime(news_daily["date"])

    candles["date"] = pd.to_datetime(candles["Open time"].dt.date)
    df = candles.merge(news_daily, on="date", how="left")

    expected_news_cols = [
        "sentiment_sum",
        "sentiment_mean",
        "sentiment_std",
        "news_count",
        "sentiment_weighted_sum",
        "sentiment_weighted_mean",
        "total_votes_mean",
        "positive_ratio_mean",
        "negative_ratio_mean",
        "reaction_intensity_mean",
        "consensus_score_mean",
    ]
    news_cols = [col for col in expected_news_cols if col in df.columns]
    for col in news_cols + nlp_cols:
        df[col] = df[col].fillna(0)
    
    # Логирование новой информации о голосах
    logger.report_scalar(
        title="User Votes Analysis",
        series="Total News Items",
        value=len(all_news),
        iteration=0
    )
    logger.report_scalar(
        title="User Votes Analysis", 
        series="Average Total Votes per News",
        value=all_news["total_votes"].mean(),
        iteration=0
    )
    logger.report_scalar(
        title="User Votes Analysis",
        series="Average Reaction Intensity", 
        value=all_news["reaction_intensity"].mean(),
        iteration=0
    )
    logger.report_scalar(
        title="User Votes Analysis",
        series="Average Consensus Score",
        value=all_news["consensus_score"].mean(),
        iteration=0
    )

    if "returns" not in df.columns:
        df["returns"] = df["Close"].pct_change()

    df["target_volatility"] = df["returns"].shift(-forecast_horizon).rolling(volatility_window).std()
    df[TARGET_COLUMN] = np.log1p(df["target_volatility"].abs() * 100)

    feature_cols = ["returns", "volatility", "volume_ratio", "price_range", "macd", "rsi"]
    df = df.dropna(subset=feature_cols + news_cols + nlp_cols + [TARGET_COLUMN])

    if len(df) > 100:
        df = df[df["target_volatility"] <= df["target_volatility"].quantile(0.999)]

    df = df[np.isfinite(df[TARGET_COLUMN])].reset_index(drop=True)

    logger.report_scalar(CLEARML_TITLE_DATASET, "sample_count", float(len(df)), 0)
    logger.report_scalar(CLEARML_TITLE_DATASET, "nlp_feature_count", float(len(nlp_cols)), 0)

    return df, feature_cols, news_cols, nlp_cols


class VolatilitySequenceDataset(Dataset):
    """Скейлеры для X, news и y; обучение на scaled target, метрики на сыром target (как в исходной логике)."""

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        target_col: str = TARGET_COLUMN,
        feature_cols: list[str] | None = None,
        news_cols: list[str] | None = None,
        nlp_cols: list[str] | None = None,
    ):
        self.sequence_length = sequence_length
        feature_cols = feature_cols or ["returns", "volatility", "volume_ratio", "price_range", "macd", "rsi"]
        news_cols = news_cols or [
            "sentiment_sum", "sentiment_mean", "sentiment_std", "news_count",
            "sentiment_weighted_sum", "sentiment_weighted_mean", "total_votes_mean",
            "positive_ratio_mean", "negative_ratio_mean", "reaction_intensity_mean",
            "consensus_score_mean"
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

        self.price_data = torch.tensor(self.scaler_price.transform(price_data), dtype=torch.float32)
        self.news_data = torch.tensor(self.scaler_news.transform(news_block), dtype=torch.float32)
        self.targets_scaled = torch.tensor(self.scaler_target.transform(target_data).flatten(), dtype=torch.float32)
        self.targets_raw = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.price_data) - self.sequence_length)

    def __getitem__(self, idx: int):
        end = idx + self.sequence_length
        return (
            self.price_data[idx:end],
            self.news_data[idx:end],
            self.targets_scaled[end - 1],
            self.targets_raw[end - 1],
        )


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

        def flatten_dataset(ds: VolatilitySequenceDataset):
            loader = DataLoader(ds, batch_size=XGBOOST_FLATTEN_BATCH_SIZE, shuffle=False)
            xs, ys = [], []
            for p_b, n_b, _, t_raw in loader:
                xs.append(np.hstack([p_b.numpy().reshape(p_b.shape[0], -1), n_b.numpy().reshape(n_b.shape[0], -1)]))
                ys.append(t_raw.numpy())
            return np.vstack(xs), np.concatenate(ys)

        x_train, y_train = flatten_dataset(train_ds)
        x_val, y_val = flatten_dataset(test_ds)

        model = create_model("xgboost", num_features=x_train.shape[1], news_dim=0)
        feat_names = [f"feature_{i}" for i in range(x_train.shape[1])]
        model.fit(x_train, y_train, x_val, y_val, feature_names=feat_names)

        preds = model.predict(x_val)
        y_true = y_val
        
        # Unified regression metrics
        metrics = calculate_metrics(y_true, preds, task_type=TASK_TYPE, threshold=CLASSIFICATION_THRESHOLD)
        
        # Важность признаков
        importance = model.get_feature_importance(top_n=30)
        if not importance.empty:
            logger.report_table(CLEARML_TITLE_FEATURE_IMPORTANCE, model_name, importance.round(4), 0)

        # Логирование всех метрик
        for metric_name, value in metrics.items():
            logger.report_scalar(CLEARML_TITLE_METRICS_FINAL, f"{model_name}.{metric_name}", value, 0)
        
        # Print all metrics
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"volatility | model={model_name} | {metrics_str}")
        
        return model, preds, y_true, metrics, train_ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = HIDDEN_DIM_LIQUID if model_name == "liquid" else HIDDEN_DIM_DEFAULT
    model = create_model(
        model_name,
        num_features=len(feature_cols),
        news_dim=len(news_cols) + len(nlp_cols),
        hidden_dim=hidden,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=training_epochs)
    loss_fn = nn.HuberLoss(delta=HUBER_LOSS_DELTA)

    param_count = sum(p.numel() for p in model.parameters())
    logger.report_scalar(CLEARML_TITLE_MODEL, f"{model_name}.parameter_count", float(param_count), 0)

    for epoch in range(training_epochs):
        model.train()
        for p_b, n_b, t_sc, _ in train_loader:
            opt.zero_grad()
            pred = model(p_b.to(device), n_b.to(device)).squeeze()
            loss_fn(pred, t_sc.to(device)).backward()
            opt.step()
        sched.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for p_b, n_b, _, t_raw in test_loader:
                pred = model(p_b.to(device), n_b.to(device)).squeeze()
                preds_list.extend(pred.cpu().numpy())
                targets_list.extend(t_raw.numpy())

        preds_arr = np.asarray(preds_list)
        y_true = np.asarray(targets_list)
        
        # Unified metrics based on TASK_TYPE
        metrics = calculate_metrics(y_true, preds_arr, task_type=TASK_TYPE, threshold=CLASSIFICATION_THRESHOLD)

        it = epoch + 1
        
        # Логирование всех метрик за эпоху
        for metric_name, value in metrics.items():
            logger.report_scalar(CLEARML_TITLE_METRICS_EPOCH, f"{model_name}.{metric_name}", value, it)

        if epoch % 10 == 0:
            metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"volatility | model={model_name} | epoch={it} | {metrics_str}")

    # Финальное логирование
    for metric_name, value in metrics.items():
        logger.report_scalar(CLEARML_TITLE_METRICS_FINAL, f"{model_name}.{metric_name}", value, 0)

    return model, preds_arr, y_true, metrics, train_ds


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
        "task": f"volatility_{TASK_TYPE}",
        "task_type": TASK_TYPE,
        "best_model_name": name,
        "selection_metric": get_metrics_config(TASK_TYPE)['primary_metric'],
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
    )

    epochs = logger.get_parameter("training_epochs", TRAINING_EPOCHS)
    batch = logger.get_parameter("batch_size", BATCH_SIZE)
    lr = logger.get_parameter("learning_rate", LEARNING_RATE)
    seq_len = logger.get_parameter("sequence_length", SEQUENCE_LENGTH)

    results: dict = {}
    best = None

    for model_name in models_to_test:
        print(f"\nTraining: {model_name}")
        model, preds, targets, metrics, train_ds = train_one_model(
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
        current_value = get_primary_metric_value(metrics, TASK_TYPE)
        best_value = get_primary_metric_value(best["metrics"], TASK_TYPE) if best else float('-inf')
        
        if best is None or current_value > best_value:
            best = {
                "model_name": model_name,
                "model": model,
                "metrics": metrics,
                "preds": preds,
                "targets": targets,
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
        .sort_values(get_metrics_config(TASK_TYPE)['primary_metric'], ascending=False)
        .reset_index(drop=True)
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    comparison_path = os.path.join(ARTIFACTS_DIR, "volatility_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    logger.report_table("model_comparison_table", "summary", comparison_df.round(4), 0)
    logger.upload_artifact("volatility_model_comparison_csv", comparison_path)

    print(f"\nModel comparison (sorted by {get_metrics_config(TASK_TYPE)['primary_metric']}):")
    print(comparison_df.to_string(index=False))
    
    best_metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in best["metrics"].items()])
    print(f"\nBest model: {best['model_name']} | {best_metrics_str}")

    save_best_model_bundle(best, feat_cols, news_cols, nlp_cols)

    logger.mark_completed()
    logger.close()
