"""
news_filters.py
Предварительная фильтрация и взвешенная агрегация новостей для улучшения качества обучения.

Два режима:
  - Hard filtering: удаляет новости, не прошедшие пороги (credibility, relevance, quality).
  - Weighted aggregation: ВСЕ новости остаются, но при агрегации по дням
    каждая новость получает вес = quality_score.
    Итоговые признаки: weighted_sum, weighted_mean, max_quality.

Фильтры:
  1. Credibility — спам, длина заголовка, минимальная активность
  2. Relevance — минимальное число положительных голосов / reaction intensity
  3. Quality score — композитный скор: reaction_intensity + consensus + votes
"""

from __future__ import annotations

import re

import pandas as pd


def compute_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет все промежуточные признаки новости: votes, reaction, consensus, quality_score."""
    df = df.copy()
    df["total_votes"] = df["positive_votes"] + df["negative_votes"] + df["important_votes"]
    df["reaction_intensity"] = (
        df["positive_votes"] * 1.0 + df["important_votes"] * 0.5 - df["negative_votes"] * 0.8
    ).abs()
    total_pos = df["positive_votes"] + df["important_votes"] + df["negative_votes"]
    df["consensus_score"] = (df["positive_votes"] - df["negative_votes"]).abs() / (total_pos + 1e-8)
    df["quality_score"] = _compute_quality_score(df)
    return df


def _compute_quality_score(df: pd.DataFrame) -> pd.Series:
    ri = df["reaction_intensity"].clip(lower=0)
    cs = df["consensus_score"].clip(lower=0)
    tv = df["total_votes"].clip(lower=0)

    ri_max = ri.max()
    cs_max = cs.max()
    tv_max = tv.max()

    ri_norm = ri / ri_max if ri_max > 0 else ri
    cs_norm = cs / cs_max if cs_max > 0 else cs
    tv_norm = tv / tv_max if tv_max > 0 else tv

    return ri_norm * 0.4 + cs_norm * 0.3 + tv_norm * 0.3


def _apply_credibility(
    df: pd.DataFrame, *, min_total_votes: int, min_title_len: int,
    max_title_len: int, spam_keywords: list[str],
) -> tuple[pd.DataFrame, int]:
    n = len(df)
    df = df[df["title"].str.len().between(min_title_len, max_title_len)]
    if spam_keywords:
        pattern = "|".join(re.escape(kw.lower()) for kw in spam_keywords)
        df = df[~df["title"].str.lower().str.contains(pattern, na=False)]
    if "total_votes" in df.columns:
        df = df[df["total_votes"] >= min_total_votes]
    return df, n - len(df)


def _apply_relevance(
    df: pd.DataFrame, *, min_positive: int, min_reaction: float,
) -> tuple[pd.DataFrame, int]:
    n = len(df)
    if min_positive > 0 and "positive_votes" in df.columns:
        df = df[df["positive_votes"] >= min_positive]
    if min_reaction > 0 and "reaction_intensity" in df.columns:
        df = df[df["reaction_intensity"] >= min_reaction]
    return df, n - len(df)


def _apply_quality_threshold(
    df: pd.DataFrame, *, min_quality: float,
) -> tuple[pd.DataFrame, int]:
    n = len(df)
    df = df[df["quality_score"] >= min_quality]
    return df, n - len(df)


def filter_news(
    news_df: pd.DataFrame,
    *,
    use_filter: bool = True,
    min_total_votes: int = 3,
    min_positive_votes: int = 1,
    min_reaction_intensity: float = 0.3,
    min_title_length: int = 10,
    max_title_length: int = 300,
    spam_keywords: list[str] | None = None,
    min_quality: float = 0.2,
    top_n_per_day: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    if not use_filter:
        df = compute_news_features(news_df.copy())
        return df, {"credibility_dropped": 0, "relevance_dropped": 0, "quality_dropped": 0, "total_dropped": 0}

    stats: dict = {}
    n_initial = len(news_df)
    df = compute_news_features(news_df)

    df, stats["credibility_dropped"] = _apply_credibility(
        df, min_total_votes=min_total_votes, min_title_len=min_title_length,
        max_title_len=max_title_length, spam_keywords=spam_keywords or [],
    )
    df, stats["relevance_dropped"] = _apply_relevance(
        df, min_positive=min_positive_votes, min_reaction=min_reaction_intensity,
    )
    df, stats["quality_dropped"] = _apply_quality_threshold(df, min_quality=min_quality)

    if top_n_per_day and top_n_per_day > 0:
        before = len(df)
        df = (
            df.sort_values("quality_score", ascending=False)
            .groupby(df["datetime"].dt.date, group_keys=False)
            .apply(lambda g: g.head(top_n_per_day))
            .reset_index(drop=True)
        )
        stats["quality_dropped"] += before - len(df)

    stats["total_dropped"] = n_initial - len(df)
    stats["total_initial"] = n_initial
    stats["total_remaining"] = len(df)
    return df, stats


def aggregate_news_weighted(
    news_df: pd.DataFrame,
    date_col: str = "datetime",
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Взвешенная агрегация новостей по дням с весами = quality_score.

    Для каждого дня вычисляет:
      - weighted_sentiment_sum: sum(news.sentiment * news.quality) / sum(quality)
      - weighted_sentiment_mean: mean(news.sentiment * news.quality) / mean(quality)
      - max_quality: максимальный quality_score за день
      - weighted_sentiment_max: sentiment новости с макс. quality_score
      - quality_weighted_count: число новостей с quality > median (активные дни)
      - total_quality: sum(quality_score) — общая "энергия" дня
      - news_count: число новостей за день
    """
    df = news_df.copy()
    df["date"] = pd.to_datetime(df[date_col].dt.date)

    weights = df["quality_score"].fillna(0).clip(lower=0)
    weighted_sent = df[sentiment_col] * weights

    agg_dict = {
        sentiment_col: ["sum", "mean", "std"],
        "quality_score": ["sum", "mean", "max", "std"],
        "total_votes": ["sum", "mean", "max"],
        "reaction_intensity": ["sum", "mean", "max"],
        "consensus_score": ["mean", "max"],
        "title": "count",
    }

    grp = df.groupby("date")
    result = grp.agg(agg_dict)
    result.columns = ["_".join(col).strip() for col in result.columns.values]
    result = result.reset_index()

    weighted_sum = grp.apply(lambda g: weighted_sent.loc[g.index].sum())
    total_weight = grp.apply(lambda g: weights.loc[g.index].sum())
    result["weighted_sentiment"] = (weighted_sum / total_weight.clip(lower=1e-8)).values

    max_q_idx = df.groupby("date")["quality_score"].idxmax()
    result["max_quality_sentiment"] = (
        df.loc[max_q_idx].set_index("date")[sentiment_col].reindex(result["date"]).values
    )

    median_q = df.groupby("date")["quality_score"].transform("median")
    result["quality_weighted_count"] = (
        df[df["quality_score"] > median_q].groupby("date").size().reindex(result["date"]).fillna(0).values
    )

    result["total_quality"] = result["quality_score_sum"]
    result.rename(columns={"title_count": "news_count"}, inplace=True)
    result.rename(columns={
        f"{sentiment_col}_sum": "sentiment_sum",
        f"{sentiment_col}_mean": "sentiment_mean",
        f"{sentiment_col}_std": "sentiment_std",
        "quality_score_mean": "quality_mean",
        "quality_score_std": "quality_std",
        "total_votes_sum": "total_votes_sum",
        "total_votes_mean": "total_votes_mean",
        "total_votes_max": "total_votes_max",
        "reaction_intensity_sum": "reaction_sum",
        "reaction_intensity_mean": "reaction_mean",
        "reaction_intensity_max": "reaction_max",
        "consensus_score_mean": "consensus_mean",
        "consensus_score_max": "consensus_max",
    }, inplace=True)

    result["date"] = pd.to_datetime(result["date"])
    return result