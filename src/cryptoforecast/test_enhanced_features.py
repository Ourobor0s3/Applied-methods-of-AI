#!/usr/bin/env python3
"""
Тестирование улучшенных NLP-признаков с учетом голосов пользователей
"""

import pandas as pd
import numpy as np
from nlp_features import create_enhanced_vote_features, NewsTitleEncoder
from experiment_config import DATA_BEAR_PATH, DATA_BULL_PATH

def test_enhanced_features():
    """Тестирование новых признаков на реальных данных"""
    
    print("=" * 60)
    print("Тестирование улучшенных NLP-признаков")
    print("=" * 60)
    
    # Загрузка данных
    print("Загрузка данных новостей...")
    bear = pd.read_csv(DATA_BEAR_PATH, parse_dates=["datetime"])
    bull = pd.read_csv(DATA_BULL_PATH, parse_dates=["datetime"])
    
    # Заполняем пропущенные колонки
    for news_df in (bear, bull):
        for col in ("positive_votes", "negative_votes", "important_votes"):
            if col not in news_df.columns:
                news_df[col] = 0
    
    all_news = pd.concat([bear, bull], ignore_index=True)
    
    print(f"Всего новостей: {len(all_news)}")
    
    # Создание улучшенных признаков
    print("\nСоздание улучшенных признаков...")
    enhanced_news = create_enhanced_vote_features(all_news)
    
    print("Колонки в данных:", enhanced_news.columns.tolist())
    
    # Анализ результатов
    print("\nСтатистика по новым признакам:")
    print(f"Среднее количество голосов: {enhanced_news['total_votes'].mean():.2f}")
    print(f"Средний сентимент: {enhanced_news['sentiment_weighted'].mean():.2f}")
    print(f"Средняя интенсивность реакции: {enhanced_news['reaction_intensity'].mean():.4f}")
    print(f"Средний консенсус: {enhanced_news['consensus_score'].mean():.3f}")
    
    # Корреляция с волатильностью
    print("\nТоп-10 новостей с наибольшей реакцией:")
    top_reacted = enhanced_news.nlargest(10, 'reaction_intensity')
    for _, row in top_reacted.iterrows():
        print(f"Реакция: {row['reaction_intensity']:.4f} | Голоса: {row['total_votes']} | {row['title'][:50]}...")
    
    print("\nТоп-10 новостей с наибольшим консенсусом:")
    top_consensus = enhanced_news.nlargest(10, 'consensus_score')
    for _, row in top_consensus.iterrows():
        print(f"Консенсус: {row['consensus_score']:.3f} | Голоса: {row['total_votes']} | {row['title'][:50]}...")
    
    return enhanced_news

if __name__ == "__main__":
    test_enhanced_features()