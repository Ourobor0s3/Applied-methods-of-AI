# nlp_features.py
"""
NLP-фичи из заголовков новостей для улучшения прогноза волатильности
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class NewsTitleEncoder:
    """
    Кодировщик заголовков новостей в векторные эмбеддинги.
    Поддерживает: Sentence Transformers (рекомендуется) или TF-IDF (fallback).
    """
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu and _SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        self.is_transformer = False
        
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                device = "cuda" if self.use_gpu else "cpu"
                self.model = SentenceTransformer(model_name, device=device)
                self.is_transformer = True
                print(f"✅ Загружена модель: {model_name} ({device})")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить SentenceTransformer: {e}")
                self._init_tfidf_fallback()
        else:
            print("⚠️ sentence-transformers не установлен, используем TF-IDF")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Fallback на TF-IDF если transformers недоступны"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.is_transformer = False
    
    def fit(self, titles: List[str]):
        """Fit только для TF-IDF режима"""
        if not self.is_transformer:
            titles_clean = [str(t) if pd.notna(t) else "" for t in titles]
            self.vectorizer.fit(titles_clean)
        return self
    
    def transform(self, titles: List[str]) -> np.ndarray:
        """Преобразование заголовков в эмбеддинги"""
        titles_clean = [str(t) if pd.notna(t) else "" for t in titles]
        
        if self.is_transformer and self.model:
            # Sentence Transformer: фиксированная размерность (384 для MiniLM)
            embeddings = self.model.encode(titles_clean, show_progress_bar=False, batch_size=32)
            # Добавляем статистики для агрегации по дням
            return embeddings
        else:
            # TF-IDF fallback
            return self.vectorizer.transform(titles_clean).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Названия фич для интерпретации"""
        if self.is_transformer:
            return [f"title_emb_{i}" for i in range(384)]  # MiniLM dimension
        else:
            return self.vectorizer.get_feature_names_out().tolist()


def aggregate_news_embeddings(df_news: pd.DataFrame, 
                              encoder: NewsTitleEncoder,
                              date_col: str = 'datetime',
                              agg_methods: List[str] = None) -> pd.DataFrame:
    """
    Агрегация эмбеддингов новостей по дням
    """
    if agg_methods is None:
        agg_methods = ['mean', 'max']
    
    df_news = df_news.copy()
    df_news['date'] = pd.to_datetime(df_news[date_col].dt.date)
    
    # Фильтруем пустые заголовки
    df_news = df_news[df_news['title'].notna() & (df_news['title'].str.len() > 0)].copy()
    
    if len(df_news) == 0:
        print("⚠️ Нет валидных заголовков для NLP-обработки")
        # Возвращаем пустой DataFrame с нужной колонкой date
        return pd.DataFrame(columns=['date'])
    
    # Кодируем заголовки
    print(f"🔤 Кодируем {len(df_news)} заголовков...")
    embeddings = encoder.transform(df_news['title'].tolist())
    
    # Добавляем эмбеддинги в DataFrame
    emb_cols = [f"nlp_emb_{i}" for i in range(embeddings.shape[1])]
    for i, col in enumerate(emb_cols):
        df_news[col] = embeddings[:, i]
    
    # 🔹 Агрегация: строим agg_dict ТОЛЬКО для существующих колонок
    agg_dict = {}
    
    # Эмбеддинги (они точно есть)
    for col in emb_cols:
        agg_dict[col] = agg_methods
    
    # Базовые агрегаты — добавляем только если колонка существует
    available_agg = {}
    if 'sentiment_score' in df_news.columns:
        available_agg['sentiment_score'] = ['sum', 'mean', 'std']
    if 'title' in df_news.columns:
        available_agg['title'] = 'count'
    if 'sentiment_abs' in df_news.columns:
        available_agg['sentiment_abs'] = ['mean', 'max']
    
    agg_dict.update(available_agg)
    
    # Если ничего не осталось для агрегации — возвращаем минимальный результат
    if not agg_dict:
        return df_news[['date']].drop_duplicates().reset_index(drop=True)
    
    result = df_news.groupby('date').agg(agg_dict).reset_index()
    
    # Flatten column names
    result.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in result.columns]
    
    print(f"✅ Создано {len([c for c in result.columns if 'nlp_emb' in c])} NLP-признаков")
    return result