# nlp_features.py
"""
NLP-фичи из заголовков новостей и голосов пользователей для улучшения прогноза волатильности
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
import subprocess
import json
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


def _safe_log(message: str) -> None:
    """Печать, устойчивая к ограничениям кодировки консоли."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="ignore").decode("ascii"))


class NewsTitleEncoder:
    """
    Кодировщик заголовков новостей и пользовательских голосов в векторные эмбеддинги.
    Поддерживает: Ollama, Sentence Transformers или TF-IDF (fallback).
    """
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu and _SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        self.is_transformer = False
        self.is_ollama = False
        self._is_fitted = False
        self.embedding_dim = None
        
        # Проверяем Ollama (модели с ':' в имени, например "qwen3-embedding:0.6b")
        if ":" in model_name:
            if _OLLAMA_AVAILABLE:
                try:
                    response = ollama.embed(model=model_name, input="test")
                    self.is_ollama = True
                    self.embedding_dim = len(response['embeddings'][0])
                    _safe_log(f"Loaded Ollama model: {model_name}, dim={self.embedding_dim}")
                    return
                except Exception as e:
                    error_msg = str(e).lower()
                    if "not found" in error_msg:
                        _safe_log(f"Ollama model not found: {model_name}. Pull with: ollama pull {model_name}")
                    else:
                        _safe_log(f"Ollama error: {e}")
            # Если Ollama недоступен или модель не найдена -> TF-IDF
            _safe_log("Using TF-IDF fallback")
            self._init_tfidf_fallback()
        # Пробуем Sentence Transformers для обычных имен (paraphrase-MiniLM-L6-v2, Qwen/Qwen3-Embedding-0.5B)
        elif _SENTENCE_TRANSFORMERS_AVAILABLE:
            self._try_sentence_transformer()
        else:
            _safe_log("Using TF-IDF fallback")
            self._init_tfidf_fallback()
    
    def _try_sentence_transformer(self):
        """Попытка загрузить Sentence Transformer."""
        try:
            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            self.is_transformer = True
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            _safe_log(f"Loaded SentenceTransformer: {self.model_name} ({device}), dim={self.embedding_dim}")
        except Exception as e:
            _safe_log(f"Failed to load SentenceTransformer: {e}")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Fallback на TF-IDF если transformers недоступны"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.is_transformer = False
    
    def fit(self, titles: List[str]):
        """Fit только для TF-IDF режима"""
        if not self.is_transformer:
            titles_clean = [str(t) if pd.notna(t) else "" for t in titles]
            self.vectorizer.fit(titles_clean)
            self._is_fitted = True
        return self
    
    def transform(self, titles: List[str]) -> np.ndarray:
        """Преобразование заголовков в эмбеддинги"""
        titles_clean = [str(t) if pd.notna(t) else "" for t in titles]
        
        if self.is_ollama:
            # Ollama embeddings
            all_embeddings = []
            batch_size = 32
            for i in range(0, len(titles_clean), batch_size):
                batch = titles_clean[i:i+batch_size]
                try:
                    response = ollama.embed(model=self.model_name, input=batch)
                    all_embeddings.extend(response['embeddings'])
                except Exception as e:
                    _safe_log(f"Ollama batch error: {e}")
                    all_embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
            return np.array(all_embeddings)
        elif self.is_transformer and self.model:
            # Sentence Transformer
            embeddings = self.model.encode(titles_clean, show_progress_bar=False, batch_size=32)
            return embeddings
        else:
            # TF-IDF fallback
            if not self._is_fitted:
                self.fit(titles_clean)
            return self.vectorizer.transform(titles_clean).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Названия фич для интерпретации"""
        if self.embedding_dim:
            return [f"title_emb_{i}" for i in range(self.embedding_dim)]
        else:
            return self.vectorizer.get_feature_names_out().tolist()


def create_enhanced_vote_features(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Создание расширенных признаков на основе пользовательских голосов
    """
    df = df_news.copy()
    
    # Базовые метрики голосов
    df['total_votes'] = df['positive_votes'] + df['negative_votes'] + df['important_votes']
    df['positive_ratio'] = df['positive_votes'] / (df['total_votes'] + 1e-8)
    df['negative_ratio'] = df['negative_votes'] / (df['total_votes'] + 1e-8)
    df['important_ratio'] = df['important_votes'] / (df['total_votes'] + 1e-8)
    
    # Комбинированный сентимент с весами
    df['sentiment_weighted'] = (
        df['positive_votes'] * 1.0 +
        df['important_votes'] * 0.7 -
        df['negative_votes'] * 1.2
    )
    
    # Интенсивность реакции (how polarized is the audience)
    df['reaction_intensity'] = df['total_votes'] / (df['title'].str.len() + 1)
    
    # Консенсус/диссонанс (показатель единодушия)
    df['consensus_score'] = abs(df['positive_votes'] - df['negative_votes']) / (df['total_votes'] + 1e-8)
    
    # Активность важности (important votes relative to total)
    df['importance_activity'] = df['important_votes'] / (df['total_votes'] + 1e-8)
    
    return df


def reduce_embedding_dimension(embeddings: np.ndarray, max_dim: int = 64) -> np.ndarray:
    """Уменьшение размерности эмбеддингов через PCA."""
    if embeddings.shape[1] <= max_dim:
        return embeddings
    
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_dim, random_state=42)
        return pca.fit_transform(embeddings)
    except Exception:
        # Fallback: просто берем первые max_dim признаков
        return embeddings[:, :max_dim]


def aggregate_news_embeddings_with_votes(df_news: pd.DataFrame, 
                                       encoder: NewsTitleEncoder,
                                       date_col: str = 'datetime',
                                       agg_methods: List[str] = None,
                                       max_emb_dim: int = 64) -> pd.DataFrame:
    """
    Агрегация эмбеддингов новостей и голосов по временным периодам.
    max_emb_dim: максимальная размерность эмбеддингов (для избежания переполнения памяти)
    """
    if agg_methods is None:
        agg_methods = ['mean', 'max', 'std']
    
    df_news = df_news.copy()
    df_news['date'] = pd.to_datetime(df_news[date_col].dt.date)
    
    # Создаем расширенные признаки голосов
    df_enhanced = create_enhanced_vote_features(df_news)
    
    # Фильтруем пустые заголовки
    df_enhanced = df_enhanced[df_enhanced['title'].notna() & (df_enhanced['title'].str.len() > 0)].copy()
    
    if len(df_enhanced) == 0:
        _safe_log("No valid titles for NLP processing")
        # Возвращаем пустой DataFrame с нужной колонкой date
        return pd.DataFrame(columns=['date'])
    
    # Кодируем заголовки
    _safe_log(f"Encoding {len(df_enhanced)} titles...")
    embeddings = encoder.transform(df_enhanced['title'].tolist())
    
    # Уменьшаем размерность если слишком много
    if embeddings.shape[1] > max_emb_dim:
        _safe_log(f"Reducing embedding dim: {embeddings.shape[1]} -> {max_emb_dim}")
        embeddings = reduce_embedding_dimension(embeddings, max_emb_dim)
    
    # Добавляем эмбеддинги в DataFrame
    emb_cols = [f"nlp_emb_{i}" for i in range(embeddings.shape[1])]
    for i, col in enumerate(emb_cols):
        df_enhanced[col] = embeddings[:, i]
    
    # 🔹 Агgregation: строим agg_dict для всех признаков
    agg_dict = {}
    
    # Эмбеддинги
    for col in emb_cols:
        agg_dict[col] = agg_methods
    
    # Расширенные признаки голосов
    vote_features = [
        'sentiment_weighted', 'total_votes', 'positive_ratio', 'negative_ratio', 
        'important_ratio', 'reaction_intensity', 'consensus_score', 'importance_activity'
    ]
    for col in vote_features:
        if col in df_enhanced.columns:
            agg_dict[col] = ['sum', 'mean', 'std', 'max']
    
    # Базовые агрегаты
    if 'sentiment_score' in df_enhanced.columns:
        agg_dict['sentiment_score'] = ['sum', 'mean', 'std']
    if 'title' in df_enhanced.columns:
        agg_dict['title'] = 'count'
    
    # Группируем по датам
    result = df_enhanced.groupby('date').agg(agg_dict).reset_index()
    
    # Плоская структура колонок для совместимости с остальным кодом
    result.columns = ['_'.join(col).strip() if col[1] else col[0] for col in result.columns.values]
    
    return result