import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import bootstrap, ttest_ind, mannwhitneyu
import warnings
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from constants import DATA_BEAR_PATH, DATA_BTC_DAY_PATH, DATA_BULL_PATH
import json
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ==========================================
# 🔬 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ИССЛЕДОВАНИЙ
# ==========================================

def safe_parse_datetime(df, column_name):
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df[column_name]):
        if df[column_name].dt.tz is not None:
            df[column_name] = df[column_name].dt.tz_localize(None)
        return df
    if pd.api.types.is_numeric_dtype(df[column_name]):
        unit = 'ms' if df[column_name].iloc[0] > 1e12 else 's'
        df[column_name] = pd.to_datetime(df[column_name], unit=unit).dt.tz_localize(None)
        return df
    df[column_name] = pd.to_datetime(df[column_name]).dt.tz_localize(None)
    return df

def filter_by_date_range(df, date_column, start_date, end_date):
    df = safe_parse_datetime(df, date_column)
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df.loc[mask].reset_index(drop=True)

def calculate_market_regime(df, price_column='Close', window=24):
    df = df.copy()
    df['returns'] = df[price_column].pct_change(window)
    bull_thr, bear_thr = 0.02, -0.02
    df['market_regime'] = df['returns'].apply(
        lambda x: 'bull' if pd.notna(x) and x > bull_thr else ('bear' if pd.notna(x) and x < bear_thr else 'sideways')
    )
    return df

def bootstrap_ci(d1, d2, n_boot=1000, conf=0.95):
    try:
        res = bootstrap((d1.values, d2.values), lambda x,y: np.corrcoef(x,y)[0,1],
                       confidence_level=conf, n_resamples=n_boot, method='percentile', random_state=42, paired=True)
        return res.confidence_interval
    except:
        return type('CI', (), {'low': np.nan, 'high': np.nan})()

def export_results(results, merged, filepath='research_output.json'):
    """Экспорт результатов для дальнейшего анализа"""
    export_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(merged),
            'date_range': results.get('date_range'),
        },
        'hypotheses': {
            'reactive_voting': results.get('reactive_voting'),
            'volatility_prediction': results.get('volatility'),
            'temporal_patterns': results.get('temporal'),
            'sentiment_analysis': results.get('sentiment'),
        },
        'statistics': {
            'correlations': {k: v for k,v in results.items() if 'corr' in k.lower() or 'p_value' in k.lower()},
            'feature_importance': results.get('importance', {}),
        }
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"   💾 Результаты экспортированы в {filepath}")


# ==========================================
# 🧠 2. ТЕКСТОВЫЙ КОДЕР + СЕНТИМЕНТ-АНАЛИЗ
# ==========================================

class ResNetBlock1D(nn.Module):
    def __init__(self, in_c, out_c, k=3, dropout=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(in_c, out_c, k, padding='same')
        self.b1 = nn.BatchNorm1d(out_c)
        self.c2 = nn.Conv1d(out_c, out_c, k, padding='same')
        self.b2 = nn.BatchNorm1d(out_c)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Identity() if in_c == out_c else nn.Conv1d(in_c, out_c, 1)
    
    def forward(self, x):
        out = self.dropout(self.relu(self.b1(self.c1(x))))
        out = self.dropout(self.b2(self.c2(out)))
        return self.relu(out + self.skip(x))

class TextEncoder(nn.Module):
    def __init__(self, arch='resnet', out_dim=8):
        super().__init__()
        if arch == 'resnet':
            self.net = nn.Sequential(
                ResNetBlock1D(1, 128), ResNetBlock1D(128, 64),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, out_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(1, 64, 3, padding='same'), nn.ReLU(),
                nn.Conv1d(64, 32, 3, padding='same'),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, out_dim)
            )
    def forward(self, x):
        return self.net(x.unsqueeze(1))

def encode_titles(titles, arch='resnet', out_dim=8, device='cpu'):
    print(f"\n🧠 Кодирование {len(titles)} заголовков ({arch.upper()})...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embeddings = embedder.encode(titles, show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    net = TextEncoder(arch, out_dim).to(device).eval()
    with torch.no_grad():
        feats = net(torch.tensor(embeddings, dtype=torch.float32, device=device)).cpu().numpy()
    return pd.DataFrame(feats, columns=[f'txt_{i}' for i in range(out_dim)])

def analyze_sentiment_bert(titles, device='cpu', batch_size=8):
    """
    Сентимент-анализ через BERTweet с корректной обработкой батчей.
    """
    print(f"\n🤖 Загрузка BERTweet для сентимент-анализа {len(titles)} заголовков...")
    
    try:
        # 🔧 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: tokenizer_kwargs для корректного батчинга
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=0 if device == 'cuda' and torch.cuda.is_available() else -1,
            batch_size=batch_size,
            truncation=True,
            max_length=128,
            tokenizer_kwargs={'padding': True, 'truncation': True}  # ← ВАЖНО!
        )
        
        # Предобработка текстов
        def preprocess(text):
            if pd.isna(text): return "neutral"
            text = str(text)
            text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
            text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\']', ' ', text)
            return text.strip()[:120]
        
        cleaned_titles = [preprocess(t) for t in titles]
        
        # Пакетная обработка
        from tqdm import tqdm
        results = []
        for i in tqdm(range(0, len(cleaned_titles), batch_size), desc="Анализ сентимента"):
            batch = cleaned_titles[i:i+batch_size]
            valid_batch = [(idx, txt) for idx, txt in enumerate(batch) if txt and len(txt) > 3]
            
            if not valid_batch:
                results.extend([0] * len(batch))
                continue
            
            indices, texts = zip(*valid_batch)
            predictions = sentiment_pipe(list(texts))  # ← явный list() для совместимости
            
            batch_sentiments = [0] * len(batch)
            for idx, pred in zip(indices, predictions):
                label = pred['label'].upper()
                score = pred['score']
                if score < 0.6:
                    batch_sentiments[idx] = 0
                elif label == 'POS':
                    batch_sentiments[idx] = 1
                elif label == 'NEG':
                    batch_sentiments[idx] = -1
            results.extend(batch_sentiments)
        
        print(f"   ✅ Обработано: {len(results)} заголовков")
        print(f"   📊 Распределение: +1={results.count(1)}, 0={results.count(0)}, -1={results.count(-1)}")
        return np.array(results)
        
    except Exception as e:
        print(f"   ⚠️ Ошибка BERTweet: {e}")
        print(f"   🔁 Возврат к keyword-методу")
        return analyze_sentiment_simple(titles)


def analyze_sentiment_simple(titles):
    """
    Фоллбэк: простой сентимент-анализ на основе ключевых слов.
    """
    positive_words = {'рост', 'бычий', 'покупка', 'лонг', 'оптимизм', 'прорыв', 'ath', 'зелён', 'bull', 'up', 'gain'}
    negative_words = {'паден', 'медвеж', 'продаж', 'шорт', 'пессимизм', 'обвал', 'красн', 'крах', 'bear', 'down', 'loss', 'crash'}
    
    sentiments = []
    for title in titles:
        if pd.isna(title):
            sentiments.append(0)
            continue
        t_lower = str(title).lower()
        pos_count = sum(1 for w in positive_words if w in t_lower)
        neg_count = sum(1 for w in negative_words if w in t_lower)
        if pos_count > neg_count:
            sentiments.append(1)
        elif neg_count > pos_count:
            sentiments.append(-1)
        else:
            sentiments.append(0)
    return np.array(sentiments)


# ==========================================
# 🔬 3. ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ
# ==========================================

def research_analysis(
    candles_df, 
    news_df, 
    start_date=None, 
    end_date=None, 
    arch='resnet',
    export_path='research_output.json'
):
    print("\n" + "="*70)
    print("🔬 ИССЛЕДОВАТЕЛЬСКИЙ ПРОЕКТ: Новости + Голоса → Поведение рынка")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if start_date: start_date = pd.to_datetime(start_date)
    if end_date: end_date = pd.to_datetime(end_date)
    
    # Загрузка данных
    candles = safe_parse_datetime(candles_df, 'Open time')
    news = safe_parse_datetime(news_df, 'datetime')
    if start_date and end_date:
        candles = filter_by_date_range(candles, 'Open time', start_date, end_date)
        news = filter_by_date_range(news, 'datetime', start_date - pd.Timedelta(days=7), end_date)
    
    # Базовые метрики цены
    candles['price_change_pct'] = ((candles['Close'] - candles['Open']) / candles['Open']) * 100
    candles['volatility'] = candles['Close'].rolling(24, min_periods=1).std() / candles['Close'] * 100
    candles = candles.sort_values('Open time').reset_index(drop=True)
    news = news.sort_values('datetime').reset_index(drop=True)
    
        # === 1. Кодирование текста + сентимент ===
    txt_feats = encode_titles(news['title'].fillna('neutral').tolist(), arch=arch, out_dim=8, device=device)
    news = pd.concat([news.reset_index(drop=True), txt_feats], axis=1)
    
    # 🤖 BERT-based сентимент (с фоллбэком на keyword)
    news['bert_sentiment'] = analyze_sentiment_bert(news['title'].tolist(), device=device)
    
    # 🔧 Фоллбэк: если BERT не сработал, создаём keyword_sentiment
    if 'bert_sentiment' not in news.columns or news['bert_sentiment'].isna().all():
        print("   ⚠️ BERT-сентимент не создан — используем keyword-метод")
        news['bert_sentiment'] = analyze_sentiment_simple(news['title'])
    
    # === 2. Инженерия признаков голосов ===
    total_votes = news['positive_votes'] + news['negative_votes'] + 1
    news['vote_ratio'] = (news['positive_votes'] - news['negative_votes']) / total_votes
    news['vote_conf'] = np.log1p(news['positive_votes'] + news['negative_votes']) / np.log1p(total_votes + 9)
    news['engagement'] = np.log1p(total_votes)
    
    # === 3. Слияние с ценой ===
    # 🔧 ИСПРАВЛЕНО: используем 'bert_sentiment', а не 'keyword_sentiment'
    sentiment_col = 'bert_sentiment' if 'bert_sentiment' in news.columns else 'keyword_sentiment'
    base_cols = ['datetime', 'vote_ratio', 'vote_conf', 'engagement', sentiment_col] + [f'txt_{i}' for i in range(8)] + ['title']
    
    # 🔧 Дополнительная проверка: все колонки должны существовать
    missing_cols = [c for c in base_cols if c not in news.columns]
    if missing_cols:
        print(f"   ⚠️ Отсутствуют колонки в news: {missing_cols} — пропускаем их")
        base_cols = [c for c in base_cols if c in news.columns]
    
    merged = pd.merge_asof(candles, news[base_cols], left_on='Open time', right_on='datetime', direction='backward')
    merged = merged.dropna(subset=['vote_ratio']).reset_index(drop=True)

    if len(merged) < 50:
        print(f"⚠️ Мало данных: {len(merged)}")
        return None, merged
    
    results = {'n_samples': len(merged), 'date_range': f"{start_date} → {end_date}" if start_date else None}
    
    # ==========================================
    # 🔍 ГИПОТЕЗА 1: Реактивное голосование
    # "Люди голосуют ПОСЛЕ движения цены, а не ДО"
    # ==========================================
    print("\n🔍 ГИПОТЕЗА 1: Реактивность голосования")
    
    # Считаем движение цены ЗА 1 час ДО новости
    merged['prev_1h_return'] = candles['Close'].pct_change(1).shift(1)
    merged = merged.dropna(subset=['prev_1h_return'])
    
    # Корреляция: пред. движение цены → текущие голоса
    corr_reactive, p_reactive = stats.pearsonr(merged['prev_1h_return'], merged['vote_ratio'])
    print(f"   corr(пред. цена → голоса): {corr_reactive:+.4f} | p={p_reactive:.4f}")
    
    # Группируем: после роста цены люди чаще голосуют "вверх"?
    up_before = merged[merged['prev_1h_return'] > 0]['vote_ratio']
    down_before = merged[merged['prev_1h_return'] < 0]['vote_ratio']
    
    if len(up_before) > 10 and len(down_before) > 10:
        t_stat, t_p = ttest_ind(up_before, down_before, nan_policy='omit')
        print(f"   T-тест: после роста vs после падения → t={t_stat:+.3f}, p={t_p:.4f}")
        results['reactive_voting'] = {
            'corr': corr_reactive, 'p_value': p_reactive,
            'ttest_p': t_p, 'up_mean': up_before.mean(), 'down_mean': down_before.mean()
        }
        if p_reactive < 0.05 or t_p < 0.05:
            print("   ✅ Гипотеза подтверждена: голосование реактивно")
        else:
            print("   ❌ Гипотеза не подтверждена: голосование не связано с пред. движением")
    
    # ==========================================
    # 📊 ГИПОТЕЗА 2: Новости предсказывают волатильность
    # "Активность в новостях → рост неопределённости"
    # ==========================================
    print(f"\n📊 ГИПОТЕЗА 2: Влияние на волатильность")
    
    # Корреляция: активность в новостях → будущая волатильность
    for lag in [1, 2, 4, 12, 24, 48, 360, 720]:
        merged[f'vol_lag_{lag}'] = merged['volatility'].shift(-lag)
        clean = merged.dropna(subset=['engagement', f'vol_lag_{lag}'])
        if len(clean) > 30:
            corr_vol, p_vol = stats.pearsonr(clean['engagement'], clean[f'vol_lag_{lag}'])
            print(f"   Лаг {lag}ч: corr(активность → волатильность) = {corr_vol:+.4f} | p={p_vol:.4f}")
            if lag == 1:
                results['volatility'] = {'corr_1h': corr_vol, 'p_1h': p_vol}
    
    # ==========================================
    # 🕐 ГИПОТЕЗА 3: Временные паттерны
    # "Когда публикуется больше новостей? Связь с активностью рынка"
    # ==========================================
    print(f"\n🕐 ГИПОТЕЗА 3: Временные паттерны")
    
    merged['hour'] = merged['datetime'].dt.hour
    merged['day_of_week'] = merged['datetime'].dt.dayofweek
    
    # Активность по часам
    hourly = merged.groupby('hour').agg({
        'engagement': 'mean',
        'price_change_pct': lambda x: x.abs().mean(),
        'volatility': 'mean'
    }).round(3)
    
    peak_hour = hourly['engagement'].idxmax()
    print(f"   Пик активности новостей: {peak_hour}:00 (сред. вовлечённость: {hourly.loc[peak_hour, 'engagement']:.2f})")
    
    # Корреляция: время суток → волатильность
    time_vol_corr = merged['hour'].corr(merged['volatility'])
    print(f"   corr(час суток → волатильность): {time_vol_corr:+.4f}")
    
    results['temporal'] = {
        'peak_hour': int(peak_hour),
        'hourly_stats': hourly.to_dict(),
        'time_vol_corr': time_vol_corr
    }
    
        # ==========================================
    # 🧠 ГИПОТЕЗА 4: Сентимент заголовков
    # ==========================================
    print(f"\n🧠 ГИПОТЕЗА 4: Сентимент заголовков")
    
    # 🔧 Определяем, какая колонка сентимента доступна
    if 'bert_sentiment' in merged.columns:
        sentiment_col = 'bert_sentiment'
        model_name = 'BERTweet'
    elif 'keyword_sentiment' in merged.columns:
        sentiment_col = 'keyword_sentiment'
        model_name = 'keyword'
    else:
        print(f"   ⚠️ Нет колонки сентимента в merged — пропускаем гипотезу 4")
        results['sentiment'] = None
        sentiment_col = None
    
    if sentiment_col:
        pos_titles = merged[merged[sentiment_col] == 1]
        neg_titles = merged[merged[sentiment_col] == -1]
        
        if len(pos_titles) >= 20 and len(neg_titles) >= 20:
            pos_move = pos_titles['price_change_pct'].mean()
            neg_move = neg_titles['price_change_pct'].mean()
            print(f"   [{model_name}] Позитив: Δ={pos_move:+.3f}% (n={len(pos_titles)})")
            print(f"   [{model_name}] Негатив: Δ={neg_move:+.3f}% (n={len(neg_titles)})")
            
            t_stat, t_p = ttest_ind(pos_titles['price_change_pct'], neg_titles['price_change_pct'], nan_policy='omit')
            print(f"   T-тест: t={t_stat:+.3f}, p={t_p:.4f}")
            results['sentiment'] = {
                'model': model_name,
                'pos_mean': pos_move, 'neg_mean': neg_move,
                'ttest_p': t_p, 'n_pos': len(pos_titles), 'n_neg': len(neg_titles)
            }
        else:
            print(f"   ⚠️ Мало данных: позитив={len(pos_titles)}, негатив={len(neg_titles)}")
            results['sentiment'] = {'error': 'insufficient_data'}
    
    # ==========================================
    # 🤖 ML: Важность признаков
    # ==========================================
    print(f"\n🤖 Важность признаков (исследовательский взгляд)")
    
    # 🔧 Динамический выбор колонки сентимента
    sentiment_feature = 'bert_sentiment' if 'bert_sentiment' in merged.columns else ('keyword_sentiment' if 'keyword_sentiment' in merged.columns else None)
    
    base_features = ['vote_ratio', 'vote_conf', 'engagement']
    if sentiment_feature:
        base_features.append(sentiment_feature)
    base_features += [f'txt_{i}' for i in range(8)]
    if 'Volume' in merged.columns:
        base_features.append('Volume')
    
    features = [f for f in base_features if f in merged.columns]  # фильтрация несуществующих
    print(f"   Используемые признаки: {features}")
    
    # Очистка данных
    target_col = 'volatility'
    if target_col not in merged.columns:
        print(f"   ⚠️ Колонка '{target_col}' отсутствует — пропускаем ML")
        results['volatility_model'] = {'error': 'target_missing'}
        results['importance'] = {}
    else:
        ml_data = merged[features + [target_col]].copy()
        ml_data = ml_data.dropna(subset=features + [target_col])
        ml_data = ml_data[~np.isinf(ml_data[target_col])]
        
        if len(ml_data) < 50:
            print(f"   ⚠️ Мало данных для ML: {len(ml_data)}")
            results['volatility_model'] = {'error': 'insufficient_data'}
            results['importance'] = {}
        else:
            X = ml_data[features]
            y_vol = ml_data[target_col]
            
            # Временное разбиение
            split_idx = int(len(X) * 0.75)
            X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_te = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]
            
            rf = RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=8, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            
            r2_vol = rf.score(X_te, y_te)
            mae_vol = mean_absolute_error(y_te, rf.predict(X_te))
            print(f"   R²: {r2_vol:+.4f} | MAE: {mae_vol:.3f}%")
            results['volatility_model'] = {'r2_test': r2_vol, 'mae': mae_vol}
            
            # Permutation importance
            perm_imp = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=42, n_jobs=-1)
            imp_data = dict(zip(features, perm_imp.importances_mean))
            print("   📊 Важность признаков:")
            for feat, val in sorted(imp_data.items(), key=lambda x: -abs(x[1])):
                bar = '█' * int(min(30, abs(val) * 200))
                sign = '+' if val >= 0 else ''
                print(f"      {feat:20s}: {sign}{val:+.4f} {bar}")
            results['importance'] = imp_data

    # Гарантия наличия market_regime в merged (на случай, если merge_asof её отбросил)
    if 'market_regime' not in merged.columns and 'Close' in merged.columns:
        merged = calculate_market_regime(merged)
        print("   ⚠️ market_regime пересчитан для merged (отсутствовал после слияния)")
    
    # Экспорт результатов
    export_results(results, merged, export_path)
    
    return results, merged


# ==========================================
# 📈 4. ВИЗУАЛИЗАЦИЯ ДЛЯ ИССЛЕДОВАНИЙ
# ==========================================

def plot_research_results(merged, results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Исследовательский анализ: Новости + Голоса', fontsize=14, fontweight='bold', y=1.02)
    
    # 1. Реактивность: пред. цена → голоса
    ax = axes[0, 0]
    ax.scatter(merged['prev_1h_return'], merged['vote_ratio'], alpha=0.3, s=10, c='steelblue')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Δ цены за 1ч ДО новости (%)'); ax.set_ylabel('Vote Ratio')
    ax.set_title('Гипотеза 1: Реактивное голосование')
    ax.grid(True, alpha=0.3)
    
    # 2. Волатильность по времени суток
    ax = axes[0, 1]
    hourly = merged.groupby('hour')['volatility'].mean()
    ax.bar(hourly.index, hourly.values, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Час суток'); ax.set_ylabel('Средняя волатильность (%)')
    ax.set_title('Гипотеза 3: Волатильность по часам')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Сентимент заголовков
    ax = axes[0, 2]
    if 'sentiment' in results:
        sent = results['sentiment']
        labels = ['Позитивные', 'Негативные']
        values = [sent['pos_mean'], sent['neg_mean']]
        colors = ['green', 'red']
        bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.7)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel('Среднее Δ цены после новости (%)')
        ax.set_title('Гипотеза 4: Влияние сентимента')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:+.2f}%', ha='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Важность признаков (горизонтальный barplot)
    ax = axes[1, 0]
    imp = results.get('importance', {})
    if imp:
        feats, vals = zip(*sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        colors = ['steelblue' if 'txt' in f else 'coral' for f in feats]
        ax.barh(feats, vals, color=colors, edgecolor='black')
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel('Влияние на R² (волатильность)')
        ax.set_title('Важность признаков')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Временной ряд: активность новостей + волатильность
    ax = axes[1, 1]
    plot_data = merged.dropna(subset=['engagement', 'volatility']).tail(300)
    if len(plot_data) > 0:
        ax2 = ax.twinx()
        ax.plot(plot_data['Open time'], plot_data['engagement'], color='blue', label='Активность', alpha=0.7)
        ax2.plot(plot_data['Open time'], plot_data['volatility'], color='red', label='Волатильность', alpha=0.7)
        ax.set_ylabel('Вовлечённость (лог)', color='blue')
        ax2.set_ylabel('Волатильность (%)', color='red')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title('Динамика: Новости ↔ Волатильность')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    # 6. Распределение vote_ratio по рыночным режимам
    ax = axes[1, 2]
    
    if 'market_regime' in merged.columns:
        regimes = ['bull', 'bear', 'sideways']
        data_pairs = []
        for r in regimes:
            subset = merged[merged['market_regime'] == r]['vote_ratio'].dropna()
            if len(subset) >= 10:
                data_pairs.append((r.upper(), subset))
        
        if data_pairs:
            labels, data = zip(*data_pairs)
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], ['green', 'red', 'gray'][:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax.axhline(0, color='black', lw=0.5, ls='--')
            ax.set_ylabel('Vote Ratio')
            ax.set_title('Голоса по режимам рынка')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Недостаточно данных\nдля боксплота', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title('Голоса по режимам рынка')
            ax.grid(True, alpha=0.3)
    else:
        # ✅ Фоллбэк, если колонка потерялась при слиянии
        ax.text(0.5, 0.5, 'market_regime\nотсутствует в merged', 
               ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_title('Голоса по режимам рынка')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 🚀 ЗАПУСК ИССЛЕДОВАНИЯ
# ==========================================
try:
    candles = pd.read_csv(DATA_BTC_DAY_PATH, parse_dates=['Open time'])
    bull = pd.read_csv(DATA_BULL_PATH, parse_dates=['datetime'])
    bear = pd.read_csv(DATA_BEAR_PATH, parse_dates=['datetime'])
    
    for df in [bull, bear]:
        for c in ['positive_votes','negative_votes','important_votes']:
            if c not in df: df[c] = 0
    bull['negative_votes'] = 0
    bear['positive_votes'] = 0
    all_news = pd.concat([bull, bear], ignore_index=True)
    
    print(f"✓ Загружено: {len(candles)} свечей, {len(all_news)} новостей")
    
    # 🔧 НАСТРОЙКИ ИССЛЕДОВАНИЯ
    START_DATE = "2025-01-16"
    END_DATE = "2025-12-16"
    ARCHITECTURE = 'densenet'
    EXPORT_PATH = 'research_output.json'
    
    results, merged = research_analysis(
        candles, all_news,
        start_date=START_DATE, end_date=END_DATE,
        arch=ARCHITECTURE,
        export_path=EXPORT_PATH
    )
    
    if results:
        print("\n" + "🎯"*30)
        print("ИССЛЕДОВАТЕЛЬСКИЕ ВЫВОДЫ:")
        
        # 1. Реактивность
        if 'reactive_voting' in results:
            rv = results['reactive_voting']
            if rv['p_value'] < 0.05:
                print(f"✅ Голосование РЕАКТИВНО: corr={rv['corr']:+.4f}, p={rv['p_value']:.4f}")
                print(f"   → Люди голосуют ПОСЛЕ движения цены, а не прогнозируют")
            else:
                print(f"⚠️ Реактивность не подтверждена: p={rv['p_value']:.4f}")
        
        # 2. Волатильность
        if 'volatility' in results:
            vol = results['volatility']
            if vol.get('p_1h', 1) < 0.1:
                print(f"✅ Активность в новостях предсказывает волатильность: corr={vol['corr_1h']:+.4f}")
        
        # 3. Сентимент
        if 'sentiment' in results:
            sent = results['sentiment']
            if sent['ttest_p'] < 0.1:
                print(f"✅ Сентимент заголовков влияет на цену: позитив={sent['pos_mean']:+.2f}%, негатив={sent['neg_mean']:+.2f}%")
        
        # 4. Финальный вывод
        print(f"   • Экспортированные данные: {EXPORT_PATH}")
        
        print("🎯"*30)
        plot_research_results(merged, results)
        
except FileNotFoundError as e:
    print(f"❌ Файл не найден: {e}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback; traceback.print_exc()