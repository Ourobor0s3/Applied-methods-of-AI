import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
from scipy import stats
from scipy.stats import bootstrap
import warnings
import json
from datetime import datetime
from constants import DATA_BEAR_PATH, DATA_BTC_DAY_PATH, DATA_BULL_PATH

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ==========================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def setup_date_axis(ax, interval_days=1):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval_days))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

def normalize_timezone(df, column_name):
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df[column_name]):
        if df[column_name].dt.tz is not None:
            df[column_name] = df[column_name].dt.tz_localize(None)
    return df

def safe_parse_datetime(df, column_name):
    df = df.copy()
    val = df[column_name].iloc[0] if len(df) > 0 else None
    
    if pd.api.types.is_datetime64_any_dtype(df[column_name]):
        return normalize_timezone(df, column_name)
    
    if pd.api.types.is_numeric_dtype(df[column_name]):
        if val > 1e12:
            df[column_name] = pd.to_datetime(df[column_name], unit='ms')
        else:
            df[column_name] = pd.to_datetime(df[column_name], unit='s')
        return normalize_timezone(df, column_name)
    
    try:
        df[column_name] = pd.to_datetime(df[column_name])
        return normalize_timezone(df, column_name)
    except Exception:
        raise ValueError(f"Не удалось распарсить дату в колонке {column_name}")

def calculate_sentiment_score(df, weight_important=1.5):
    """
    Расчёт сентимента с взвешиванием важных голосов
    """
    df = df.copy()
    
    # Взвешенные голоса
    df['weighted_positive'] = df['positive_votes'] + df.get('important_votes', 0) * weight_important
    df['weighted_negative'] = df['negative_votes'] + df.get('important_votes', 0) * weight_important * 0.5  # important в негативе весит меньше
    
    # Индекс настроений
    numerator = df['weighted_positive'] - df['weighted_negative']
    denominator = df['weighted_positive'] + df['weighted_negative'] + 1
    df['sentiment_score'] = numerator / denominator
    
    # Дополнительные метрики
    df['sentiment_magnitude'] = df['sentiment_score'].abs()
    df['trend_type'] = np.sign(df['sentiment_score'])
    df['confidence'] = (df['positive_votes'] + df['negative_votes']) / (df['positive_votes'] + df['negative_votes'] + 10)  # Уверенность в сентименте
    
    return df

def filter_by_date_range(df, date_column, start_date, end_date):
    df = df.copy()
    df = safe_parse_datetime(df, date_column)
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df.loc[mask].reset_index(drop=True)

def calculate_market_regime(df, price_column='Close', window=24):
    """
    Определение рыночного режима: 'bull', 'bear', 'sideways'
    """
    df = df.copy()
    df['returns'] = df[price_column].pct_change(window)
    
    # Пороги для определения режима (можно настроить)
    bull_threshold = 0.02   # +2% за окно
    bear_threshold = -0.02  # -2% за окно
    
    def classify(x):
        if pd.isna(x): return 'unknown'
        if x > bull_threshold: return 'bull'
        elif x < bear_threshold: return 'bear'
        else: return 'sideways'
    
    df['market_regime'] = df['returns'].apply(classify)
    return df

def bootstrap_confidence_interval(data1, data2, n_bootstrap=1000, confidence=0.95):
    """
    Расчёт доверительного интервала для корреляции через бутстрап
    """
    def corr_stat(d1, d2):
        return np.corrcoef(d1, d2)[0, 1]
    
    try:
        result = bootstrap(
            (data1.values, data2.values),
            corr_stat,
            confidence_level=confidence,
            n_resamples=n_bootstrap,
            method='percentile',
            random_state=42,
            paired=True
        )
        return result.confidence_interval
    except Exception as e:
        # Фолбэк: если бутстрап не работает, возвращаем NaN
        print(f"   ⚠️ Бутстрап не удался: {e}")
        return type('CI', (), {'low': np.nan, 'high': np.nan})()

def validate_random_forest(merged, features, target='price_change_pct', test_size=0.3):
    """
    🔍 Полная проверка RandomForest на переобучение
    """
    print("\n" + "="*70)
    print("🤖 ПРОВЕРКА RANDOM FOREST НА ПЕРЕОБУЧЕНИЕ")
    print("="*70)
    
    # Подготовка данных
    X = merged[features].dropna()
    y = merged.loc[X.index, target]
    
    if len(X) < 50:
        print("⚠️ Мало данных для валидации модели")
        return None
    
    # === 1. Train/Test Split ===
    print("\n📋 1. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
    
    # Важно: shuffle=False для временных рядов (не перемешиваем)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=42
    )
    
    print(f"   Обучающая выборка: {len(X_train)} записей ({100*(1-test_size):.0f}%)")
    print(f"   Тестовая выборка:  {len(X_test)} записей ({100*test_size:.0f}%)")
    print(f"   Период train: {X_train.index.min()} → {X_train.index.max()}")
    print(f"   Период test:  {X_test.index.min()} → {X_test.index.max()}")
    
    # === 2. Обучение модели ===
    print("\n🔧 2. ОБУЧЕНИЕ МОДЕЛИ")
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,        # Ограничиваем глубину для борьбы с переобучением
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # === 3. Оценка качества ===
    print("\n📊 3. ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    
    r2_train = rf.score(X_train, y_train)
    r2_test = rf.score(X_test, y_test)
    mae_train = mean_absolute_error(y_train, rf.predict(X_train))
    mae_test = mean_absolute_error(y_test, rf.predict(X_test))
    
    print(f"   R² Train: {r2_train:.4f}")
    print(f"   R² Test:  {r2_test:.4f}")
    print(f"   Разница:  {r2_train - r2_test:+.4f}")
    print(f"   MAE Train: {mae_train:.3f}%")
    print(f"   MAE Test:  {mae_test:.3f}%")
    
    # === 4. Диагностика переобучения ===
    print("\n⚠️  4. ДИАГНОСТИКА ПЕРЕОБУЧЕНИЯ")
    
    overfitting_status = []
    
    # Проверка 1: Большая разница R²
    r2_diff = r2_train - r2_test
    if r2_diff > 0.2:
        print(f"   🔴 ПЕРЕОБУЧЕНИЕ: Разница R² = {r2_diff:.4f} (> 0.2)")
        overfitting_status.append('overfitting')
    elif r2_diff > 0.1:
        print(f"   🟡 УМЕРЕННОЕ: Разница R² = {r2_diff:.4f} (0.1-0.2)")
        overfitting_status.append('moderate')
    else:
        print(f"   🟢 НОРМА: Разница R² = {r2_diff:.4f} (< 0.1)")
        overfitting_status.append('ok')
    
    # Проверка 2: Отрицательный R² на тесте
    if r2_test < 0:
        print(f"   🔴 КРИТИЧНО: R² на тесте отрицательный ({r2_test:.4f})")
        overfitting_status.append('negative_r2')
    else:
        print(f"   🟢 R² на тесте положительный ({r2_test:.4f})")
    
    # Проверка 3: Сравнение с Dummy-моделью (базовый уровень)
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    r2_dummy = dummy.score(X_test, y_test)
    
    print(f"\n   📏 Сравнение с базовой моделью (Dummy):")
    print(f"      Dummy R²: {r2_dummy:.4f}")
    print(f"      RF R²:    {r2_test:.4f}")
    print(f"      Улучшение: {r2_test - r2_dummy:+.4f}")
    
    if r2_test <= r2_dummy:
        print(f"   🔴 ВНИМАНИЕ: RF не лучше простой средней!")
        overfitting_status.append('worse_than_dummy')
    else:
        print(f"   🟢 RF лучше базовой модели")
    
    # === 5. Кросс-валидация ===
    print("\n🔄 5. КРОСС-ВАЛИДАЦИЯ (5 folds)")
    
    # Для временных рядов используем TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    
    print(f"   R² по фолдам: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   Средний R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Мин R²:       {cv_scores.min():.4f}")
    print(f"   Макс R²:      {cv_scores.max():.4f}")
    
    if cv_scores.std() > 0.15:
        print(f"   🟡 Высокая дисперсия между фолдами (нестабильная модель)")
    else:
        print(f"   🟢 Стабильная модель across folds")
    
    # === 6. Перестановочная важность признаков 🔥 ===
    print("\n🎯 6. ПЕРЕСТАНОВОЧНАЯ ВАЖНОСТЬ ПРИЗНАКОВ")
    
    # Перестановочная важность более надёжна, чем встроенная
    perm_importance = permutation_importance(
        rf, X_test, y_test, 
        n_repeats=30, 
        random_state=42,
        n_jobs=-1
    )
    
    print("   Важность признаков (перестановочная):")
    for i, feat in enumerate(features):
        imp_mean = perm_importance.importances_mean[i]
        imp_std = perm_importance.importances_std[i]
        bar = '█' * int(max(0, imp_mean) * 20)
        status = '✅' if imp_mean > 0.01 else '⚠️'
        print(f"   {status} {feat:20s}: {imp_mean:+.4f} ± {imp_std:.4f} {bar}")
    
    # === 7. Кривая обучения 🔥 ===
    print("\n📈 7. КРИВАЯ ОБУЧЕНИЯ")
    
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, 
        cv=tscv, 
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    
    print(f"   Размер выборки → R² Train | R² Test")
    for size, t, v in zip(train_sizes, train_mean, test_mean):
        gap = t - v
        marker = '🔴' if gap > 0.2 else '🟢'
        print(f"   {size:5d} → {t:+.3f} | {v:+.3f} {marker}")
    
    # === 8. Итоговый вердикт ===
    print("\n" + "="*70)
    print("🎯 ИТОГОВЫЙ ВЕРДИКТ")
    print("="*70)
    
    verdict_score = 0
    verdict_details = []
    
    if r2_test > 0.1:
        verdict_score += 1
        verdict_details.append("✅ R² на тесте > 0.1")
    else:
        verdict_details.append("❌ R² на тесте < 0.1")
    
    if r2_diff < 0.1:
        verdict_score += 1
        verdict_details.append("✅ Нет сильного переобучения")
    else:
        verdict_details.append("❌ Признаки переобучения")
    
    if r2_test > r2_dummy:
        verdict_score += 1
        verdict_details.append("✅ Лучше базовой модели")
    else:
        verdict_details.append("❌ Не лучше базовой модели")
    
    if cv_scores.mean() > 0:
        verdict_score += 1
        verdict_details.append("✅ Кросс-валидация положительная")
    else:
        verdict_details.append("❌ Кросс-валидация отрицательная")
    
    # Вывод вердикта
    print(f"\n   Оценка: {verdict_score}/4\n")
    
    for detail in verdict_details:
        print(f"   {detail}")
    
    print("\n   📋 РЕКОМЕНДАЦИЯ:")
    if verdict_score >= 3:
        print("   🟢 МОДЕЛЬ НАДЁЖНА — можно использовать для предсказаний")
    elif verdict_score == 2:
        print("   🟡 МОДЕЛЬ ТРЕБУЕТ ОСТОРОЖНОСТИ — нужна дополнительная валидация")
    else:
        print("   🔴 МОДЕЛЬ НЕНАДЁЖНА — не использовать для трейдинга")
    
    # === 9. Визуализация ===
    plot_model_validation(rf, X_test, y_test, features, perm_importance, 
                         train_sizes, train_mean, test_mean, r2_train, r2_test)
    
    # Возврат результатов
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'r2_dummy': r2_dummy,
        'r2_diff': r2_diff,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'permutation_importance': dict(zip(features, perm_importance.importances_mean)),
        'verdict_score': verdict_score,
        'overfitting_status': overfitting_status
    }

def plot_model_validation(rf, X_test, y_test, features, perm_importance,
                         train_sizes, train_mean, test_mean, r2_train, r2_test):
    """Визуализация валидации модели"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Валидация RandomForest: Проверка на переобучение', fontsize=14, fontweight='bold')
    
    # === 1. Фактические vs Предсказанные значения ===
    ax = axes[0, 0]
    y_pred = rf.predict(X_test)
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='black', s=30)  # ✅ scatter использует edgecolors (множ.)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Идеальное предсказание')
    ax.set_xlabel('Фактическое изменение цены (%)')
    ax.set_ylabel('Предсказанное изменение цены (%)')
    ax.set_title(f'Тестовая выборка (R² = {r2_test:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === 2. Важность признаков (сравнение) ===
    ax = axes[0, 1]
    built_in_imp = rf.feature_importances_
    perm_imp = perm_importance.importances_mean
    
    x = np.arange(len(features))
    width = 0.35
    
    # ✅ bar использует edgecolor (ед.)
    bars1 = ax.bar(x - width/2, built_in_imp, width, label='Встроенная', alpha=0.7, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, perm_imp, width, label='Перестановочная', alpha=0.7, color='coral', edgecolor='black')
    
    ax.set_xlabel('Признак')
    ax.set_ylabel('Важность')
    ax.set_title('Сравнение важности признаков')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # === 3. Кривая обучения ===
    ax = axes[1, 0]
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Train R²', linewidth=2)
    ax.plot(train_sizes, test_mean, 's-', color='red', label='Test R²', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_mean.std()/2, train_mean + train_mean.std()/2, alpha=0.15, color='blue')
    ax.fill_between(train_sizes, test_mean - test_mean.std()/2, test_mean + test_mean.std()/2, alpha=0.15, color='red')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Размер обучающей выборки')
    ax.set_ylabel('R²')
    ax.set_title('Кривая обучения')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === 4. Распределение ошибок (ИСПРАВЛЕНО) ===
    ax = axes[1, 1]
    errors = y_test - y_pred
    # ✅ hist использует edgecolor (ед.), а не edgecolors
    ax.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black', linewidth=1.2)
    ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Ноль ошибок')
    ax.axvline(errors.mean(), color='green', linewidth=2, label=f'Среднее: {errors.mean():.3f}%')
    ax.set_xlabel('Ошибка предсказания (%)')
    ax.set_ylabel('Частота')
    ax.set_title(f'Распределение ошибок (MAE = {mean_absolute_error(y_test, y_pred):.3f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. УЛУЧШЕННЫЙ АНАЛИЗ
# ==========================================

def analyze_correlation_advanced(
    candles_df, 
    news_df, 
    candle_interval_minutes=60,
    start_date=None, 
    end_date=None,
    export_results=True
):
    """
    Расширенный анализ с новыми методами
    """
    print("\n" + "="*70)
    print("🔬 РАСШИРЕННЫЙ АНАЛИЗ: Новости ↔ Цена (PRO версия)")
    print("="*70)
    
    # Парсинг дат
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
    
    # Подготовка данных
    candles_df = safe_parse_datetime(candles_df, 'Open time')
    news_df = safe_parse_datetime(news_df, 'datetime')
    news_df = calculate_sentiment_score(news_df)
    
    # Фильтрация по датам
    if start_date and end_date:
        print(f"\n📅 ФИЛЬТРАЦИЯ: {start_date.date()} → {end_date.date()}")
        candles_df = filter_by_date_range(candles_df, 'Open time', start_date, end_date)
        news_df = filter_by_date_range(news_df, 'datetime', start_date - pd.Timedelta(days=7), end_date)
    
    # Расчёт метрик цены
    candles_df['price_change_pct'] = ((candles_df['Close'] - candles_df['Open']) / candles_df['Open']) * 100
    candles_df = calculate_market_regime(candles_df)  # 🔥 НОВОЕ: рыночный режим
    candles_df = candles_df.sort_values('Open time').reset_index(drop=True)
    news_df = news_df.sort_values('datetime').reset_index(drop=True)
    
    # Слияние
    merged = pd.merge_asof(
        candles_df, 
        news_df[['datetime', 'sentiment_score', 'sentiment_magnitude', 'trend_type', 'confidence', 'title']], 
        left_on='Open time', 
        right_on='datetime',
        direction='backward'
    )
    merged = merged.dropna(subset=['sentiment_score']).reset_index(drop=True)
    
    if len(merged) < 30:
        print(f"⚠️ Мало данных: {len(merged)} записей")
        return None, merged
    
    results = {'date_range': {'start': str(start_date), 'end': str(end_date)} if start_date else None}
    
    # ==========================================
    # 📊 АНАЛИЗ 1: Базовая корреляция + бутстрап ДИ
    # ==========================================
    print("\n📋 1. БАЗОВАЯ КОРРЕЛЯЦИЯ + ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ")
    
    corr, p_value = stats.pearsonr(merged['sentiment_score'], merged['price_change_pct'])
    ci = bootstrap_confidence_interval(merged['sentiment_score'], merged['price_change_pct'])
    
    print(f"   Корреляция Пирсона: {corr:.4f}")
    print(f"   95% ДИ: [{ci.low:.4f}, {ci.high:.4f}]")
    print(f"   P-value: {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
    
    # Линейная регрессия
    X = merged[['sentiment_score']]
    y = merged['price_change_pct']
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    
    print(f"   R²: {r2:.4f} | Коэф: {model.coef_[0]:.4f}% / ед. сентимента")
    results['base'] = {'corr': corr, 'p_value': p_value, 'r2': r2, 'coef': model.coef_[0], 'ci': (ci.low, ci.high)}
    
    # ==========================================
    # 🔄 АНАЛИЗ 2: По рыночным режимам
    # ==========================================
    print(f"\n🔄 2. АНАЛИЗ ПО РЫНОЧНЫМ РЕЖИМАМ")
    
    for regime in ['bull', 'bear', 'sideways']:
        subset = merged[merged['market_regime'] == regime]
        if len(subset) >= 15:
            corr_r, p_r = stats.pearsonr(subset['sentiment_score'], subset['price_change_pct'])
            avg_change = subset['price_change_pct'].mean()
            sig = '✅' if p_r < 0.05 else '❌'
            print(f"   {regime.upper():8s}: corr={corr_r:+.4f} {sig} | ср. Δ цены: {avg_change:+.3f}% (n={len(subset)})")
            results[f'regime_{regime}'] = {'corr': corr_r, 'p_value': p_r, 'mean_change': avg_change, 'n': len(subset)}
    
    # ==========================================
    # ⏱️ АНАЛИЗ 3: Лаги с бутстрап ДИ
    # ==========================================
    print(f"\n⏱️  3. АНАЛИЗ ЛАГОВ (с доверительными интервалами)")
    
    lag_results = {}
    for lag in [1, 2, 4, 8]:
        merged[f'future_{lag}'] = merged['price_change_pct'].shift(-lag)
        clean = merged.dropna(subset=[f'future_{lag}', 'sentiment_score'])
        
        if len(clean) > 25:
            corr_lag, p_lag = stats.pearsonr(clean['sentiment_score'], clean[f'future_{lag}'])
            ci_lag = bootstrap_confidence_interval(clean['sentiment_score'], clean[f'future_{lag}'])
            lag_results[lag] = {'corr': corr_lag, 'p_value': p_lag, 'ci': (ci_lag.low, ci_lag.high)}
            
            sig = '✅' if p_lag < 0.05 else '❌'
            ci_str = f"[{ci_lag.low:+.3f}, {ci_lag.high:+.3f}]"
            print(f"   Лаг {lag} ({lag*candle_interval_minutes} мин): {corr_lag:+.4f} {sig} {ci_str}")
    
    if lag_results:
        best = max(lag_results.keys(), key=lambda k: abs(lag_results[k]['corr']))
        results['best_lag'] = {'lag': best, **lag_results[best]}
    
    results['lags'] = lag_results
    
    # ==========================================
    # 🎯 АНАЛИЗ 4: Влияние уверенности в сентименте
    # ==========================================
    print(f"\n🎯 4. ВЛИЯНИЕ УВЕРЕННОСТИ В СЕНТИМЕНТЕ")
    
    # Разделяем по уверенности (количество голосов)
    confidence_threshold = merged['confidence'].median()
    high_conf = merged[merged['confidence'] >= confidence_threshold]
    low_conf = merged[merged['confidence'] < confidence_threshold]
    
    if len(high_conf) >= 15:
        corr_hc, p_hc = stats.pearsonr(high_conf['sentiment_score'], high_conf['price_change_pct'])
        print(f"   Высокая уверенность (голосов ≥ медиана): corr={corr_hc:+.4f} {'✅' if p_hc<0.05 else '❌'} (n={len(high_conf)})")
    
    if len(low_conf) >= 15:
        corr_lc, p_lc = stats.pearsonr(low_conf['sentiment_score'], low_conf['price_change_pct'])
        print(f"   Низкая уверенность (голосов < медиана):  corr={corr_lc:+.4f} {'✅' if p_lc<0.05 else '❌'} (n={len(low_conf)})")
    
    results['confidence_split'] = {
        'high': {'corr': corr_hc if len(high_conf)>=15 else None, 'n': len(high_conf)},
        'low': {'corr': corr_lc if len(low_conf)>=15 else None, 'n': len(low_conf)}
    }
    
    # ==========================================
    # 🤖 АНАЛИЗ 5: Нелинейная модель (RandomForest)
    # ==========================================
    print(f"\n🤖 5. НЕЛИНЕЙНЫЙ АНАЛИЗ (RandomForest)")

    features = ['sentiment_score', 'sentiment_magnitude', 'confidence']
    if 'Volume' in merged.columns:
        features.append('Volume')

    X_rf = merged[features].dropna()
    y_rf = merged.loc[X_rf.index, 'price_change_pct']

    if len(X_rf) >= 50:
        # Базовая модель для сравнения
        rf_base = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_base.fit(X_rf, y_rf)
        y_pred_base = rf_base.predict(X_rf)
        
        r2_rf = r2_score(y_rf, y_pred_base)
        mae_rf = mean_absolute_error(y_rf, y_pred_base)
        
        print(f"   R² (на всех данных): {r2_rf:.4f}")
        print(f"   MAE: {mae_rf:.3f}%")
        print(f"   Важность признаков (встроенная):")
        for feat, imp in sorted(zip(features, rf_base.feature_importances_), key=lambda x: -x[1]):
            print(f"      • {feat:20s}: {imp:.3f}")
        
        results['random_forest'] = {
            'r2': r2_rf, 
            'mae': mae_rf, 
            'feature_importance': dict(zip(features, rf_base.feature_importances_))
        }
        
        # 🔥 ЗАПУСК ПРОВЕРКИ НА ПЕРЕОБУЧЕНИЕ
        rf_validation = validate_random_forest(merged, features)
        results['rf_validation'] = rf_validation
    else:
        print("   ⚠️ Мало данных для ML-модели")
    
    # ==========================================
    # ⚖️ АНАЛИЗ 6: Асимметрия реакции
    # ==========================================
    print(f"\n⚖️  6. АСИММЕТРИЯ: Влияние позитива vs негатива")
    
    # Сильный позитив и негатив
    strong_bull = merged[merged['sentiment_score'] > 0.5]
    strong_bear = merged[merged['sentiment_score'] < -0.5]
    
    if len(strong_bull) >= 10:
        mean_bull = strong_bull['price_change_pct'].mean()
        std_bull = strong_bull['price_change_pct'].std()
        print(f"   Сильный позитив (>{0.5}): Δ цены = {mean_bull:+.3f}% ± {std_bull:.2f}% (n={len(strong_bull)})")
    
    if len(strong_bear) >= 10:
        mean_bear = strong_bear['price_change_pct'].mean()
        std_bear = strong_bear['price_change_pct'].std()
        print(f"   Сильный негатив (<{-0.5}): Δ цены = {mean_bear:+.3f}% ± {std_bear:.2f}% (n={len(strong_bear)})")
    
    # Статистический тест на разницу средних
    if len(strong_bull) >= 10 and len(strong_bear) >= 10:
        t_stat, t_p = stats.ttest_ind(strong_bull['price_change_pct'], strong_bear['price_change_pct'])
        print(f"   Тест на разницу средних: t={t_stat:.3f}, p={t_p:.4f} {'✅ Асимметрия' if t_p<0.05 else '❌ Нет асимметрии'}")
        results['asymmetry'] = {'bull_mean': mean_bull, 'bear_mean': mean_bear, 'ttest_p': t_p}
    
    # ==========================================
    # 📈 АНАЛИЗ 7: Скользящая корреляция + смена режима
    # ==========================================
    print(f"\n📈 7. ДИНАМИКА КОРРЕЛЯЦИИ ВО ВРЕМЕНИ")
    
    window = 30
    merged['rolling_corr'] = merged['sentiment_score'].rolling(window).corr(merged['price_change_pct'])
    
    if merged['rolling_corr'].notna().any():
        rolling = merged['rolling_corr'].dropna()
        
        # Статистика
        print(f"   Средняя корреляция (окно {window}): {rolling.mean():+.4f} ± {rolling.std():.4f}")
        print(f"   Диапазон: [{rolling.min():+.4f}, {rolling.max():+.4f}]")
        
        # Поиск точек смены режима
        if len(rolling) > 20:
            diff = rolling.diff().abs()
            regime_changes = rolling[diff > diff.quantile(0.9)].index
            if len(regime_changes) > 0:
                print(f"   ⚠️ Обнаружено {len(regime_changes)} точек смены режима корреляции")
        
        results['rolling_stats'] = {'mean': rolling.mean(), 'std': rolling.std(), 'min': rolling.min(), 'max': rolling.max()}
    
    # ==========================================
    # 📊 ВИЗУАЛИЗАЦИЯ
    # ==========================================
    plot_advanced_results_v2(merged, results, start_date, end_date)
    
    # ==========================================
    # 💾 ЭКСПОРТ РЕЗУЛЬТАТОВ
    # ==========================================
    # if export_results:
    #     export_path = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
    #     # Конвертация для JSON
    #     export_data = {}
    #     for k, v in results.items():
    #         if isinstance(v, dict):
    #             export_data[k] = {}
    #             for k2, v2 in v.items():
    #                 if isinstance(v2, (np.floating, np.integer)):
    #                     export_data[k][k2] = float(v2)
    #                 elif isinstance(v2, tuple):
    #                     export_data[k][k2] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v2]
    #                 else:
    #                     export_data[k][k2] = v2
    #         else:
    #             export_data[k] = v
        
    #     with open(export_path, 'w', encoding='utf-8') as f:
    #         json.dump(export_data, f, indent=2, ensure_ascii=False)
    #     print(f"\n💾 Результаты экспортированы: {export_path}")
    
    return results, merged


def plot_advanced_results_v2(merged, results, start_date=None, end_date=None):
    """Улучшенная визуализация"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Анализ: Новости ↔ Цена {f"({start_date.date()}→{end_date.date()})" if start_date else ""}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # === 1. Scatter с цветовым кодированием по режиму рынка ===
    ax1 = plt.subplot(2, 3, 1)
    colors_regime = {'bull': 'green', 'bear': 'red', 'sideways': 'gray', 'unknown': 'lightgray'}
    for regime in merged['market_regime'].unique():
        subset = merged[merged['market_regime'] == regime]
        ax1.scatter(subset['sentiment_score'], subset['price_change_pct'], 
                   label=regime, alpha=0.5, s=20, c=colors_regime.get(regime, 'black'), edgecolors='black')
    
    # Линия регрессии
    X = merged[['sentiment_score']]
    y = merged['price_change_pct']
    model = LinearRegression()
    model.fit(X, y)
    x_line = np.linspace(X.min(), X.max(), 100)
    ax1.plot(x_line, model.predict(x_line), 'k--', linewidth=1.5, label=f'Тренд (coef={model.coef_[0]:.3f})')
    
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.set_xlabel('Сентимент')
    ax1.set_ylabel('Δ цены (%)')
    ax1.set_title('Зависимость по рыночным режимам')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # === 2. Корреляция по лагам с ДИ ===
    ax2 = plt.subplot(2, 3, 2)
    if 'lags' in results and results['lags']:
        lags = list(results['lags'].keys())
        corrs = [results['lags'][l]['corr'] for l in lags]
        ci_low = [results['lags'][l]['ci'][0] for l in lags]
        ci_high = [results['lags'][l]['ci'][1] for l in lags]
        
        ax2.errorbar(lags, corrs, yerr=[np.array(corrs)-np.array(ci_low), np.array(ci_high)-np.array(corrs)], 
                    fmt='o', capsize=5, color='blue', ecolor='gray', alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('Лаг (свечи)')
        ax2.set_ylabel('Корреляция ± 95% ДИ')
        ax2.set_title('Корреляция с учётом задержки')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # === 3. Boxplot: реакция цены по квантилям сентимента ===
    ax3 = plt.subplot(2, 3, 3)
    merged_clean = merged.dropna(subset=['sentiment_score', 'price_change_pct'])
    merged_clean['sentile'] = pd.qcut(merged_clean['sentiment_score'].rank(method='first'), 5, 
                                      labels=['🐻🐻','🐻','➖','🐂','🐂🐂'])
    
    data_box = [merged_clean[merged_clean['sentile']==q]['price_change_pct'].dropna() 
                for q in ['🐻🐻','🐻','➖','🐂','🐂🐂'] 
                if len(merged_clean[merged_clean['sentile']==q]) >= 5]
    labels_box = [q for q in ['🐻🐻','🐻','➖','🐂','🐂🐂'] 
                  if len(merged_clean[merged_clean['sentile']==q]) >= 5]
    
    if data_box:
        bp = ax3.boxplot(data_box, labels=labels_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['darkred', 'red', 'gray', 'green', 'darkgreen'][:len(data_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax3.set_ylabel('Δ цены (%)')
        ax3.set_title('Распределение реакции по сентименту')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # === 4. Скользящая корреляция во времени ===
    ax4 = plt.subplot(2, 3, 4)
    if 'rolling_corr' in merged.columns and merged['rolling_corr'].notna().any():
        plot_data = merged[merged['rolling_corr'].notna()]
        ax4.plot(plot_data['Open time'], plot_data['rolling_corr'], color='purple', linewidth=1.5)
        ax4.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax4.axhline(0.3, color='green', linewidth=0.5, linestyle=':', alpha=0.7)
        ax4.axhline(-0.3, color='red', linewidth=0.5, linestyle=':', alpha=0.7)
        ax4.fill_between(plot_data['Open time'], -0.3, 0.3, alpha=0.1, color='gray', label='Зона шума')
        ax4.set_xlabel('Дата')
        ax4.set_ylabel('Корреляция (окно 30)')
        ax4.set_title('Динамика корреляции')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        setup_date_axis(ax4, interval_days=14)
    
    # === 5. Важность признаков (RandomForest) ===
    ax5 = plt.subplot(2, 3, 5)
    if 'random_forest' in results and 'feature_importance' in results['random_forest']:
        feat_imp = results['random_forest']['feature_importance']
        features, importance = zip(*sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
        
        bars = ax5.barh(features, importance, color='steelblue', edgecolor='black')
        ax5.set_xlabel('Важность')
        ax5.set_title('Важность признаков (RF)')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Подписать значения
        for bar, val in zip(bars, importance):
            ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    
    # === 6. Сводная таблица метрик ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"📊 СВОДКА РЕЗУЛЬТАТОВ:\n" + "-"*35 + "\n\n"
    
    if 'base' in results:
        base = results['base']
        summary_text += f"Базовая корреляция: {base['corr']:+.4f}\n"
        summary_text += f"95% ДИ: [{base['ci'][0]:+.3f}, {base['ci'][1]:+.3f}]\n"
        summary_text += f"P-value: {base['p_value']:.4f} {'✅' if base['p_value']<0.05 else '❌'}\n"
        summary_text += f"R²: {base['r2']:.4f} ({base['r2']*100:.2f}%)\n\n"
    
    if 'best_lag' in results:
        bl = results['best_lag']
        summary_text += f"🎯 Лучший лаг: {bl['lag']} свечи\n"
        summary_text += f"   corr={bl['corr']:+.4f}, p={bl['p_value']:.4f}\n\n"
    
    if 'asymmetry' in results:
        asym = results['asymmetry']
        summary_text += f"⚖️ Асимметрия:\n"
        summary_text += f"   🐂 Позитив: {asym['bull_mean']:+.3f}%\n"
        summary_text += f"   🐻 Негатив: {asym['bear_mean']:+.3f}%\n"
        summary_text += f"   p={asym['ttest_p']:.4f} {'✅' if asym['ttest_p']<0.05 else '❌'}\n"
    
    ax6.text(0.05, 0.95, summary_text, fontsize=9, family='monospace', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


try:
    # Загрузка
    candles_df = pd.read_csv(DATA_BTC_DAY_PATH, parse_dates=['Open time'])
    bull_df = pd.read_csv(DATA_BULL_PATH, parse_dates=['datetime'])
    bear_df = pd.read_csv(DATA_BEAR_PATH, parse_dates=['datetime'])
    
    # гарантируем наличие колонок
    for col in ['positive_votes', 'negative_votes', 'important_votes']:
        if col not in bull_df:
            bull_df[col] = 0
        if col not in bear_df:
            bear_df[col] = 0

    # корректируем голоса по типу новости
    bull_df['negative_votes'] = 0
    bear_df['positive_votes'] = 0
    all_news = pd.concat([bull_df, bear_df], ignore_index=True)
    
    print(f"✓ Загружено: {len(candles_df)} свечей, {len(all_news)} новостей")
    
    # Настройки периода
    START_DATE = "2025-01-16"
    END_DATE = "2025-08-16"
    
    # 🚀 Запуск PRO-анализа
    results, merged_data = analyze_correlation_advanced(
        candles_df, 
        all_news, 
        candle_interval_minutes=60,
        start_date=START_DATE,
        end_date=END_DATE,
        export_results=True  # Сохранить результаты в JSON
    )
    
    # 💡 Финальная интерпретация
    if results:
        print("\n" + "🎯"*25)
        print("АВТО-ИНТЕРПРЕТАЦИЯ:")
        
        base = results.get('base', {})
        
        # Проверка доверительного интервала
        if base.get('ci'):
            ci_low, ci_high = base['ci']
            if ci_low < 0 < ci_high:
                print("⚠️ 95% ДИ включает 0 → корреляция может быть случайной")
            elif ci_low > 0:
                print("✅ Положительная связь статистически подтверждена")
            elif ci_high < 0:
                print("✅ Отрицательная связь статистически подтверждена")
        
        # Проверка рыночных режимов
        regimes = [k for k in results.keys() if k.startswith('regime_')]
        if regimes:
            best_regime = max(regimes, key=lambda k: abs(results[k].get('corr', 0)))
            corr_val = results[best_regime].get('corr', 0)
            regime_name = best_regime.replace('regime_', '').upper()
            if abs(corr_val) > 0.2:
                print(f"🔍 Сильная связь в режиме {regime_name}: corr={corr_val:+.4f}")
        
        # Рекомендация
        if base.get('r2', 0) < 0.05:
            print("💡 Новости объясняют <5% движения цены → используйте как ДОП. фильтр")
        elif base.get('r2', 0) > 0.15:
            print("💡 Новости объясняют >15% движения → можно использовать как основной сигнал")
        
        print("🎯"*25)
    
except FileNotFoundError as e:
    print(f"❌ Файл не найден: {e}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()