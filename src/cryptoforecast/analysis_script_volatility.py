import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import torch.nn.functional as F
from constants import DATA_BEAR_PATH, DATA_BTC_DAY_PATH, DATA_BULL_PATH
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 🔹 Импорт ClearML-логгера
from clearml_logger import create_logger

# 🔹 Инициализация логгера
logger = create_logger(
    project_name="CryptoForecast",
    task_name="analysis_volatility_v2",
    task_type="training",
    reuse_last_task_id=False,
    # Авто-логирование гиперпараметров:
    forecast_horizon=3,
    volatility_window=5,
    seq_len=25,
    hidden_dim=48,
    lr=0.0015,
    batch_size=32,
    epochs=80
)


# ============================================
# 🔧 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ============================================

def load_and_prepare_data(candles_path, bull_path, bear_path, 
                          start_date="2025-01-16", end_date="2025-12-16",
                          forecast_horizon=3, volatility_window=5):
    """Загрузка данных + создание признаков"""
    
    # Логируем параметры загрузки
    logger.connect_configuration({
        "start_date": start_date,
        "end_date": end_date,
        "forecast_horizon": forecast_horizon,
        "volatility_window": volatility_window,
        "candles_path": candles_path,
        "bull_path": bull_path,
        "bear_path": bear_path
    }, name="data_config")

    # ============================================
    # 🔹 1. ЗАГРУЗКА ДАННЫХ
    # ============================================
    
    candles = pd.read_csv(candles_path, parse_dates=['Open time'])
    bull = pd.read_csv(bull_path, parse_dates=['datetime'])
    bear = pd.read_csv(bear_path, parse_dates=['datetime'])
    
    # Нормализация колонок голосов
    for df in [bull, bear]:
        for c in ['positive_votes','negative_votes','important_votes']:
            if c not in df: df[c] = 0
    bull['negative_votes'] = 0
    bear['positive_votes'] = 0
    all_news = pd.concat([bull, bear], ignore_index=True)
    
    print(f"✓ Загружено: {len(candles)} свечей, {len(all_news)} новостей")
    
    # Фильтрация по датам
    candles = candles[(candles['Open time'] >= start_date) & 
                      (candles['Open time'] <= end_date)].copy()
    all_news = all_news[(all_news['datetime'] >= start_date) & 
                        (all_news['datetime'] <= end_date)].copy()
    
    # ============================================
    # 🔹 2. FEATURE ENGINEERING: Свечи
    # ============================================
    
    candles['returns'] = candles['Close'].pct_change()
    candles['log_returns'] = np.log(candles['Close'] / candles['Close'].shift(1))
    candles['volatility'] = candles['returns'].rolling(volatility_window).std()
    
    candles['price_range'] = (candles['High'] - candles['Low']) / candles['Close']
    candles['price_change'] = candles['Close'] - candles['Open']
    candles['price_change_pct'] = candles['price_change'] / candles['Open']
    
    candles['volume_ma'] = candles['Volume'].rolling(5).mean()
    candles['volume_ratio'] = candles['Volume'] / (candles['volume_ma'] + 1e-8)
    candles['volume_change'] = candles['Volume'].pct_change()
    candles['taker_buy_ratio'] = candles['Taker buy base asset volume'] / (candles['Volume'] + 1e-8)
    
    for span in [5, 10, 20]:
        candles[f'sma_{span}'] = candles['Close'].rolling(span).mean()
    candles['sma_5_20_diff'] = candles['sma_5'] - candles['sma_20']
    
    candles['ema_12'] = candles['Close'].ewm(span=12, adjust=False).mean()
    candles['ema_26'] = candles['Close'].ewm(span=26, adjust=False).mean()
    candles['macd'] = candles['ema_12'] - candles['ema_26']
    candles['macd_signal'] = candles['macd'].ewm(span=9, adjust=False).mean()
    
    delta = candles['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    candles['rsi'] = 100 - (100 / (1 + rs))
    
    for w in [3, 5, 10]:
        candles[f'volatility_{w}d'] = candles['returns'].rolling(w).std()
    
    candles['realized_vol'] = (candles['log_returns']**2).rolling(5).sum()**0.5
    candles['vol_trend'] = candles['volatility'] - candles['volatility'].shift(5)
    
    # ============================================
    # 🔹 3. FEATURE ENGINEERING: Новости
    # ============================================
    
    all_news['sentiment_score'] = (
        all_news['positive_votes'] * 1.0 + 
        all_news['important_votes'] * 0.5 - 
        all_news['negative_votes'] * 0.8
    )
    all_news['sentiment_abs'] = all_news['sentiment_score'].abs()
    
    all_news['date'] = all_news['datetime'].dt.date
    news_daily = all_news.groupby('date').agg({
        'sentiment_score': ['sum', 'mean', 'std'],
        'sentiment_abs': ['mean', 'max'],
        'title': 'count'
    }).reset_index()
    news_daily.columns = ['date', 'sentiment_sum', 'sentiment_mean', 
                          'sentiment_std', 'sentiment_abs_mean', 
                          'sentiment_abs_max', 'news_count']
    news_daily['date'] = pd.to_datetime(news_daily['date'])
    
    # ============================================
    # 🔹 4. ОБЪЕДИНЕНИЕ ДАННЫХ
    # ============================================
    
    candles['date'] = candles['Open time'].dt.date
    candles['date'] = pd.to_datetime(candles['date'])
    
    df = candles.merge(news_daily, on='date', how='left')
    
    news_cols = ['sentiment_sum', 'sentiment_mean', 'sentiment_std', 
                 'sentiment_abs_mean', 'sentiment_abs_max', 'news_count']
    for col in news_cols:
        df[col] = df[col].fillna(0)
    
    # ============================================
    # 🔹 5. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
    # ============================================
    
    df['target_volatility'] = df['returns'].shift(-forecast_horizon).rolling(volatility_window).std()
    df['target_volatility_log'] = np.log1p(df['target_volatility'].abs() * 100)
    
    # ============================================
    # 🔹 6. ОЧИСТКА ДАННЫХ
    # ============================================
    
    feature_cols = ['returns', 'volatility', 'volume_ratio', 'price_range', 
                    'macd', 'rsi', 'volatility_5d', 'volatility_10d', 'vol_trend',
                    'price_change_pct', 'taker_buy_ratio']
    
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"✓ Доступные признаки: {available_features}")
    
    df = df.dropna(subset=available_features + news_cols + ['target_volatility_log'])
    
    if len(df) > 100:
        vol_999 = df['target_volatility'].quantile(0.999)
        df = df[df['target_volatility'] <= vol_999]
    
    df = df.reset_index(drop=True)
    df = df[np.isfinite(df['target_volatility_log'])]
    
    print(f"✓ Итоговый датасет: {len(df)} строк")
    print(f"   Целевая волатильность: медиана={df['target_volatility'].median()*100:.2f}%, "
          f"95-й перцентиль={df['target_volatility'].quantile(0.95)*100:.2f}%")
    
    # 🔹 Логируем статистику датасета
    logger.report_scalar(title="dataset", series="total_samples", value=len(df), iteration=0)
    logger.report_scalar(title="dataset", series="target_median_pct", 
                        value=df['target_volatility'].median()*100, iteration=0)
    logger.report_scalar(title="dataset", series="target_p95_pct", 
                        value=df['target_volatility'].quantile(0.95)*100, iteration=0)
    
    return df, available_features, news_cols


# ============================================
# 📦 DATASET ДЛЯ РЕГРЕССИИ (Один класс)
# ============================================

class VolatilityDataset(Dataset):
    def __init__(self, df, seq_length=20, target_col='target_volatility_log', 
                 feature_cols=None, news_cols=None):
        self.seq_length = seq_length
        
        if feature_cols is None:
            feature_cols = ['returns', 'volatility', 'volume_ratio', 'price_range', 
                           'macd', 'rsi', 'volatility_5d', 'volatility_10d', 'vol_trend']
        if news_cols is None:
            news_cols = ['sentiment_sum', 'sentiment_mean', 'sentiment_std', 
                        'sentiment_abs_mean', 'sentiment_abs_max', 'news_count']
        
        self.feature_cols = feature_cols
        self.news_cols = news_cols
        
        self.scaler_price = RobustScaler()
        self.scaler_news = RobustScaler()
        self.scaler_target = RobustScaler()
        
        price_data = df[feature_cols].values
        news_data = df[news_cols].values
        target_data = df[[target_col]].values
        
        split_idx = int(len(df) * 0.8)
        self.scaler_price.fit(price_data[:split_idx])
        self.scaler_news.fit(news_data[:split_idx])
        self.scaler_target.fit(target_data[:split_idx])
        
        price_scaled = self.scaler_price.transform(price_data)
        news_scaled = self.scaler_news.transform(news_data)
        target_scaled = self.scaler_target.transform(target_data).flatten()
        
        self.price_data = torch.tensor(price_scaled, dtype=torch.float32)
        self.news_data = torch.tensor(news_scaled, dtype=torch.float32)
        self.targets = torch.tensor(target_scaled, dtype=torch.float32)
        self.targets_raw = torch.tensor(df[target_col].values, dtype=torch.float32)
        
    def __len__(self):
        return max(0, len(self.price_data) - self.seq_length)
    
    def __getitem__(self, idx):
        end_idx = idx + self.seq_length
        return (
            self.price_data[idx:end_idx],
            self.news_data[idx:end_idx], 
            self.targets[end_idx - 1],
            self.targets_raw[end_idx - 1]
        )


# ============================================
# 🧠 МОДЕЛИ
# ============================================

class LiquidLayer(nn.Module):
    def __init__(self, in_features, hidden_features, dt=0.1):
        super(LiquidLayer, self).__init__()
        self.dt = dt
        self.hidden_features = hidden_features
        self.W_in = nn.Linear(in_features, hidden_features)
        self.W_h = nn.Linear(hidden_features, hidden_features)
        self.tau_raw = nn.Parameter(torch.randn(hidden_features) * 0.5 + 1.5)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_features, device=x.device)
        outputs = []
        for t in range(seq_len):
            forcing = torch.tanh(self.W_in(x[:, t, :]) + self.W_h(h))
            tau = torch.clamp(F.softplus(self.tau_raw) + 0.05, min=0.05, max=5.0)
            dh = (-h / tau) + forcing
            h = h + self.dt * dh
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class LiquidVolatilityPredictor(nn.Module):
    def __init__(self, num_features, hidden_dim=32, news_dim=6, out_dim=1):
        super(LiquidVolatilityPredictor, self).__init__()
        
        self.price_proj = nn.Linear(num_features, hidden_dim)
        self.news_proj = nn.Linear(news_dim, hidden_dim // 2)
        self.liquid = LiquidLayer(hidden_dim + hidden_dim//2, hidden_dim)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, price_seq, news_seq):
        p_emb = self.price_proj(price_seq)
        n_emb = self.news_proj(news_seq)
        x = torch.cat([p_emb, n_emb], dim=-1)
        liquid_out = self.liquid(x)
        return self.readout(liquid_out[:, -1, :])


# ============================================
# 🚀 ОБУЧЕНИЕ
# ============================================

def train_volatility_model(df, feature_cols=None, news_cols=None, 
                          epochs=100, batch_size=64, lr=0.002, seq_len=20):
    """Обучение модели с логированием в ClearML"""
    
    from sklearn.model_selection import TimeSeriesSplit

    # Логируем гиперпараметры обучения
    logger.connect_configuration({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seq_len": seq_len,
        "hidden_dim": 48,
        "weight_decay": 1e-4,
        "early_stop_patience": 20,
        "feature_cols": feature_cols,
        "news_cols": news_cols
    }, name="training_config")
    
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(df))
    train_idx, test_idx = splits[-1]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    train_ds = VolatilityDataset(train_df, seq_length=seq_len, 
                                  feature_cols=feature_cols, news_cols=news_cols)
    test_ds = VolatilityDataset(test_df, seq_length=seq_len,
                                 feature_cols=feature_cols, news_cols=news_cols)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = LiquidVolatilityPredictor(
        num_features=len(feature_cols) if feature_cols else 7,
        hidden_dim=48,
        news_dim=len(news_cols) if news_cols else 4,
        out_dim=1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.report_scalar(title="model", series="total_parameters", value=total_params, iteration=0)
    logger.report_text(f"Model architecture:\n{model}", iteration=0)

    print(f"✓ Обучение на {device}: {total_params:,} параметров")
    
    best_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for price_batch, news_batch, target_scaled, target_raw in train_loader:
            price_batch = price_batch.to(device)
            news_batch = news_batch.to(device)
            target_scaled = target_scaled.to(device)
            
            optimizer.zero_grad()
            pred_scaled = model(price_batch, news_batch).squeeze()
            loss = criterion(pred_scaled, target_scaled)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Валидация
        model.eval()
        val_loss = 0
        all_preds_raw, all_targets_raw = [], []
        
        with torch.no_grad():
            for price_batch, news_batch, target_scaled, target_raw in test_loader:
                pred_scaled = model(price_batch.to(device), news_batch.to(device)).squeeze()
                loss = criterion(pred_scaled, target_scaled.to(device))
                val_loss += loss.item()
                
                pred_cpu = np.clip(pred_scaled.cpu().numpy().reshape(-1, 1), -10, 10)
                try:
                    pred_raw = train_ds.scaler_target.inverse_transform(pred_cpu).flatten()
                except:
                    pred_raw = pred_cpu.flatten()
                
                valid_mask = np.isfinite(pred_raw) & np.isfinite(target_raw.numpy())
                all_preds_raw.extend(pred_raw[valid_mask])
                all_targets_raw.extend(target_raw.numpy()[valid_mask])
        
        if len(all_preds_raw) < 10:
            continue
            
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        all_preds_raw_arr = np.array(all_preds_raw)
        all_targets_raw_arr = np.array(all_targets_raw)
        
        mae = mean_absolute_error(all_targets_raw_arr, all_preds_raw_arr)
        rmse = np.sqrt(mean_squared_error(all_targets_raw_arr, all_preds_raw_arr))
        r2 = r2_score(all_targets_raw_arr, all_preds_raw_arr)
        
        # 🔹 Логируем метрики обучения в ClearML
        logger.report_scalar(title="loss", series="train", value=train_loss, iteration=epoch+1)
        logger.report_scalar(title="loss", series="val", value=val_loss, iteration=epoch+1)
        logger.report_scalar(title="metrics", series="mae_pct", value=mae*100, iteration=epoch+1)
        logger.report_scalar(title="metrics", series="rmse_pct", value=rmse*100, iteration=epoch+1)
        logger.report_scalar(title="metrics", series="r2", value=r2, iteration=epoch+1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"MAE: {mae*100:.2f}% | R²: {r2:.3f}")
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'model_state': model.state_dict(),
                'scaler_target': train_ds.scaler_target,
                'feature_cols': train_ds.feature_cols if hasattr(train_ds, 'feature_cols') else None,
            }
            torch.save(checkpoint, 'best_volatility_model.pt')
            # 🔹 Загружаем артефакт в ClearML
            logger.upload_artifact('best_model_checkpoint', 'best_volatility_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.report_text(f"Early stopping at epoch {epoch+1}", iteration=epoch+1)
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break
    
    # ============================================
    # 📊 ФИНАЛЬНАЯ ОЦЕНКА
    # ============================================
    
    if os.path.exists('best_volatility_model.pt'):
        checkpoint = torch.load('best_volatility_model.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        target_scaler = checkpoint['scaler_target']
    
    model.eval()
    all_preds_raw, all_targets_raw = [], []
    
    with torch.no_grad():
        for price_batch, news_batch, target_scaled, target_raw in test_loader:
            pred_scaled = model(price_batch.to(device), news_batch.to(device)).squeeze()
            pred_cpu = np.clip(pred_scaled.cpu().numpy().reshape(-1, 1), -10, 10)
            try:
                pred_raw = target_scaler.inverse_transform(pred_cpu).flatten()
            except:
                pred_raw = pred_cpu.flatten()
            
            valid_mask = np.isfinite(pred_raw) & np.isfinite(target_raw.numpy())
            all_preds_raw.extend(pred_raw[valid_mask])
            all_targets_raw.extend(target_raw.numpy()[valid_mask])
    
    all_preds_raw = np.array(all_preds_raw, dtype=np.float64)
    all_targets_raw = np.array(all_targets_raw, dtype=np.float64)
    
    valid_final = np.isfinite(all_preds_raw) & np.isfinite(all_targets_raw)
    all_preds_raw = all_preds_raw[valid_final]
    all_targets_raw = all_targets_raw[valid_final]
    
    if len(all_preds_raw) < 50:
        print("❌ ОШИБКА: слишком мало валидных данных для оценки!")
        return model, [], [], 0.0, -1.0
    
    mae = mean_absolute_error(all_targets_raw, all_preds_raw)
    rmse = np.sqrt(mean_squared_error(all_targets_raw, all_preds_raw))
    r2 = r2_score(all_targets_raw, all_preds_raw)
    
    if len(all_preds_raw) > 1:
        pred_direction = np.sign(np.diff(all_preds_raw))
        true_direction = np.sign(np.diff(all_targets_raw))
        dir_acc = (pred_direction == true_direction).mean()
    else:
        dir_acc = 0.5
    
    vol_median = np.median(all_targets_raw)
    preds_class = (all_preds_raw > vol_median).astype(int)
    targets_class = (all_targets_raw > vol_median).astype(int)
    class_acc = (preds_class == targets_class).mean()

    # 🔹 Логируем финальные метрики
    logger.report_scalar(title="final_metrics", series="mae_pct", value=mae*100, iteration=0)
    logger.report_scalar(title="final_metrics", series="rmse_pct", value=rmse*100, iteration=0)
    logger.report_scalar(title="final_metrics", series="r2", value=r2, iteration=0)
    logger.report_scalar(title="final_metrics", series="directional_acc", value=dir_acc, iteration=0)
    logger.report_scalar(title="final_metrics", series="class_acc", value=class_acc, iteration=0)
    
    print("\n" + "="*70)
    print("📈 РЕЗУЛЬТАТЫ: ПРЕДСКАЗАНИЕ ВОЛАТИЛЬНОСТИ")
    print("="*70)
    print(f"🎯 MAE: {mae*100:.3f}%")
    print(f"🎯 RMSE: {rmse*100:.3f}%")
    print(f"📊 R²: {r2:.4f} {'✅ Хорошо' if r2 > 0.15 else '⚠️ Можно улучшить'}")
    logger.report_text(f"R² interpretation: {'Good' if r2 > 0.15 else 'Needs improvement'}", iteration=0)
    
    # 🔹 Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_targets_raw[::10]*100, all_preds_raw[::10]*100, alpha=0.5, s=10)
    ax.plot([all_targets_raw.min()*100, all_targets_raw.max()*100],
            [all_targets_raw.min()*100, all_targets_raw.max()*100], 
            'r--', label='Ideal')
    ax.set_xlabel('Actual Volatility (%)')
    ax.set_ylabel('Predicted Volatility (%)')
    ax.set_title(f'Prediction vs Actual (R²={r2:.3f})')
    ax.legend(); ax.grid(alpha=0.3)
    logger.report_matplotlib_figure(title="prediction_scatter", series="test", iteration=0, figure=fig)
    plt.close(fig)
    
    # 🔹 Time series plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(all_targets_raw[:200]*100, label='Actual', linewidth=1.5)
    ax.plot(all_preds_raw[:200]*100, label='Predicted', alpha=0.8)
    ax.axhline(y=np.median(all_targets_raw)*100, color='gray', linestyle='--', label='Median')
    ax.set_xlabel('Time step'); ax.set_ylabel('Volatility (%)')
    ax.set_title('Volatility Prediction: First 200 Test Samples')
    ax.legend(); ax.grid(alpha=0.3)
    logger.report_matplotlib_figure(title="prediction_timeseries", series="test", iteration=0, figure=fig)
    plt.close(fig)
    
    # 🔹 Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(all_targets_raw*100, bins=30, alpha=0.7, label='Actual', edgecolor='black')
    axes[0].set_xlabel('Volatility (%)'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('Actual Distribution'); axes[0].grid(alpha=0.3)
    
    axes[1].hist(all_preds_raw*100, bins=30, alpha=0.7, label='Predicted', edgecolor='black', color='orange')
    axes[1].set_xlabel('Volatility (%)'); axes[1].set_title('Predicted Distribution')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    logger.report_matplotlib_figure(title="volatility_distribution", series="test", iteration=0, figure=fig)
    plt.close(fig)
    
    # 🔹 Анализ tau
    tau_values = torch.clamp(F.softplus(model.liquid.tau_raw) + 0.05, min=0.05, max=5.0).detach().cpu().numpy()
    tau_stats = {
        "min": float(tau_values.min()), "max": float(tau_values.max()),
        "mean": float(tau_values.mean()), "std": float(tau_values.std()),
        "spread_ratio": float(tau_values.max()/tau_values.min())
    }
    for k, v in tau_stats.items():
        logger.report_scalar(title="liquid_tau", series=k, value=v, iteration=0)
    logger.report_text(f"Liquid Layer tau: min={tau_stats['min']:.3f}, max={tau_stats['max']:.3f}, spread={tau_stats['spread_ratio']:.2f}x", iteration=0)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(tau_values, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Tau value'); ax.set_ylabel('Frequency')
    ax.set_title('Liquid Layer Time Constants Distribution')
    ax.grid(alpha=0.3)
    logger.report_matplotlib_figure(title="tau_distribution", series="liquid_layer", iteration=0, figure=fig)
    plt.close(fig)
    
    # 🔹 Артефакты: предсказания
    results_df = pd.DataFrame({
        'target_volatility': all_targets_raw,
        'predicted_volatility': all_preds_raw,
        'error_pct': np.abs(all_targets_raw - all_preds_raw) / (all_targets_raw + 1e-8) * 100
    })
    results_df.to_csv('volatility_predictions.csv', index=False)
    logger.upload_artifact('test_predictions', 'volatility_predictions.csv')
    
    return model, all_preds_raw, all_targets_raw, dir_acc, r2


# ============================================
# ▶️ ЗАПУСК
# ============================================

if __name__ == "__main__":
    print(f"🔹 ClearML: {'✅ ВКЛЮЧЁН' if logger.is_enabled else '❌ ВЫКЛЮЧЕН'}")
    
    FORECAST_HORIZON = 3
    VOLATILITY_WINDOW = 5
    
    df, feature_cols, news_cols = load_and_prepare_data(
        DATA_BTC_DAY_PATH, DATA_BULL_PATH, DATA_BEAR_PATH,
        start_date="2025-01-16", end_date="2025-12-16",
        forecast_horizon=FORECAST_HORIZON,
        volatility_window=VOLATILITY_WINDOW
    )
    
    print(f"\n🔍 Используемые признаки:")
    print(f"   Price features ({len(feature_cols)}): {feature_cols}")
    print(f"   News features ({len(news_cols)}): {news_cols}")
    
    model, predictions, targets, dir_acc, r2 = train_volatility_model(
        df,
        feature_cols=feature_cols,
        news_cols=news_cols,
        epochs=80,
        batch_size=32,
        lr=0.0015,
        seq_len=25
    )

    logger.mark_completed()
    logger.close()
    
    if logger.is_enabled and logger.get_task_id():
        print(f"\n✅ Эксперимент завершён! Проверьте ClearML: {logger.get_task_id()}")
    
    # 🔹 Локальная визуализация (дублирует логи, но удобна для локального просмотра)
    if len(predictions) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        axes[0].plot(targets[:200] * 100, label='Реальная волатильность', linewidth=2)
        axes[0].plot(predictions[:200] * 100, label='Предсказание', alpha=0.8)
        axes[0].axhline(y=np.median(targets)*100, color='gray', linestyle='--', label='Медиана')
        axes[0].set_xlabel('Время (дни)'); axes[0].set_ylabel('Волатильность (%)')
        axes[0].set_title('Предсказание волатильности: первые 200 точек теста')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        
        axes[1].scatter(targets[::10] * 100, predictions[::10] * 100, alpha=0.5, s=10)
        axes[1].plot([targets.min()*100, targets.max()*100], 
                     [targets.min()*100, targets.max()*100], 
                     'r--', label='Идеальное предсказание')
        axes[1].set_xlabel('Реальная волатильность (%)')
        axes[1].set_ylabel('Предсказанная волатильность (%)')
        axes[1].set_title(f'Предсказание vs Реальность (R²={r2:.3f})')
        axes[1].legend(); axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('volatility_predictions.png', dpi=150, bbox_inches='tight')
        print("📊 График сохранён: volatility_predictions.png")
        plt.close(fig)  # Освобождаем память
        # plt.show()  # Раскомментируйте, если нужен интерактивный показ