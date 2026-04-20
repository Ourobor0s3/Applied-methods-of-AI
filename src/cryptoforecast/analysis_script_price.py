import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import warnings
import torch.nn.functional as F
from constants import DATA_BEAR_PATH, DATA_BTC_DAY_PATH, DATA_BULL_PATH
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 🔹 Импорт ClearML-логгера
from clearml_logger import create_logger, ENABLE_CLEARML

# 🔹 Инициализация логгера
logger = create_logger(
    project_name="CryptoForecast",
    task_name="analysis_price_v2",
    task_type="training",
    reuse_last_task_id=False,
    # Дополнительные параметры для авто-логирования:
    forecast_horizon=3,
    seq_len=10,
    hidden_dim=64,
    lr=0.002,
    batch_size=64,
    epochs=50
)


# ============================================
# 🔧 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ============================================

def load_and_prepare_data(candles_path, bull_path, bear_path, 
                          start_date="2025-01-16", end_date="2025-12-16"):
    """Загрузка и синхронизация свечей и новостей"""
    
    # Логируем параметры загрузки
    logger.connect_configuration({
        "start_date": start_date,
        "end_date": end_date,
        "candles_path": candles_path,
        "bull_path": bull_path,
        "bear_path": bear_path
    }, name="data_loading_config")
    
    candles = pd.read_csv(candles_path, parse_dates=['Open time'])
    bull = pd.read_csv(bull_path, parse_dates=['datetime'])
    bear = pd.read_csv(bear_path, parse_dates=['datetime'])
    
    for df in [bull, bear]:
        for c in ['positive_votes','negative_votes','important_votes']:
            if c not in df: df[c] = 0
    bull['negative_votes'] = 0
    bear['positive_votes'] = 0
    all_news = pd.concat([bull, bear], ignore_index=True)
    
    print(f"✓ Загружено: {len(candles)} свечей, {len(all_news)} новостей")
    
    candles = candles[(candles['Open time'] >= start_date) & 
                      (candles['Open time'] <= end_date)].copy()
    all_news = all_news[(all_news['datetime'] >= start_date) & 
                        (all_news['datetime'] <= end_date)].copy()
    
    # Feature Engineering: Свечи
    candles['returns'] = candles['Close'].pct_change()
    candles['log_returns'] = np.log(candles['Close'] / candles['Close'].shift(1))
    candles['volatility'] = candles['returns'].rolling(5).std()
    candles['volume_ma'] = candles['Volume'].rolling(5).mean()
    candles['volume_ratio'] = candles['Volume'] / candles['volume_ma']
    candles['price_range'] = (candles['High'] - candles['Low']) / candles['Close']
    
    candles['ema_12'] = candles['Close'].ewm(span=12, adjust=False).mean()
    candles['ema_26'] = candles['Close'].ewm(span=26, adjust=False).mean()
    candles['macd'] = candles['ema_12'] - candles['ema_26']
    
    delta = candles['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    candles['rsi'] = 100 - (100 / (1 + rs))
    
    # Feature Engineering: Новости
    all_news['sentiment_score'] = (
        all_news['positive_votes'] * 1.0 + 
        all_news['important_votes'] * 0.5 - 
        all_news['negative_votes'] * 0.8
    )
    
    all_news['date'] = all_news['datetime'].dt.date
    news_daily = all_news.groupby('date').agg({
        'sentiment_score': ['sum', 'mean', 'std'],
        'title': 'count'
    }).reset_index()
    news_daily.columns = ['date', 'sentiment_sum', 'sentiment_mean', 
                          'sentiment_std', 'news_count']
    news_daily['date'] = pd.to_datetime(news_daily['date'])
    
    # Объединение
    candles['date'] = candles['Open time'].dt.date
    candles['date'] = pd.to_datetime(candles['date'])
    
    df = candles.merge(news_daily, on='date', how='left')
    
    for col in ['sentiment_sum', 'sentiment_mean', 'sentiment_std', 'news_count']:
        df[col] = df[col].fillna(0)
    
    # Целевая переменная
    forecast_horizon = 3
    df['target'] = (df['Close'].shift(-forecast_horizon) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)
    
    print(f"✓ Итоговый датасет: {len(df)} строк, {df['target'].mean()*100:.1f}% положительных исходов")
    
    # Логируем статистику
    logger.report_scalar(title="dataset_stats", series="total_samples", value=len(df), iteration=0)
    logger.report_scalar(title="dataset_stats", series="positive_ratio_pct", 
                        value=df['target'].mean()*100, iteration=0)
    
    return df


# ============================================
# 🧠 2. МОДЕЛЬ (Liquid Neural Network)
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


class LiquidNewsTrader(nn.Module):
    def __init__(self, num_features, hidden_dim=32, news_dim=4, out_dim=1):
        super(LiquidNewsTrader, self).__init__()
        self.price_proj = nn.Linear(num_features, hidden_dim)
        self.news_proj = nn.Linear(news_dim, hidden_dim // 2)
        self.liquid = LiquidLayer(hidden_dim + hidden_dim//2, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, price_seq, news_seq):
        p_emb = self.price_proj(price_seq)
        n_emb = self.news_proj(news_seq)
        x = torch.cat([p_emb, n_emb], dim=-1)
        liquid_out = self.liquid(x)
        return self.readout(liquid_out[:, -1, :])


# ============================================
# 📦 3. DATASET
# ============================================

class TradingDataset(TorchDataset):
    def __init__(self, df, seq_length=20, target_col='target', 
                 feature_cols=None, news_cols=None):
        self.seq_length = seq_length
        
        if feature_cols is None:
            feature_cols = ['returns', 'volatility', 'volume_ratio', 
                           'price_range', 'macd', 'rsi', 'Close']
        if news_cols is None:
            news_cols = ['sentiment_sum', 'sentiment_mean', 
                        'sentiment_std', 'news_count']
        
        self.scaler_price = StandardScaler()
        self.scaler_news = StandardScaler()
        
        price_data = df[feature_cols].values
        news_data = df[news_cols].values
        
        split_idx = int(len(df) * 0.8)
        self.scaler_price.fit(price_data[:split_idx])
        self.scaler_news.fit(news_data[:split_idx])
        
        price_scaled = self.scaler_price.transform(price_data)
        news_scaled = self.scaler_news.transform(news_data)
        
        self.price_data = torch.tensor(price_scaled, dtype=torch.float32)
        self.news_data = torch.tensor(news_scaled, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.price_data) - self.seq_length
    
    def __getitem__(self, idx):
        end_idx = idx + self.seq_length
        return (
            self.price_data[idx:end_idx],
            self.news_data[idx:end_idx],
            self.targets[end_idx - 1]
        )


# ============================================
# 🚀 4. ОБУЧЕНИЕ
# ============================================

def train_model(df, epochs=100, batch_size=64, lr=0.003, seq_len=30):
    from sklearn.model_selection import TimeSeriesSplit
    
    # Логируем гиперпараметры
    logger.connect_configuration({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seq_len": seq_len,
        "hidden_dim": 64,
        "weight_decay": 1e-4,
        "early_stop_patience": 15
    }, name="training_hyperparams")
    
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(df))
    train_idx, test_idx = splits[-1]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    train_ds = TradingDataset(train_df, seq_length=seq_len)
    test_ds = TradingDataset(test_df, seq_length=seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    num_price_features = 7
    model = LiquidNewsTrader(
        num_features=num_price_features,
        hidden_dim=64,
        news_dim=4,
        out_dim=1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Обучение на {device}: {total_params:,} параметров")
    
    logger.report_text(f"Model architecture:\n{model}", iteration=0)
    logger.report_scalar(title="model_info", series="total_parameters", value=total_params, iteration=0)
    
    best_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for price_batch, news_batch, target_batch in train_loader:
            price_batch = price_batch.to(device)
            news_batch = news_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(price_batch, news_batch).squeeze()
            loss = criterion(logits, target_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for price_batch, news_batch, target_batch in test_loader:
                logits = model(
                    price_batch.to(device), 
                    news_batch.to(device)
                ).squeeze()
                loss = criterion(logits, target_batch.to(device))
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(target_batch.numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        acc = (preds_binary == np.array(all_targets)).mean()
        
        # 🔹 Логируем метрики
        logger.report_scalar(title="loss", series="train", value=train_loss, iteration=epoch+1)
        logger.report_scalar(title="loss", series="val", value=val_loss, iteration=epoch+1)
        logger.report_scalar(title="metrics", series="accuracy", value=acc, iteration=epoch+1)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc*100:.2f}%")
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_price_model.pt')
            logger.upload_artifact('best_model_checkpoint', 'best_price_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                logger.report_text(f"Early stopping triggered at epoch {epoch+1}", iteration=epoch+1)
                break
    
    # ============================================
    # 📊 ФИНАЛЬНАЯ ОЦЕНКА
    # ============================================
    
    if os.path.exists('best_price_model.pt'):
        model.load_state_dict(torch.load('best_price_model.pt', weights_only=True))
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for price_batch, news_batch, target_batch in test_loader:
            logits = model(
                price_batch.to(device), 
                news_batch.to(device)
            ).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(target_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    preds_binary = (all_preds > 0.5).astype(int)
    
    # Classification Report
    report = classification_report(all_targets, preds_binary, target_names=['Down', 'Up'], output_dict=True)
    logger.report_table(title="classification_report", series="test_results", 
                       iteration=0, table_plot=pd.DataFrame(report).T.round(4))
    
    print("\n" + "="*60)
    print("📈 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*60)
    print(classification_report(all_targets, preds_binary, target_names=['Down', 'Up']))
    
    # ROC-AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
        logger.report_scalar(title="final_metrics", series="roc_auc", value=auc, iteration=0)
        print(f"🎯 ROC-AUC: {auc:.4f} {'✅ Хорошо' if auc > 0.55 else '⚠️ Можно улучшить'}")
    except Exception as e:
        logger.report_text(f"AUC calculation error: {str(e)}", iteration=0)
    
    # Confusion Matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(all_targets, preds_binary)
    fig = cm_display.figure_
    logger.report_matplotlib_figure(title="confusion_matrix", series="test", 
                                   iteration=0, figure=fig)
    plt.close(fig)
    
    # Predictions Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_preds[:100], label='Pred', alpha=0.8)
    ax.plot(all_targets[:100], label='Actual', alpha=0.7)
    ax.legend(); ax.set_title('Predictions vs Actual (first 100 samples)')
    ax.grid(True, alpha=0.3)
    logger.report_matplotlib_figure(title="predictions_plot", series="test", 
                                   iteration=0, figure=fig)
    plt.close(fig)
    
    # Tau Analysis
    tau_values = torch.clamp(
        F.softplus(model.liquid.tau_raw) + 0.05, 
        min=0.05, max=5.0
    ).detach().cpu().numpy()
    
    tau_stats = {
        "min": float(tau_values.min()),
        "max": float(tau_values.max()),
        "mean": float(tau_values.mean()),
        "std": float(tau_values.std()),
        "spread_ratio": float(tau_values.max()/tau_values.min())
    }
    
    logger.report_scalar(title="liquid_tau", series="min", value=tau_stats["min"], iteration=0)
    logger.report_scalar(title="liquid_tau", series="max", value=tau_stats["max"], iteration=0)
    logger.report_scalar(title="liquid_tau", series="mean", value=tau_stats["mean"], iteration=0)
    logger.report_scalar(title="liquid_tau", series="spread_ratio", value=tau_stats["spread_ratio"], iteration=0)
    
    print(f"\n🔍 Liquid Layer — анализ постоянных времени (tau):")
    print(f"   Мин: {tau_stats['min']:.3f} | Макс: {tau_stats['max']:.3f} | Разброс: {tau_stats['spread_ratio']:.2f}x")
    
    # Tau Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tau_values, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Tau value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Liquid Layer Time Constants')
    ax.grid(True, alpha=0.3)
    logger.report_matplotlib_figure(title="tau_distribution", series="liquid_layer", 
                                   iteration=0, figure=fig)
    plt.close(fig)
    
    # Save predictions
    results_df = pd.DataFrame({
        'target': all_targets,
        'pred_prob': all_preds,
        'pred_binary': preds_binary
    })
    results_df.to_csv('price_predictions.csv', index=False)
    logger.upload_artifact('price_predictions', 'price_predictions.csv')
    
    return model, all_preds, all_targets


# ============================================
# ▶️ ЗАПУСК
# ============================================

if __name__ == "__main__":    
    print(f"🔹 ClearML: {'✅ ВКЛЮЧЁН' if logger.is_enabled else '❌ ВЫКЛЮЧЕН'}")
    
    df = load_and_prepare_data(
        DATA_BTC_DAY_PATH, DATA_BULL_PATH, DATA_BEAR_PATH,
        start_date="2025-01-16", end_date="2025-12-16"
    )
    
    model, predictions, targets = train_model(
        df, 
        epochs=50, 
        batch_size=64, 
        lr=0.002,
        seq_len=20
    )
    
    # Завершаем логгер (опционально)
    logger.mark_completed()
    logger.close()
    
    print("\n✅ Эксперимент завершён!")
    if logger.is_enabled:
        print(f"📊 Проверьте результаты в ClearML UI: {logger.get_task_id()}")