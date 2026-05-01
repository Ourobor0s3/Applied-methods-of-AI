# models_factory.py
"""
Фабрика моделей: Liquid, ResNet1D, DenseNet1D, XGBoost
Единый интерфейс для обучения и сравнения
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# ============================================
# 🔹 LIQUID LAYER
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
# 🔹 PyTorch: ResNet1D Block
# ============================================

class ResidualBlock1D(nn.Module):
    """1D Residual Block без изменения временной оси (stride=1)"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # ✅ Размеры всегда совпадают
        return self.relu(out)


class ResNetVolatilityPredictor(nn.Module):
    def __init__(self, num_features, news_dim, hidden_dim=64, n_blocks=2, out_dim=1):
        super().__init__()
        self.price_proj = nn.Linear(num_features, hidden_dim)
        self.news_proj = nn.Linear(news_dim, hidden_dim // 2) if news_dim > 0 else nn.Identity()
        
        in_channels = hidden_dim + (hidden_dim // 2 if news_dim > 0 else 0)
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        
        # Безопасные residual-блоки (не меняют T)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(hidden_dim) for _ in range(n_blocks)
        ])
        
        # Финальный пуллинг сжимает T до 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, price_seq, news_seq):
        p_emb = self.price_proj(price_seq).transpose(1, 2)  # [B, H, T]
        
        if isinstance(self.news_proj, nn.Linear):
            n_emb = self.news_proj(news_seq).transpose(1, 2)  # [B, H/2, T]
            x = torch.cat([p_emb, n_emb], dim=1)              # [B, 1.5H, T]
        else:
            x = p_emb
            
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)  # T остаётся прежним
        x = self.pool(x).squeeze(-1)  # [B, H]
        
        return self.readout(x)


# ============================================
# 🔹 PyTorch: DenseNet1D Block
# ============================================

class DenseBlock1D(nn.Module):
    """Dense Block с конкатенацией (как в DenseNet)"""
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=3):
        super(DenseBlock1D, self).__init__()
        
        layers = []
        curr_channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm1d(curr_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(curr_channels, growth_rate, kernel_size=kernel_size, 
                         padding=kernel_size//2, bias=False)
            ))
            curr_channels += growth_rate  # конкатенация увеличивает каналы
        
        self.layers = nn.ModuleList(layers)
        self.out_channels = curr_channels
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class DenseNetVolatilityPredictor(nn.Module):
    """DenseNet-style для временных рядов"""
    def __init__(self, num_features, news_dim, hidden_dim=32, growth_rate=16, 
                 num_layers=4, out_dim=1):
        super(DenseNetVolatilityPredictor, self).__init__()
        
        # Вход
        self.price_proj = nn.Linear(num_features, hidden_dim)
        self.news_proj = nn.Linear(news_dim, hidden_dim // 2) if news_dim > 0 else nn.Identity()
        
        in_channels = hidden_dim + (hidden_dim // 2 if news_dim > 0 else 0)
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # Dense block
        self.dense_block = DenseBlock1D(hidden_dim, growth_rate, num_layers)
        
        # Transition + readout
        final_channels = hidden_dim + growth_rate * num_layers
        self.transition = nn.Sequential(
            nn.BatchNorm1d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(final_channels, final_channels // 2, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.readout = nn.Sequential(
            nn.Linear(final_channels // 2, final_channels // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(final_channels // 4, out_dim)
        )
        
    def forward(self, price_seq, news_seq):
        p_emb = self.price_proj(price_seq).transpose(1, 2)
        
        if isinstance(self.news_proj, nn.Linear):
            n_emb = self.news_proj(news_seq).transpose(1, 2)
            x = torch.cat([p_emb, n_emb], dim=1)
        else:
            x = p_emb
        
        x = F.relu(self.conv_in(x))
        x = self.dense_block(x)
        x = self.transition(x).squeeze(-1)
        
        return self.readout(x)


# ============================================
# 🔹 XGBoost Wrapper
# ============================================

class XGBoostVolatilityPredictor:
    """Обёртка над XGBoost для регрессии волатильности"""
    
    def __init__(self, **xgb_params):
        if not _XGB_AVAILABLE:
            raise ImportError("Установите xgboost: pip install xgboost")
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # быстрее для больших данных
            'early_stopping_rounds': 20,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(xgb_params)
        self.params = default_params
        self.model = None
        self.feature_names = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray,
            feature_names: Optional[list] = None,
            verbose: bool = True):
        
        self.feature_names = feature_names
        
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Топ признаков по важности"""
        if not self.model or not self.feature_names:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    @property
    def best_iteration(self) -> int:
        return self.model.best_iteration if hasattr(self.model, 'best_iteration') else None


# ============================================
# 🔹 Factory Function
# ============================================

def create_model(model_type: str, num_features: int, news_dim: int, 
                 nlp_dim: int = 0, **kwargs):
    """Фабрика моделей. Все классы теперь объявлены в этом файле."""
    total_news = news_dim + nlp_dim
    
    if model_type == 'liquid':
        return LiquidVolatilityPredictor(
            num_features=num_features,
            hidden_dim=kwargs.get('hidden_dim', 48),
            news_dim=total_news,
            out_dim=1
        )
    
    elif model_type == 'resnet':
        return ResNetVolatilityPredictor(
            num_features=num_features, news_dim=total_news,
            hidden_dim=kwargs.get('hidden_dim', 64),
            n_blocks=kwargs.get('n_blocks', 2), out_dim=1
        )
    
    elif model_type == 'densenet':
        return DenseNetVolatilityPredictor(
            num_features=num_features, news_dim=total_news,
            hidden_dim=kwargs.get('hidden_dim', 32),
            growth_rate=kwargs.get('growth_rate', 16),
            num_layers=kwargs.get('num_layers', 4), out_dim=1
        )
    
    elif model_type == 'xgboost':
        return XGBoostVolatilityPredictor(**kwargs.get('xgb_params', {}))
    
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Доступно: liquid, resnet, densenet, xgboost")