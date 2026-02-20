# create_task.py

from clearml import Dataset
from constants import DATA_BTC_PATH

# Загрузка последней версии
dataset = Dataset.get(
    dataset_project='CryptoForecast',
    dataset_name='BTC Hourly OHLCV Dataset'
)
local_path = dataset.get_local_copy()

# Загрузка в pandas
import pandas as pd
df = pd.read_csv(f"{DATA_BTC_PATH}")