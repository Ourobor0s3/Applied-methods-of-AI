# constans.py
from pathlib import Path

FILES1 = "btc_1h_data_2018_to_2025.csv"
FILES2 = "btc_1d_data_2018_to_2025.csv"
FILES3 = "btc_4h_data_2018_to_2025.csv"
FILES4 = "btc_15m_data_2018_to_2025.csv"
FILESBEAR = "bear.csv"
FILESBULL = "bull.csv"


# 1. Получаем папку, где лежит скрипт
SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Формируем путь к папке данных (БЕЗ проблем с экранированием!)
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Путь к локальному датасету
DATA_BTC_PATH = DATA_DIR / FILES1
DATA_BTC_DAY_PATH = DATA_DIR / FILES4
DATA_BEAR_PATH = DATA_DIR / FILESBEAR
DATA_BULL_PATH = DATA_DIR / FILESBULL