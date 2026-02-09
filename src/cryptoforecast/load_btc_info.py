import kagglehub
from pathlib import Path

# 1. Получаем папку, где лежит скрипт
SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Формируем путь к папке данных (БЕЗ проблем с экранированием!)
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download(
    "novandraanugrah/bitcoin-historical-datasets-2018-2024",
    output_dir=DATA_DIR)

print("Path to dataset files:", path)