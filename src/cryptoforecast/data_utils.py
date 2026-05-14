"""Утилиты для работы с данными: загрузка и создание датасета ClearML."""

import os
import pandas as pd
from experiment_config import DATA_BTC_PATH, DATA_DIR
import kagglehub

def download_data():
    """Загрузка данных BTC с Kaggle."""
    path = kagglehub.dataset_download("novandraanugrah/bitcoin-historical-datasets-2018-2024", output_dir=DATA_DIR)
    print(f"Data downloaded to: {path}")
    return path

def create_clearml_dataset():
    """Создание датасета ClearML."""
    from clearml import Dataset, Task
    
    if not os.path.exists(DATA_BTC_PATH):
        print(f"File not found: {DATA_BTC_PATH}")
        print("Run download_data() first")
        return
    
    try:
        Task.init(project_name="CryptoForecast", task_name="connection_test")
        print("ClearML connection OK")
    except Exception as e:
        print(f"ClearML error: {e}. Run 'clearml-init' first")
        return
    
    dataset = Dataset.create(dataset_project="CryptoForecast", dataset_name="BTC Hourly OHLCV Dataset")
    dataset.add_tags(["bitcoin", "btc", "ohlc", "hourly", "binance", "2018-2025"])
    
    dataset.add_files(path=DATA_BTC_PATH)
    
    df = pd.read_csv(DATA_BTC_PATH)
    df['datetime'] = pd.to_datetime(df['Open time'])
    
    info = f"BTC Hourly Dataset\n{df.shape[0]:,} rows\n{df['datetime'].min()} → {df['datetime'].max()}"
    dataset.get_logger().report_text(info, print_console=False)
    
    dataset.upload()
    dataset.finalize()
    print(f"Dataset created: {dataset.id}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            download_data()
        elif sys.argv[1] == "dataset":
            create_clearml_dataset()
    else:
        print("Usage: python data_utils.py [download|dataset]")