# creation_dataset.py

import os
import pandas as pd
from clearml import Dataset
from constants import DATA_BTC_PATH


print("=" * 60)
print("Создание датасета ClearML")
print("=" * 60)

# Проверяем существование файла
if not os.path.exists(DATA_BTC_PATH):
    print(f"❌ Файл не найден: {DATA_BTC_PATH}")
    print("Пожалуйста, сначала загрузите данные с помощью load_btc_info.py")
    exit(1)

# Сначала проверьте подключение
try:
    from clearml import Task
    Task.init(project_name="CryptoForecast", task_name="connection_test")
    print("✅ Подключение к ClearML успешно!")
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    print("Пожалуйста, запустите 'clearml-init' для настройки")
    exit(1)

# 1. Создаем новый датасет
print("\n1️⃣ Создание нового датасета...")
dataset = Dataset.create(
    dataset_project="CryptoForecast", 
    dataset_name="BTC Hourly OHLCV Dataset"
)

# Добавляем теги
dataset.add_tags(["bitcoin", "btc", "ohlc", "hourly", "binance", "2018-2025"])

print(f"✅ Датасет создан с ID: {dataset.id}")

# 2. Добавляем файлы в датасет
print("\n2️⃣ Добавление файлов в датасет...")
dataset.add_files(path=DATA_BTC_PATH)
print(f"✅ Файл добавлен: {DATA_BTC_PATH}")

# 3. Добавляем метаданные и статистику
print("\n3️⃣ Анализ данных и добавление метаданных...")

try:
    # Загружаем датасет
    df = pd.read_csv(DATA_BTC_PATH)
    
    print(f"📊 Размер датасета: {df.shape[0]:,} строк × {df.shape[1]} колонок")
    print(f"📋 Колонки: {list(df.columns)}")
    
    # Преобразуем временные метки
    try:
        # Open time уже в формате строки 'YYYY-MM-DD HH:MM:SS'
        df['datetime'] = pd.to_datetime(df['Open time'])
        time_info = {
            "start": df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
            "end": df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S'),
            "total_hours": len(df),
            "years_covered": round((df['datetime'].max() - df['datetime'].min()).days / 365.25, 1)
        }
    except Exception as e:
        print(f"⚠️ Не удалось преобразовать временные метки: {e}")
        time_info = {
            "start": "N/A",
            "end": "N/A",
            "total_hours": len(df),
            "years_covered": "N/A"
        }
    
    # Основная статистика цен
    price_stats = {
        "open_min": df['Open'].min(),
        "open_max": df['Open'].max(),
        "close_min": df['Close'].min(),
        "close_max": df['Close'].max(),
        "high_max": df['High'].max(),
        "low_min": df['Low'].min(),
        "volume_total": df['Volume'].sum(),
        "avg_hourly_volume": df['Volume'].mean()
    }
    
    # Формируем информативный отчет
    info_text = f"""
₿ BITCOIN HOURLY OHLCV DATASET (Binance)
==========================================
Период: {time_info['start']} → {time_info['end']}
Охват: ~{time_info['years_covered']} лет ({time_info['total_hours']:,} часов)

💰 Ценовые экстремумы:
  • Минимальная цена открытия: ${price_stats['open_min']:,.2f}
  • Максимальная цена открытия: ${price_stats['open_max']:,.2f}
  • Минимальная цена закрытия: ${price_stats['close_min']:,.2f}
  • Максимальная цена закрытия: ${price_stats['close_max']:,.2f}
  • Абсолютный максимум (High): ${price_stats['high_max']:,.2f}
  • Абсолютный минимум (Low): ${price_stats['low_min']:,.2f}

📊 Объемы:
  • Суммарный объем: {price_stats['volume_total']:,.0f} BTC
  • Средний объем за час: {price_stats['avg_hourly_volume']:,.2f} BTC

📋 Структура данных:
  • Колонки: {', '.join(df.columns)}
  • Пропусков в данных: {df.isnull().sum().sum()}
"""
    
    dataset.get_logger().report_text(info_text, print_console=False)
    
    # Добавляем таблицу с основной статистикой
    stats = df.describe()
    dataset.get_logger().report_table(
        title="Статистика по колонкам", 
        series="Основные метрики", 
        table_plot=stats
    )
    
    # Добавляем информацию о корреляции цен
    price_cols = ['Open', 'High', 'Low', 'Close']
    if all(col in df.columns for col in price_cols):
        corr = df[price_cols].corr()
        dataset.get_logger().report_matrix(
            title="Корреляция ценовых колонок",
            series="Correlation",
            matrix=corr.values,
            xaxis=price_cols,
            yaxis=price_cols
        )
    
    print("✅ Метаданные успешно добавлены")
    print(f"   Период: {time_info['start']} → {time_info['end']}")
    print(f"   Макс. цена: ${price_stats['high_max']:,.2f}")
    
except Exception as e:
    print(f"❌ Ошибка при обработке данных: {e}")
    import traceback
    traceback.print_exc()
    
    # Корректное удаление датасета
    print("\n⚠️ Откатываем изменения...")
    try:
        dataset.finalize()
        # Удаляем через альтернативный метод
        from clearml import Dataset as DS
        DS.delete(dataset_id=dataset.id, force=True)
        print("✅ Датасет удален")
    except Exception as del_err:
        print(f"⚠️ Не удалось полностью удалить датасет: {del_err}")
        print(f"   ID для ручного удаления: {dataset.id}")
    exit(1)

# 4. Загрузка на сервер
print("\n4️⃣ Загрузка датасета на сервер ClearML...")
dataset.upload(output_url=None)  # output_url=None для локального хранения
print("✅ Датасет успешно загружен!")

# 5. Финализация
dataset.finalize()
print("✅ Датасет финализирован (read-only)")

# Итоговая информация
print("\n" + "=" * 60)
print("✅ ДАТАСЕТ ГОТОВ К ИСПОЛЬЗОВАНИЮ")
print("=" * 60)
print(f"📁 Название: {dataset.name}")
print(f"📦 ID: {dataset.id}")
print(f"🏷️  Теги: {', '.join(dataset.tags)}")
print(f"📈 Строк: {df.shape[0]:,}")
print(f"⏱️  Период: {time_info['start']} → {time_info['end']}")
print("=" * 60)