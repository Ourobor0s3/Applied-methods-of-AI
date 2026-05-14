# Эксперименты: цена и волатильность

Документ описывает устройство `analysis_script_price.py` и `analysis_script_volatility.py`,
где править настройки и какие артефакты попадают в ClearML / файловую систему.

---

## Структура файлов

| Файл | Назначение |
|------|------------|
| `src/cryptoforecast/experiment_config.py` | **Единая точка правки**: пути к CSV, даты, модели, гиперпараметры, ClearML, артефакты. |
| `src/cryptoforecast/clearml_logger.py` | Обёртка над ClearML; при отключённом ClearML (`ENABLE_CLEARML=False`) параметры из `create_logger(**kwargs)` доступны через `get_parameter`. |
| `src/cryptoforecast/models_factory.py` | Фабрика моделей: `liquid`, `resnet`, `densenet`, `xgboost`. |
| `src/cryptoforecast/nlp_features.py` | Эмбеддинги заголовков новостей (Ollama / Sentence Transformers) и агрегация по дням. |
| `src/cryptoforecast/utils.py` | Метрики (`calculate_metrics`, `should_stop_early`) + функции визуализации с поддержкой ClearML. |

---

## Запуск

```bash
# Из корня репозитория (с OMP для стабильности XGBoost)
$env:OMP_NUM_THREADS="4"; python src/cryptoforecast/analysis_script_volatility.py
$env:OMP_NUM_THREADS="4"; python src/cryptoforecast/analysis_script_price.py
```

---

## Поток данных (общий)

1. **Загрузка CSV** — OHLCV 15-минутные свечи + новости bull/bear с голосами
2. **Ценовые признаки** — returns, rolling volatility, volume_ratio, price_range, MACD, RSI
3. **Новостные признаки** — sentiment_score (weighted sum голосов), news_count;
   при `USE_NLP=True` — PCA-эмбеддинги заголовков через Ollama (`qwen3-embedding:0.6b`)
4. **Merge по дате** — дневная агрегация новостей присоединяется к свечам; пропуски = 0
5. **Таргет** — определяется скриптом (см. ниже)
6. **TimeSeriesSplit** (`time_series_cv_splits=3`) — последний сплит используется как train/test
7. **Dataset** — скользящее окно `sequence_length=25`; метка на последнем шаге
8. **Скейлеры** — `RobustScaler` fit на первых 80% train-части (утечка исключена)

---

## Скрипт волатильности (`analysis_script_volatility.py`)

**Задача:** регрессия — предсказать `target_volatility_log = log1p(|future_volatility| * 100)`
(лог-масштаб будущей волатильности доходности).

**Метрики:** `rmse`, `r2`, `mae` — выбор лучшей модели по `r2` (выше = лучше).

**PyTorch:** `HuberLoss` (устойчив к выбросам) на масштабированном таргете;
метрики считаются на сыром таргете (`targets_raw`).

**XGBoost:** обучение на сыром таргете из flatten-признаков;
early stopping через `eval_set`.

**Логируемые графики (ClearML + файл):**
- Кривые обучения (`Training/Loss` → `{model_name}`)
- Bar chart сравнения моделей (`Model Comparison` → `r2`)
- Scatter: предсказания vs реальность (`Predictions` → `{best_model}`)
- Распределение предсказаний (`Prediction Distribution` → `{best_model}`)
- Важность признаков — XGBoost (`Feature Importance` → `{best_model}`)

**Результат (14.05.2026, все 4 модели):**
- ✅ xgboost: **R²=0.5942** — лучшая модель
- ❌ liquid: R²=-31.49 (переобучение, ранняя остановка ep 8)
- ❌ resnet: R²=-34.38 (переобучение, ранняя остановка ep 8)
- ❌ densenet: R²=-47.67 (переобучение, ранняя остановка ep 8)

---

## Скрипт цены (`analysis_script_price.py`)

**Задача:** бинарная классификация — вырастет ли цена через `forecast_horizon` периодов.

**Метрики:** `roc_auc`, `accuracy` — выбор по `roc_auc` (выше = лучше).

**PyTorch:** `BCEWithLogitsLoss` → sigmoid на валидации.

**XGBoost:** предсказания клипируются в [0, 1], метрики считаются аналогично.

**Логируемые графики (ClearML + файл):**
- Кривые обучения (`Training/Loss` → `{model_name}`)
- Bar chart сравнения моделей (`Model Comparison` → `roc_auc`)
- Scatter предсказаний (`Predictions` → `{best_model}`)
- Распределение вероятностей (`Prediction Distribution` → `{best_model}`)
- ROC AUC кривая (`ROC Curve` → `{best_model}`)
- Матрица ошибок (`Confusion Matrix` → `{best_model}`)

**Результат (14.05.2026, все 4 модели):**
- ✅ liquid: **ROC=0.5314**, ACC=0.5220 — лучшая модель
- ⚠️ densenet: ROC=0.5266 (на уровне случайного)
- ❌ resnet: ROC=0.5148 (около случайного)
- ❌ xgboost: ROC=0.5141 (около случайного)

---

## Артефакты (директория `artifacts/`)

### Волатильность
```
best_volatility_xgboost_model.joblib   — лучшая модель (XGBoost)
best_volatility_scalers.joblib          — RobustScaler (price, news, target)
best_volatility_predictions.csv         — предсказания vs реальные значения
best_volatility_model_meta.json         — конфиг, метрики, пути
best_volatility_prediction_scatter.png   — scatter + residuals
best_volatility_prediction_distribution.png — гистограммы / линейный график
best_volatility_feature_importance.png   — топ-20 признаков (XGBoost)
volatility_model_comparison.png          — bar chart R² по моделям
{liquid|densenet|resnet}_training_losses.png — кривые обучения
```

### Цена
```
best_price_torch_model.pt               — лучшая модель (PyTorch: liquid)
best_price_scalers.joblib                — RobustScaler (price, news)
best_price_predictions.csv              — предсказания вероятностей
best_price_model_meta.json              — конфиг, метрики, пути
best_price_prediction_scatter.png       — scatter предсказаний
best_price_prediction_distribution.png  — распределение вероятностей
best_price_roc_curve.png               — ROC AUC кривая
best_price_confusion_matrix.png         — матрица ошибок
price_model_comparison.png             — bar chart ROC AUC по моделям
{liquid|densenet|resnet}_training_losses.png — кривые обучения
```

---

## ClearML: что логируется

| Что | Как | Когда |
|-----|-----|-------|
| Параметры конфигурации | `logger.connect_configuration()` | До загрузки данных |
| Число параметров модели | `logger.report_scalar("model", "{model}.param_count", ...)` | До начала обучения |
| Train/Val loss по эпохам | `logger.report_scalar("Training/Loss", "{model}.train_loss", ...)` | Каждую эпоху |
| Метрики моделей (скаляр) | `logger.report_scalar("model_comparison", ...)` | После обучения каждой модели |
| Таблица сравнения | `logger.report_table("model_comparison_table", ...)` | После цикла |
| Графики matplotlib | `logger.report_matplotlib_figure(...)` | После цикла (через `utils.plot_*`) |
| Артефакты (файлы) | `logger.upload_artifact(...)` | После сохранения модели |
| Текстовые логи | `logger.report_text(...)` | Ключевые события |

---

## Конфигурация запуска (`experiment_config.py`)

Ключевые параметры, влияющие на результат:

| Параметр | Влияние |
|----------|---------|
| `MODELS_TO_TEST` | Какие модели обучаются |
| `TASK_TYPE_PRICE` / `TASK_TYPE_VOLATILITY` | Тип задачи (classification / regression) |
| `DATA_START_DATE` / `DATA_END_DATE` | Период данных |
| `SEQUENCE_LENGTH=25` | Длина скользящего окна |
| `TRAINING_EPOCHS=30` | Макс. число эпох |
| `EARLY_STOPPING_PATIENCE=7` | Early stopping threshold |
| `LEARNING_RATE=0.0005` | Скорость обучения |
| `USE_NLP=False` | Включить NLP-эмбеддинги (нужен Ollama) |
| `ENABLE_CLEARML` | Вкл/выкл логирование в ClearML |

---

## Быстрая проверка работоспособности

```bash
$env:OMP_NUM_THREADS="4"; python src/cryptoforecast/analysis_script_volatility.py
# Должно напечатать: ClearML enabled=True | models_to_test=[...]
# Завершиться строками: Best model: xgboost | R2=0.XXXX
```

Если ClearML не инициализирован — измените `ENABLE_CLEARML = False` в `experiment_config.py`.