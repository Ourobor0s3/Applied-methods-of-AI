# Эксперименты: цена и волатильность

Документ описывает, как устроены `analysis_script_price.py` и `analysis_script_volatility.py`, где править настройки и что попадает в ClearML.

## Где что лежит

| Файл | Назначение |
|------|------------|
| `src/cryptoforecast/experiment_config.py` | **Единая точка правки**: пути к CSV (`DATA_DIR`, `DATA_BTC_PATH`, …), даты, модели, обучение, ClearML, артефакты. |
| `src/cryptoforecast/clearml_logger.py` | Обёртка над ClearML; при отключённом ClearML параметры из `create_logger(**kwargs)` всё равно доступны через `get_parameter`. |
| `src/cryptoforecast/models_factory.py` | Создание моделей: `liquid`, `resnet`, `densenet`, `xgboost`. |
| `src/cryptoforecast/nlp_features.py` | Эмбеддинги заголовков новостей и агрегация по дням. |

Запуск (из корня репозитория, с учётом `PYTHONPATH` к `src/cryptoforecast` или из этой папки):

```bash
python analysis_script_price.py
python analysis_script_volatility.py
```

## Поток данных (общий для обоих скриптов)

1. **Загрузка** — дневные свечи (`Open time`, OHLCV) и два потока новостей (bull/bear) с голосами.
2. **Инженерия по свечам** — доходность, скользящая волатильность доходности, относительный диапазон дня, отношение объёма к скользящему среднему, MACD, RSI. Параметры окон заданы в `experiment_config.py` (`VOLATILITY_WINDOW` для таргета волатильности — отдельно; для price краткосрочная волатильность признака — `ROLLING_VOLATILITY_SHORT`).
3. **Новости** — дневной сентимент (взвешенная сумма голосов), счётчик новостей; при `use_nlp=True` — NLP-агрегаты по дням (`nlp_features`).
4. **Merge по календарной дате** — к свечам подмешивается дневная таблица новостей; пропуски по новостям заполняются нулями.
5. **Таргет** — см. разделы ниже.
6. **Разбиение** — `TimeSeriesSplit` с числом сплитов `time_series_cv_splits`; для обучения и отчёта берётся **последний** сплит (как и раньше): последний блок времени — валидация.
7. **Датасет** — скользящее окно длины `sequence_length`: на вход подаётся последовательность дней; метка — на **последнем** шаге окна (класс или значение волатильности в лог-масштабе).
8. **Скейлеры** — `RobustScaler` подгоняется на первых 80% строк **внутри** train- или test-части конкретного `DataFrame` после сплита (как в исходной реализации). Для волатильности таргет дополнительно масштабируется для обучения PyTorch; метрики на валидации для PyTorch считаются в **исходной шкале таргета** (`targets_raw`), для XGBoost — на сыром таргете из плоских признаков.

## Скрипт цены (`analysis_script_price.py`)

**Задача:** бинарная классификация — вырастет ли цена закрытия через `forecast_horizon` дней относительно текущего закрытия.

- **Модели PyTorch:** `BCEWithLogitsLoss`, на валидации вероятность — `sigmoid(logits)`, порог `classification_threshold` из конфига.
- **XGBoost:** признаки — конкатенация развёрнутых во времени ценовых и новостных каналов; предсказания клипуются в \([0, 1]\) для метрик в том же формате, что и вероятности.

**Метрики:** `accuracy`, `roc_auc` (AUC в скрипте защищён от одного класса на валидации).

**Выбор лучшей модели:** максимум `roc_auc` (ключ `selection_metric` в метаданных).

**Артефакты** (каталог `artifacts_dir`): `best_price_model_meta.json`, веса/модель, скейлеры, CSV предсказаний. В метаданных поля переименованы в единый стиль: `feature_columns`, `news_columns`, `nlp_columns`, `best_model_name`.

## Скрипт волатильности (`analysis_script_volatility.py`)

**Задача:** регрессия `target_volatility_log = log1p(|rolling_std(future_returns)| * 100)` — сглаженная будущая волатильность доходности в лог-масштабе.

- **PyTorch:** `HuberLoss` по **масштабированному** таргету; на валидации сравнение предсказаний с **сырым** `target_volatility_log` (как в исходном коде: сеть учится в scaled space, метрики — в исходных единицах таргета).
- **XGBoost:** обучение на сыром таргете из плоских признаков.

**Метрики:** `mae`, `rmse`, `r2`.

**Выбор лучшей модели:** максимум `r2`.

**Артефакты:** аналогично price, префикс `best_volatility_*`.

## Имена параметров и ClearML

При инициализации логгера в ClearML уходят параметры из `clearml_base_parameters()` в `experiment_config.py`:

- `models_to_test`, `use_nlp`, `nlp_sentence_model_name`
- `data_start_date`, `data_end_date`, `forecast_horizon`, `volatility_window`
- `sequence_length`, `time_series_cv_splits`, `training_epochs`, `batch_size`, `learning_rate`, `weight_decay`
- `hidden_dim_default`, `hidden_dim_liquid`, `huber_loss_delta`, `artifacts_dir`

Дополнительно подключаются конфиги с именами:

- `run` — фактический список `models_to_test` на запуск
- `data` — параметры загрузки/таргета для конкретного скрипта
- `training_<model_name>` — гиперпараметры одного прогона

**Заголовки (title) скаляров** унифицированы:

| Title | Содержание |
|-------|------------|
| `dataset` | `sample_count`, `nlp_feature_count`, для price ещё `positive_class_ratio_pct` |
| `metrics_epoch` | по эпохам: `<model>.accuracy`, `<model>.roc_auc` (price) или `<model>.mae`, `.rmse`, `.r2` (volatility) |
| `metrics_final` | итог по модели после цикла обучения |
| `model` | `<model>.parameter_count` |
| `model_comparison` | серии вида `roc_auc.<model>`, `accuracy.<model>` или `r2.<model>`, `mae.<model>`, `rmse.<model>` (функция `comparison_scalar_series`) |
| `model_comparison_table` | таблица со сводкой по всем моделям |

Так проще искать одни и те же метрики в UI ClearML для разных экспериментов.

## Практические замечания

- Порог классификации, имена файлов данных и пути артефактов меняются в `experiment_config.py`.
- Если меняется список NLP-колонок, переобучите модель и обновите артефакты — в `meta.json` сохраняется полный список `nlp_columns`.
- Старые `meta.json` с ключами `feature_cols` / `best_model_type` отличаются от новых имён; для инференса ориентируйтесь на актуальный формат из свежего прогона.
