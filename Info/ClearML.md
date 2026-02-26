# ClearML Info

## 📚 **Официальная документация**

- [ClearML Docs — Главная](https://clear.ml/docs/latest/docs/)

- [Datasets (включая CLI `clearml-data`)](https://clear.ml/docs/latest/docs/clearml_data/)

- [Agent (локальный и Colab)](https://clear.ml/docs/latest/docs/clearml_agent/)

- [Google Colab + Agent](https://clear.ml/docs/latest/docs/guides/ide/google_colab/)

- [Pipelines](https://clear.ml/docs/latest/docs/pipelines/)

- [Reports (GUI)](https://clear.ml/docs/latest/docs/webapp/webapp_reports/)

- [Logger](https://clear.ml/docs/latest/docs/fundamentals/logger/)

### 1. Установка и настройка

  Выполни в терминале:

  ```bash
  uv add clearml clearml-agent
  ```

 Но в данном репозитории данные пакеты уже добавлены, поэтому можно просто синхронизировать через

 ```bash
  uv sync
  ```

  Затем:

  ```bash
  clearml-init
  ```

  Следуй инструкциям: получи API Key и Secret на [странице настроек](https://app.clear.ml/settings/workspace-configuration).

  **Где хранится конфиг?**

- **Linux**: `~/clearml.conf`

- **Mac**: `$HOME/clearml.conf`

- **Windows**: `C:\Users\<твоё_имя>\.clearml\clearml.conf`

### 2. Запуск агента локально

  1. Открой **новый терминал**

  2. Запусти агент:

     ```bash
     clearml-agent daemon --queue default
     ```

     Агент будет ждать задачи в очереди `default`.

     !!! Если запускать вне виртуальной среды, то используй команду:

     ```bash

     uvx clearml-agent daemon --queue default

     ```

  3. Запусти файл  в **первом терминале**:

     ```bash
     python <name>.py
     ```

     Скрипт мгновенно завершится, но задача появится в очереди.

  4. Во **втором терминале** (где запущен агент) ты увидишь лог выполнения.

  🔗 Подробнее: [ClearML Agent Docs](https://clear.ml/docs/latest/docs/clearml_agent/)
