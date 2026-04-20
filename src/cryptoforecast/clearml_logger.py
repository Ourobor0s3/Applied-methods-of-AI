"""
clearml_logger.py
Модуль для интеграции с ClearML.
Если ClearML не установлен или отключён — все методы становятся no-op.
"""

from typing import Optional
import pandas as pd

# Флаг включения ClearML
ENABLE_CLEARML = True  # ← измените на False для отключения

class ClearMLLogger:
    """
    Универсальный логгер для ClearML с безопасным fallback.
    Все методы работают даже если ClearML отключён.
    """
    
    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: str = "training",
        **task_kwargs
    ):
        self.enabled = ENABLE_CLEARML
        self.task = None
        self.logger = None
        
        if self.enabled:
            try:
                # Маппинг строки в тип задачи ClearML
                from clearml import Task as ClearMLTask
                task_type_map = {
                    "training": ClearMLTask.TaskTypes.training,
                    "inference": ClearMLTask.TaskTypes.inference,
                    "testing": ClearMLTask.TaskTypes.testing,
                }
                clearml_task_type = task_type_map.get(task_type, ClearMLTask.TaskTypes.training)
                
                self.task = ClearMLTask.init(
                    project_name=project_name,
                    task_name=task_name,
                    task_type=clearml_task_type,
                    **task_kwargs
                )
                self.logger = self.task.get_logger()
                print(f"✅ ClearML подключён: {project_name}/{task_name}")
            except Exception as e:
                print(f"⚠️ Не удалось инициализировать ClearML: {e}")
                self.enabled = False
    
    # ─────────────────────────────────────────────────────────────
    # Конфигурация и параметры
    # ─────────────────────────────────────────────────────────────
    
    def connect_configuration(self, config: dict, name: str = "config"):
        """Логирование словаря конфигурации"""
        if self.enabled and self.task:
            try:
                self.task.connect_configuration(config, name=name)
            except: pass
    
    def connect_parameters(self, params: dict):
        """Логирование гиперпараметров"""
        if self.enabled and self.task:
            try:
                self.task.connect(params)
            except: pass
    
    # ─────────────────────────────────────────────────────────────
    # Метрики и графики
    # ─────────────────────────────────────────────────────────────
    
    def report_scalar(self, title: str, series: str, value: float, iteration: int):
        """Скалярная метрика (loss, accuracy, etc.)"""
        if self.enabled and self.logger:
            try:
                self.logger.report_scalar(title=title, series=series, value=value, iteration=iteration)
            except: pass
    
    def report_table(self, title: str, series: str, table_plot: pd.DataFrame, iteration: int = 0):
        """Таблица (например, classification report)"""
        if self.enabled and self.logger:
            try:
                self.logger.report_table(title=title, series=series, table_plot=table_plot, iteration=iteration)
            except: pass
    
    def report_matplotlib_figure(self, title: str, series: str, figure, iteration: int = 0):
        """Matplotlib график"""
        if self.enabled and self.logger:
            try:
                self.logger.report_matplotlib_figure(title=title, series=series, iteration=iteration, figure=figure)
            except: pass
    
    def report_text(self, text: str, iteration: int = 0):
        """Текстовый лог"""
        if self.enabled and self.logger:
            try:
                self.logger.report_text(text, iteration=iteration)
            except: pass
    
    # ─────────────────────────────────────────────────────────────
    # Артефакты
    # ─────────────────────────────────────────────────────────────
    
    def upload_artifact(self, name: str, artifact_path: str):
        """Загрузка файла как артефакта"""
        if self.enabled and self.task:
            try:
                self.task.upload_artifact(name, artifact_path)
            except: pass
    
    def get_artifact(self, name: str):
        """Получение артефакта (для загрузки в других скриптах)"""
        if self.enabled and self.task:
            try:
                return self.task.get_artifact(name)
            except: pass
        return None
    
    # ─────────────────────────────────────────────────────────────
    # Управление задачей
    # ─────────────────────────────────────────────────────────────
    
    def close(self):
        """Завершение задачи"""
        if self.enabled and self.task:
            try:
                self.task.close()
            except: pass
    
    def mark_completed(self):
        """Пометить задачу как завершённую"""
        if self.enabled and self.task:
            try:
                self.task.mark_completed()
            except: pass
    
    # ─────────────────────────────────────────────────────────────
    # Утилиты
    # ─────────────────────────────────────────────────────────────
    
    @property
    def is_enabled(self) -> bool:
        """Проверка, активен ли ClearML"""
        return self.enabled
    
    def get_task_id(self) -> Optional[str]:
        """Получить ID задачи (если доступен)"""
        if self.enabled and self.task:
            return getattr(self.task, 'id', None)
        return None


# ─────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────

def create_logger(
    project_name: str,
    task_name: str,
    task_type: str = "training",
    reuse_last_task_id: bool = False,
    **kwargs
) -> ClearMLLogger:
    """
    Фабричная функция для создания логгера.
    
    Пример:
        logger = create_logger(
            project_name="CryptoForecast",
            task_name="btc_news_v2",
            lr=0.002,
            epochs=50
        )
    """
    logger = ClearMLLogger(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        reuse_last_task_id=reuse_last_task_id
    )
    
    # Авто-логирование дополнительных параметров из kwargs
    if kwargs and logger.is_enabled:
        logger.connect_parameters(kwargs)
    
    return logger