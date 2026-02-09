# 1_task_with_logger.py

# Импортируем библиотеки для визуализации
import os  # библиотека для работы с операционными системами
from collections import Counter #  библиотека для подсчета

import joblib  # библиотека для сохранения и загрузки моделей
import matplotlib.pyplot as plt  # библиотека для создания статических графиков
import pandas as pd  # библиотека для работы с табличными данными
import plotly.graph_objects as go  # библиотека для интерактивной визуализации

# Импортируем основные компоненты ClearML
from clearml import Task  # основной класс для создания экспериментов в ClearML
from sklearn.decomposition import PCA  # анализ главных компонент
from sklearn.linear_model import (
    LogisticRegression,  # логистическая регрессия для бинарной классификации
)
from sklearn.metrics import (  # метрики для оценки качества модели
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,  # функция для разделения данных
)
from sklearn.pipeline import (
    Pipeline,  # конвейер для объединения трансформеров и модели
)

# Импортируем компоненты scikit-learn для машинного обучения
from sklearn.preprocessing import (
    PolynomialFeatures,  # класс для создания полиномиальных признаков
    StandardScaler,
)

# Инициализируем задачу в ClearML
# - project_name: имя проекта в ClearML, в котором будет зарегистрирован эксперимент
# - task_name: уникальное имя эксперимента
# - output_uri: позволяет сохранять результаты эксперимента в удалённом хранилище
task: Task = Task.init(
    project_name="Tutorial",                    # имя проекта в ClearML
    task_name="Simple Task with Polynomial Regression",  # имя эксперимента
    output_uri=True,                           # включаем сохранение результатов в удалённое хранилище
)
# Добавляем теги к задаче
task.add_tags(["polynomial-regression", "tutorial"])
# Получаем логгер для отправки метрик, графиков, текста и других артефактов в ClearML
logger = task.get_logger()

#####################
######## EDA ########
#####################

# Загружаем датасет
print("Загрузка датасета из файла...")  # выводим сообщение о начале загрузки данных
df = pd.read_csv("Test/data/synthetic_dataset.csv")  # загружаем табличные данные из CSV файла

# Выводим общую информацию о датасете
print("EDA: Общая информация о датасете")
print(df.info())
# Логируем основную информацию о датасете
dataset_info = {
    "Dataset shape": str(df.shape),
    "Number of features": str(df.shape[1]-1),  # исключаем целевую переменную
    "Number of samples": str(df.shape[0]),
    "Target variable": "target"
}
info_df = pd.DataFrame(list(dataset_info.items()), columns=["Property", "Value"])
logger.report_table(title="Dataset Statistics", series="Basic Info", iteration=0, table_plot=info_df)

# Выводим статистики по числовым признакам
print("EDA: Статистики по числовым признакам")
stats_df = df.describe()
print(stats_df)
logger.report_table(title="Dataset Statistics", series="Numerical Features", iteration=0, table_plot=stats_df)

# Разделяем признаки (X) и целевую переменную (y)
# Предположим, что последняя колонка - это целевая переменная (y), а остальные - признаки (X)
X = df.drop("target", axis=1)  # X содержит все признаки (все колонки кроме 'target')
y = df["target"]               # y содержит целевую переменную (колонка 'target')

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                    # признаки и целевая переменная
    test_size=0.2,           # 20% данных выделяется на тестовую выборку
    random_state=42          # фиксированное значение для воспроизводимости результата
)

# Выводим информацию о размерах выборок
print(f"Размер обучающей выборки: {X_train.shape}")  # количество строк и столбцов в обучающей выборке
print(f"Размер тестовой выборки: {X_test.shape}")    # количество строк и столбцов в тестовой выборке

# Создаем и логируем matplotlib график PCA scatter plot
print("Создаем и логируем matplotlib график PCA scatter plot")
pca = PCA(n_components=2)  # уменьшаем размерность до 2 компонент
X_pca = pca.fit_transform(X) # преобразуем исходные данные

# Создаем scatter plot с цветом точек в зависимости от класса
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)  # добавляем цветовую шкалу
plt.title("PCA Scatter Plot (2 Components)")  # заголовок графика
plt.xlabel("First Principal Component")  # подпись оси X
plt.ylabel("Second Principal Component")  # подпись оси Y

# Логируем matplotlib график
logger.report_matplotlib_figure(
    title="Dataset Visualization",  # заголовок графика в ClearML
    series="PCA Scatter Plot",     # серия данных
    figure=plt,                    # сам объект matplotlib фигуры
)

# Создаем и логируем корреляционную матрицу
print("Создаем и логируем корреляционную матрицу")
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix.columns)
plt.title("Correlation Matrix")
plt.tight_layout()

# Логируем корреляционную матрицу
logger.report_matplotlib_figure(
    title="Dataset Visualization",  # заголовок графика в ClearML
    series="Correlation Matrix",     # серия данных
    figure=plt,                    # сам объект matplotlib фигуры
)

# Определяем гиперпараметры для модели
# Эти параметры будут использоваться в процессе обучения и оптимизации
hyperparams = {
    # Диапазон степеней полинома для тестирования (от 1 до 5)
    "poly_degree_range": list(range(1, 5)),
    # Фиксированное значение для воспроизводимости результатов
    "random_state": 2,
    # Параметр регуляризации для логистической регрессии (обратно пропорционально силе регуляризации)
    "C": 1.0,
    # Максимальное количество итераций для сходимости
    "max_iter": 100,
}
# Подключаем гиперпараметры к задаче для автоматического логирования
task.connect(hyperparams)

###############################
##### Preprocessing stage #####
###############################

# Логируем информацию о preprocessing
print("Препроцессинг данных...")

# Масштабирование признаков

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Проверяем баланс классов
train_class_counts = Counter(y_train)
test_class_counts = Counter(y_test)

# Визуализируем распределение классов
fig, ax = plt.subplots()
ax.bar(list(train_class_counts.keys()), list(train_class_counts.values()))
ax.set_title("Train Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
logger.report_matplotlib_figure(title="Preprocessing Visualization", series="Train Class Distribution", figure=plt)

fig, ax = plt.subplots()
ax.bar(list(test_class_counts.keys()), list(test_class_counts.values()))
ax.set_title("Test Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
logger.report_matplotlib_figure(title="Preprocessing Visualization", series="Test Class Distribution", figure=plt)

#######################################
##### Hyperparameter tuning stage #####
#######################################

# Обучаем простую модель
print("Начинаем подбор гиперпараметров...")

# Обучение с логированием метрик на каждой эпохе (итерации)
# Используем гиперпараметры, определенные ранее
poly_degree_range = hyperparams["poly_degree_range"]  # диапазон степеней полинома для тестирования
train_accuracies = []  # список для хранения точностей на обучающей выборке
val_accuracies = []    # список для хранения точностей на валидационной выборке

# Проходим по каждому значению степени полинома из диапазона
for degree in poly_degree_range:
    # Создаем модель с текущими гиперпараметрами
    # Используем Pipeline для объединения PolynomialFeatures и LogisticRegression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),  # преобразование признаков в полиномиальные
        ('logistic', LogisticRegression(              # логистическая регрессия для бинарной классификации
            random_state=hyperparams["random_state"], # фиксированное значение для воспроизводимости
            C=hyperparams["C"],                       # параметр регуляризации
            max_iter=hyperparams["max_iter"],         # максимальное количество итераций для сходимости
        ))
    ])
    # Обучаем модель на обучающей выборке
    model.fit(X_train_scaled, y_train)

    # Получаем предсказания модели на обучающей и тестовой выборках
    y_train_pred = model.predict(X_train_scaled)  # предсказания на обучающей выборке
    y_test_pred = model.predict(X_test_scaled)    # предсказания на тестовой выборке

    # Вычисляем точность модели на обучающей и тестовой выборках
    train_acc = accuracy_score(y_train, y_train_pred)  # точность на обучающей выборке
    val_acc = accuracy_score(y_test, y_test_pred)      # точность на тестовой выборке

    # Вычисляем precision, recall и f1 для тестовой и тренировочной выборки
    val_precision = float(precision_score(y_test, y_test_pred))
    val_recall = float(recall_score(y_test, y_test_pred))
    val_f1 = float(f1_score(y_test, y_test_pred))
    train_precision = float(precision_score(y_train, y_train_pred))
    train_recall = float(recall_score(y_train, y_train_pred))
    train_f1 = float(f1_score(y_train, y_train_pred))

    # Добавляем полученные точности в соответствующие списки
    train_accuracies.append(train_acc)  # добавляем точность на обучающей выборке
    val_accuracies.append(val_acc)      # добавляем точность на тестовой выборке

    # Логируем скалярные значения точности для визуализации в ClearML
    logger.report_scalar(
        title="Accuracy",      # заголовок графика
        series="train",        # серия данных (обучающая выборка)
        value=float(train_acc), # значение точности
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    logger.report_scalar(
        title="Accuracy",      # заголовок графика
        series="validation",   # серия данных (валидационная/тестовая выборка)
        value=float(val_acc),  # значение точности
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    # Логируем скалярные значения precision, recall и f1 для визуализации в ClearML
    logger.report_scalar(
        title="Precision",        # заголовок графика
        series="train",
        value=train_precision,  # значение точности
        iteration=degree,
    )
    
    logger.report_scalar(
        title="Precision",     # заголовок графика
        series="validation",   # серия данных (валидационная/тестовая выборка)
        value=val_precision,       # значение precision
        iteration=degree,      # итерация (используется как ось X на графике)
    )
    
    logger.report_scalar(
        title="Recall",        # заголовок графика
        series="train",   # серия данных (валидационная/тесточная выборка)
        value=train_recall,      # значение recall
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    logger.report_scalar(
        title="Recall",        # заголовок графика
        series="validation",   # серия данных (валидационная/тестовая выборка)
        value=val_recall,          # значение recall
        iteration=degree,      # итерация (используется как ось X на графике)
    )
    
    logger.report_scalar(
        title="F1 Score",     # заголовок графика
        series="train",   # серия данных (валидационная/тесточная выборка)
        value=train_f1,        # значение precision
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    logger.report_scalar(
        title="F1 Score",      # заголовок графика
        series="validation",   # серия данных (валидационная/тестовая выборка)
        value=val_f1,              # значение f1
        iteration=degree,      # итерация (используется как ось X на графике)
    )

    # Выводим информацию о текущем прогрессе обучения в консоль
    print(
        f"polynomial_degree={degree}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
    )

# Находим лучшую точность на валидационной выборке и соответствующее значение степени полинома
best_val_accuracy = max(val_accuracies)  # наилучшая точность среди всех значений
best_poly_degree = poly_degree_range[val_accuracies.index(best_val_accuracy)]  # значение степени полинома, соответствующее лучшей точности

# Логируем финальную точность
final_accuracy = best_val_accuracy
logger.report_single_value(name="final_accuracy", value=final_accuracy)

# Логируем лучшие гиперпараметры
best_hyperparams = {
    "best_poly_degree": best_poly_degree,
    "random_state": hyperparams["random_state"],
    "C": hyperparams["C"],
    "max_iter": hyperparams["max_iter"],
}

# Логируем лучшие гиперпараметры в виде таблицы
best_hyperparams_df = pd.DataFrame(list(best_hyperparams.items()), columns=["Hyperparameter", "Value"])
logger.report_table(title="Best Hyperparameters", series="Tuned Values", iteration=0, table_plot=best_hyperparams_df)


# Логируем таблицу с результатами
# Создаем DataFrame с результатами для удобного отображения
print("Создаем DataFrame с результатами и logger.report_table")
results_df = pd.DataFrame(
    {
        "polynomial_degree": poly_degree_range,    # степень полинома
        "train_accuracy": train_accuracies,        # точность на обучающей выборке
        "validation_accuracy": val_accuracies,     # точность на валидационной выборке
    }
)
logger.report_table(
    title="Training Results",     # заголовок таблицы в ClearML
    series="Results",             # серия данных
    iteration=0,                 # итерация (для согласованности)
    table_plot=results_df,       # сам DataFrame с результатами
)

######################################
##### Final model training stage #####
######################################

print("Обучение финальной модели с лучшими гиперпараметрами...")
# Очень похоже на стандартный print, но с возможностями не выводить в консоль и установить уровень логирования
logger.report_text("Training final model with best hyperparameters...")

# Обучаем финальную модель с лучшими гиперпараметрами
final_model = Pipeline([
    ('poly', PolynomialFeatures(degree=best_poly_degree)),  # преобразование признаков в полиномиальные
    ('logistic', LogisticRegression(                        # логистическая регрессия для бинарной классификации
        random_state=hyperparams["random_state"],          # фиксированное значение для воспроизводимости
        C=hyperparams["C"],                                # параметр регуляризации
        max_iter=hyperparams["max_iter"],                  # максимальное количество итераций для сходимости
    ))
])
final_model.fit(X_train_scaled, y_train)  # обучаем модель с лучшими гиперпараметрами

# Получаем вероятности предсказаний для построения ROC-кривой
y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]  # вероятности для положительного класса
y_pred = final_model.predict(X_test_scaled) # получаем предсказания на тестовой выборке

##################################
##### Model evaluation stage #####
##################################

print("Оценка производительности модели...")

# Вычисляем значения для ROC-кривой
print("Вычисляем ROC curve и строим через plotly")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)  # ложноположительные и истинноположительные rates
roc_auc = auc(fpr, tpr)  # площадь под ROC-кривой

fig = go.Figure()  # создаем объект графика Plotly
# Добавляем ROC-кривую
fig.add_trace(
    go.Scatter(
        x=fpr,                   # ось X: ложноположительные rates
        y=tpr,                   # ось Y: истинноположительные rates
        mode='lines',            # режим отображения: линии
        name=f'ROC Curve (AUC = {roc_auc:.4f})', # название серии данных с AUC
    )
)
# Добавляем диагональную линию (random classifier)
fig.add_trace(
    go.Scatter(
        x=[0, 1],               # ось X: от 0 до 1
        y=[0, 1],               # ось Y: от 0 до 1
        mode='lines',           # режим отображения: линии
        name='Random Classifier',  # название случайного классификатора
        line=dict(dash='dash'), # пунктирная линия
    )
)
fig.update_layout(
    title="ROC Curve (Plotly)",  # заголовок графика
    xaxis_title="False Positive Rate",  # подпись оси X
    yaxis_title="True Positive Rate",   # подпись оси Y
    xaxis=dict(range=[0, 1]),           # диапазон оси X
    yaxis=dict(range=[0, 1]),           # диапазон оси Y
)

# Логируем Plotly график
logger.report_plotly(
    title="Training Results", # заголовок графика в ClearML
    series="ROC Curve",         # серия данных
    figure=fig,                 # сам объект графика
)

# Логируем confusion matrix
print("Вычисляем confusion matrix...")
cm = confusion_matrix(y_test, y_pred)  # вычисляем матрицу ошибок

logger.report_confusion_matrix(
    title="Confusion Matrix",  # заголовок матрицы ошибок в ClearML
    series="Validation",       # серия данных
    iteration=0,              # итерация (для согласованности)
    matrix=cm,                # сама матрица ошибок
    xaxis="Predicted",        # подпись оси X
    yaxis="Actual",           # подпись оси Y
)

# Вычисляем и логируем дополнительные метрики
val_precision = float(precision_score(y_test, y_pred))
val_recall = float(recall_score(y_test, y_pred))
val_f1 = float(f1_score(y_test, y_pred))

logger.report_single_value(name="precision", value=val_precision)
logger.report_single_value(name="recall", value=val_recall)
logger.report_single_value(name="f1_score", value=val_f1)

print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-score: {val_f1:.4f}")

# Логируем примеры предсказаний для отладки
print("Логгируем часть предсказаний")
predictions_df = pd.DataFrame(
    {"true_label": y_test, "predicted_label": y_pred, "prediction_proba": y_pred_proba}  # создаем DataFrame с истинными и предсказанными метками
)
logger.report_table(
    title="Sample Predictions",      # заголовок таблицы в ClearML
    series="Debug Samples",          # серия данных
    iteration=0,                    # итерация (для согласованности)
    table_plot=predictions_df.head(20),  # первые 20 строк таблицы с предсказаниями
)

##############################
##### Model saving stage #####
##############################

print("Сохранение модели...")

# Сохраняем финальную модель в файл
model_path = "models/polynomial.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(final_model, model_path, compress=True)  # сохраняем финальную модель с лучшими гиперпараметрами

# Завершаем задачу (опционально)
task.close()  # закрываем задачу в ClearML, чтобы указать, что эксперимент завершен
print(f"✅ Обучение завершено! Финальная точность: {final_accuracy:.4f}")  # выводим итоговую точность
print("Метрики и графики доступны в веб-интерфейсе ClearML")  # информируем пользователя о доступности результатов
