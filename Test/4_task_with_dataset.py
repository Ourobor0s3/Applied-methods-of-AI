# 4_task_with_dataset.py

"""
Этот скрипт демонстрирует использование ClearML Dataset в эксперименте.
Датасет будет автоматически загружен с сервера ClearML по имени и проекту,
что позволяет запускать эксперимент как локально, так и на удалённых агентах
БЕЗ необходимости вручную указывать Dataset ID.

ВАЖНО: Перед запуском создайте датасет с помощью 3_dataset_creation.py
или через CLI: clearml-data create/add/close
"""

import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from clearml import Dataset, Task
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 🎯 Параметры датасета - работаем по имени и проекту!
# Это автоматически использует последнюю версию датасета
DATASET_PROJECT = "Tutorial"
DATASET_NAME = "Synthetic Dataset"

# Альтернативно: можно указать конкретный ID версии для полной воспроизводимости
# Раскомментируй и укажи ID, если нужна конкретная версия
DATASET_ID = "your_dataset_id_here"
USE_SPECIFIC_VERSION = False  # Установи True, чтобы использовать DATASET_ID

# Инициализируем задачу в ClearML
task = Task.init(
    project_name="Tutorial",
    task_name="Task with ClearML Dataset",
    output_uri=True,
)

# Добавляем теги к задаче
task.add_tags(["polynomial-regression", "tutorial", "with-dataset"])

# Получаем логгер
logger = task.get_logger()

print("=" * 60)
print("Загрузка датасета ClearML")
print("=" * 60)

###########################
##### Dataset loading #####
###########################

# Загружаем датасет с сервера ClearML
# Датасет автоматически кэшируется локально для повторного использования

try:
    if USE_SPECIFIC_VERSION and "DATASET_ID" in globals():
        # Способ 1: Загрузка конкретной версии по ID
        print(f"\n📥 Загрузка датасета по ID: {DATASET_ID}")
        dataset = Dataset.get(dataset_id=DATASET_ID)
    else:
        # Способ 2: Загрузка последней версии по имени и проекту (РЕКОМЕНДУЕТСЯ)
        print(
            f"\n📥 Загрузка датасета по имени: {DATASET_PROJECT}/{DATASET_NAME}"
        )
        print("   (автоматически используется последняя версия)")
        dataset = Dataset.get(
            dataset_project=DATASET_PROJECT,
            dataset_name=DATASET_NAME,
        )

    print(f"✅ Датасет загружен: {dataset.name}")
    print(f"   ID версии: {dataset.id}")
    print(f"   Проект: {dataset.project}")

except Exception as e:
    print(f"\n❌ Ошибка загрузки датасета: {e}")
    print("\n💡 Убедись, что датасет существует:")
    print("   1. Запусти: python 3_dataset_creation.py")
    print("   2. Или через CLI:")
    print(
        f'      clearml-data create --project {DATASET_PROJECT} --name "{DATASET_NAME}"'
    )
    print("      clearml-data add --files ./data/synthetic_dataset.csv")
    print("      clearml-data close")
    print("\n   3. Проверь список датасетов:")
    print(f"      clearml-data list --project {DATASET_PROJECT}")
    raise

# Получаем локальный путь к датасету
# get_local_copy() скачивает датасет в локальный кэш и возвращает путь к нему
dataset_path = dataset.get_local_copy()
print(f"   Локальный путь: {dataset_path}")

# Список всех файлов в датасете
files = dataset.list_files()
print(f"   Количество файлов: {len(files)}")
print(f"   Файлы: {files}")

# Загружаем CSV файл
# Предполагаем, что в датасете один CSV файл
csv_file = [f for f in files if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(dataset_path, csv_file))

print("\n✅ Датасет успешно загружен и готов к использованию!")
print("=" * 60)

#####################
######## EDA ########
#####################

# Выводим общую информацию о датасете
print("\nEDA: Общая информация о датасете")
print(df.info())

# Логируем основную информацию о датасете
dataset_info = {
    "Dataset shape": str(df.shape),
    "Number of features": str(df.shape[1] - 1),
    "Number of samples": str(df.shape[0]),
    "Target variable": "target",
    "ClearML Dataset ID": dataset.id,
    "Dataset Version": dataset.name,
}
info_df = pd.DataFrame(
    list(dataset_info.items()), columns=["Property", "Value"]
)
logger.report_table(
    title="Dataset Statistics",
    series="Basic Info",
    iteration=0,
    table_plot=info_df,
)

# Выводим статистики по числовым признакам
print("EDA: Статистики по числовым признакам")
stats_df = df.describe()
print(stats_df)
logger.report_table(
    title="Dataset Statistics",
    series="Numerical Features",
    iteration=0,
    table_plot=stats_df,
)

# Разделяем признаки (X) и целевую переменную (y)
X = df.drop("target", axis=1)
y = df["target"]

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Создаем и логируем matplotlib график PCA scatter plot
print("Создаем и логируем matplotlib график PCA scatter plot")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.colorbar(scatter)
plt.title("PCA Scatter Plot (2 Components)")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

logger.report_matplotlib_figure(
    title="Dataset Visualization",
    series="PCA Scatter Plot",
    figure=plt,
)

# Создаем и логируем корреляционную матрицу
print("Создаем и логируем корреляционную матрицу")
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix.values, cmap="coolwarm", aspect="auto")
plt.colorbar(im)
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(correlation_matrix.columns)
plt.title("Correlation Matrix")
plt.tight_layout()

logger.report_matplotlib_figure(
    title="Dataset Visualization",
    series="Correlation Matrix",
    figure=plt,
)

# Определяем гиперпараметры для модели
hyperparams = {
    "poly_degree_range": list(range(1, 5)),
    "random_state": 2,
    "C": 1.0,
    "max_iter": 100,
}
task.connect(hyperparams)

###############################
##### Preprocessing stage #####
###############################

print("Препроцессинг данных...")

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
logger.report_matplotlib_figure(
    title="Preprocessing Visualization",
    series="Train Class Distribution",
    figure=plt,
)

fig, ax = plt.subplots()
ax.bar(list(test_class_counts.keys()), list(test_class_counts.values()))
ax.set_title("Test Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
logger.report_matplotlib_figure(
    title="Preprocessing Visualization",
    series="Test Class Distribution",
    figure=plt,
)

#######################################
##### Hyperparameter tuning stage #####
#######################################

print("Начинаем подбор гиперпараметров...")

poly_degree_range = hyperparams["poly_degree_range"]
train_accuracies = []
val_accuracies = []

for degree in poly_degree_range:
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            (
                "logistic",
                LogisticRegression(
                    random_state=hyperparams["random_state"],
                    C=hyperparams["C"],
                    max_iter=hyperparams["max_iter"],
                ),
            ),
        ]
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_test, y_test_pred)

    val_precision = float(precision_score(y_test, y_test_pred))
    val_recall = float(recall_score(y_test, y_test_pred))
    val_f1 = float(f1_score(y_test, y_test_pred))
    train_precision = float(precision_score(y_train, y_train_pred))
    train_recall = float(recall_score(y_train, y_train_pred))
    train_f1 = float(f1_score(y_train, y_train_pred))

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    logger.report_scalar(
        title="Accuracy",
        series="train",
        value=float(train_acc),
        iteration=degree,
    )
    logger.report_scalar(
        title="Accuracy",
        series="validation",
        value=float(val_acc),
        iteration=degree,
    )
    logger.report_scalar(
        title="Precision",
        series="train",
        value=train_precision,
        iteration=degree,
    )
    logger.report_scalar(
        title="Precision",
        series="validation",
        value=val_precision,
        iteration=degree,
    )
    logger.report_scalar(
        title="Recall", series="train", value=train_recall, iteration=degree
    )
    logger.report_scalar(
        title="Recall", series="validation", value=val_recall, iteration=degree
    )
    logger.report_scalar(
        title="F1 Score", series="train", value=train_f1, iteration=degree
    )
    logger.report_scalar(
        title="F1 Score", series="validation", value=val_f1, iteration=degree
    )

    print(
        f"polynomial_degree={degree}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
    )

best_val_accuracy = max(val_accuracies)
best_poly_degree = poly_degree_range[val_accuracies.index(best_val_accuracy)]

final_accuracy = best_val_accuracy
logger.report_single_value(name="final_accuracy", value=final_accuracy)

best_hyperparams = {
    "best_poly_degree": best_poly_degree,
    "random_state": hyperparams["random_state"],
    "C": hyperparams["C"],
    "max_iter": hyperparams["max_iter"],
}

best_hyperparams_df = pd.DataFrame(
    list(best_hyperparams.items()), columns=["Hyperparameter", "Value"]
)
logger.report_table(
    title="Best Hyperparameters",
    series="Tuned Values",
    iteration=0,
    table_plot=best_hyperparams_df,
)

print("Создаем DataFrame с результатами и logger.report_table")
results_df = pd.DataFrame(
    {
        "polynomial_degree": poly_degree_range,
        "train_accuracy": train_accuracies,
        "validation_accuracy": val_accuracies,
    }
)
logger.report_table(
    title="Training Results",
    series="Results",
    iteration=0,
    table_plot=results_df,
)

######################################
##### Final model training stage #####
######################################

print("Обучение финальной модели с лучшими гиперпараметрами...")
logger.report_text("Training final model with best hyperparameters...")

final_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=best_poly_degree)),
        (
            "logistic",
            LogisticRegression(
                random_state=hyperparams["random_state"],
                C=hyperparams["C"],
                max_iter=hyperparams["max_iter"],
            ),
        ),
    ]
)
final_model.fit(X_train_scaled, y_train)

y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
y_pred = final_model.predict(X_test_scaled)

##################################
##### Model evaluation stage #####
##################################

print("Оценка производительности модели...")

print("Вычисляем ROC curve и строим через plotly")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
)
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(dash="dash"),
    )
)
fig.update_layout(
    title="ROC Curve (Plotly)",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
)

logger.report_plotly(title="Training Results", series="ROC Curve", figure=fig)

print("Вычисляем confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

logger.report_confusion_matrix(
    title="Confusion Matrix",
    series="Validation",
    iteration=0,
    matrix=cm,
    xaxis="Predicted",
    yaxis="Actual",
)

val_precision = float(precision_score(y_test, y_pred))
val_recall = float(recall_score(y_test, y_pred))
val_f1 = float(f1_score(y_test, y_pred))

logger.report_single_value(name="precision", value=val_precision)
logger.report_single_value(name="recall", value=val_recall)
logger.report_single_value(name="f1_score", value=val_f1)

print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-score: {val_f1:.4f}")

print("Логгируем часть предсказаний")
predictions_df = pd.DataFrame(
    {
        "true_label": y_test,
        "predicted_label": y_pred,
        "prediction_proba": y_pred_proba,
    }
)
logger.report_table(
    title="Sample Predictions",
    series="Debug Samples",
    iteration=0,
    table_plot=predictions_df.head(20),
)

##############################
##### Model saving stage #####
##############################

print("Сохранение модели...")

model_path = "models/polynomial_with_dataset.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(final_model, model_path, compress=True)

task.close()
print(f"✅ Обучение завершено! Финальная точность: {final_accuracy:.4f}")
print("Метрики и графики доступны в веб-интерфейсе ClearML")
print(f"Использован датасет: {dataset.name} (ID версии: {dataset.id})")
