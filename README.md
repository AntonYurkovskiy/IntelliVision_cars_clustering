# Кластеризация изображений транспортных средств IntelliVision

Исследование применимости методов кластеризации к предвычисленным дескрипторам изображений для автоматизации разметки датасета и поиска выбросов.

## Кратко о проекте

Проект проверяет гипотезу: можно ли заменить дорогую многослойную ручную разметку 416 314 изображений транспортных средств кластеризацией по готовым дескрипторам нейронных сетей. В работе сравниваются четыре дескриптора (`vdc_type`, `vdc_color`, `osnet`, `efficientnet-b7`), два варианта масштабирования, два алгоритма кластеризации (`KMeans`, `MiniBatchKMeans`) и несколько методов поиска выбросов.

## Быстрые ссылки

- [Итоговый отчёт с метриками и рекомендациями](results/final_report.md)
- [Кластеризация с максимальными n_components на 16 GB RAM](results/clustering_max_components_report.md)
- [Расшифровка кластеров и выбросов](results/cluster_descriptions.md)
- [Пошаговый аттестационный ноутбук](project_6_clustering.ipynb)
- [Аналитический ноутбук с графиками](notebooks/03_analysis.ipynb)

## Структура репозитория

```
.
├── README.md                         # эта страница
├── requirements.txt                  # зависимости Python
├── src/                              # модульный код проекта
│   ├── config.py                     # пути и константы
│   ├── load_data.py                  # загрузка дескрипторов и путей к изображениям
│   ├── preprocess.py                 # масштабирование и PCA
│   ├── clustering.py                 # KMeans, MiniBatchKMeans, DBSCAN
│   ├── metrics.py                    # Calinski-Harabasz, Davies-Bouldin
│   ├── outliers.py                   # Isolation Forest, LOF
│   ├── visualization.py                # t-SNE и сетки изображений
│   └── utils.py                      # вспомогательные функции
├── scripts/                          # скрипты пайплайна
│   ├── 01_explore_data.py            # первичный анализ дескрипторов
│   ├── 02_preprocess.py              # масштабирование + PCA до 128 компонент
│   ├── 03_cluster.py                 # кластеризация всех конфигураций
│   ├── 04_analyze_and_visualize.py   # ранжирование и визуализация топ-конфигураций
│   ├── 04b_metrics_plots.py          # сводные графики метрик
│   ├── 05_best_and_outliers.py       # сохранение лучшей конфигурации и выбросов
│   ├── 06_final_report.py            # генерация markdown-отчётов
│   ├── 07_combine_descriptors.py     # эксперименты с комбинированными дескрипторами
│   └── 08_verify_project.py          # проверка целостности проекта
├── tests/                            # юнит-тесты (pytest)
├── notebooks/                        # аналитические ноутбуки
│   └── 03_analysis.ipynb             # интерактивный анализ результатов
├── project_6_clustering.ipynb        # аттестационный ноутбук
├── results/                          # сгенерированные артефакты
│   ├── final_report.md               # итоговый отчёт
│   ├── cluster_descriptions.md       # описание кластеров и выбросов
│   ├── clustering_summary.csv        # метрики всех конфигураций
│   ├── clustering_summary_max_components.csv        # метрики при максимальных n_components
│   ├── combined_clustering_summary_max_components.csv # метрики комбинаций при максимальных n_components
│   ├── clustering_max_components_report.md          # отчёт и рекомендации
│   ├── best_configurations.csv       # топ-конфигурации
│   ├── best_clustering.csv         # путь → кластер для лучшей конфигурации (генерируется scripts/05_best_and_outliers.py)
│   ├── figures/                      # t-SNE, сетки изображений и графики метрик
│   └── combined_clustering_summary.csv # метрики комбинированных дескрипторов
├── out_of_rep/                       # служебные файлы (не входят в git)
│   ├── service_scripts/              # скрипты заполнения/починки ноутбуков
│   ├── backups/                      # backup-копии ноутбуков
│   └── notes/                        # рабочие заметки (progress.md)
└── data/                             # сырые данные (не входят в git)
    └── IntelliVision_case/
        ├── descriptors/              # pickle-дескрипторы
        ├── raw_data/               # изображения
        └── images_paths.csv
```

## Как воспроизвести

### Системные требования

- Python 3.9+
- 16 GB RAM (для стандартного пайплайна с 128 компонентами достаточно 8–12 GB; для максимальных n_components — 16 GB)
- ~10 GB свободного места на диске для данных и сгенерированных артефактов

### Клонирование и запуск

```bash
git clone https://github.com/AntonYurkovskiy/IntelliVision_cars_clustering
cd IntelliVision_cars_clustering

# Создание и активация виртуального окружения (Windows)
python -m venv .venv
.venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

1. Скачать [`данные`](https://drive.google.com/file/d/1vkQaj0Lr4Jwkumli7k9IzCtxFP1tIoXH/view).
2. Поместите данные в `data/IntelliVision_case/` согласно структуре выше.
3. Запустите пайплайн в порядке номеров:
   ```bash
   python scripts/01_explore_data.py
   python scripts/02_preprocess.py
   python scripts/03_cluster.py
   python scripts/04_analyze_and_visualize.py
   python scripts/04b_metrics_plots.py
   python scripts/05_best_and_outliers.py
   python scripts/06_final_report.py
   python scripts/07_combine_descriptors.py
   python scripts/08_verify_project.py
   ```
4. Откройте [`project_6_clustering.ipynb`](project_6_clustering.ipynb) или [`notebooks/03_analysis.ipynb`](notebooks/03_analysis.ipynb) для интерактивного просмотра результатов.
5. Запустите тесты:
   ```bash
   pytest tests/ -q
   ```

## Ключевые результаты

### Две лучшие конфигурации на `vdc_type` + KMeans + 5 кластеров

- **Лучшая по Calinski-Harabasz:** `vdc_type / MinMax / KMeans` (CH 107 466, DB 1,81).
- **Лучшая по Davies-Bouldin:** `vdc_type / StandardScaler / KMeans` (CH 94 664, DB 1,62).

### Сравнение дескрипторов (5 кластеров, KMeans)

| Дескриптор | Лучший скейлер | Calinski-Harabasz | Davies-Bouldin | Интерпретируемость |
|---|---|---:|---:|---|
| **vdc_type** | standard | 94 664 | **1,62** | **Высокая** — тип кузова и ракурс |
| vdc_type | minmax | **107 466** | 1,81 | Высокая — максимальное разделение |
| vdc_color | minmax | 69 147 | 2,31 | Средняя — преимущественно цвет |
| osnet | minmax | 32 412 | 2,79 | Низкая — re-identification дескриптор |
| efficientnet-b7 | standard | 20 476 | 4,26 | Низкая — ImageNet-дескриптор |

### Поиск выбросов

- **Isolation Forest** находит редкие классы объектов (белые грузовики, угловатые старые автомобили).
- **Local Outlier Factor** выявляет аномальные условия съёмки (ночь, заслонение объектов, тёмные автомобили).

### Кластеризация с максимальными n_components (16 GB RAM)

При поднятии размерности до максимума, который держит локальная машина, получены следующие лучшие результаты (5 кластеров, KMeans):

| Дескриптор | n_components | Scaler | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---:|
| `vdc_type` | 512 | standard | 91 537 | **1,65** |
| `vdc_type` | 512 | minmax | **104 694** | 1,83 |
| `vdc_color` | 128 | minmax | 69 147 | 2,31 |
| `osnet` | 512 | minmax | 28 885 | 2,96 |
| `efficientnet-b7` | 1536 | minmax | 10 527 | 5,91 |
| `vdc_type+vdc_color` | 512+128 | standard | 63 662 | 2,03 |
| `vdc_type+osnet` | 512+512 | standard | 35 161 | 2,76 |

Подробный отчёт и рекомендации по памяти: [`results/clustering_max_components_report.md`](results/clustering_max_components_report.md).

### Рекомендации

1. Для разметки по **типу кузова и ракурсу** используй `vdc_type` + `StandardScaler` + `KMeans` с 5 кластерами (лучший DB) или `vdc_type` + `MinMax` + `KMeans` (лучший CH).
2. Для совместной сегментации по **типу и цвету** используй `vdc_type + vdc_color` + `KMeans` с 5 кластерами.
3. Для поиска **редких классов** применяй **Isolation Forest**.
4. Для поиска **аномальных условий съёмки** применяй **Local Outlier Factor**.
5. Для работы на 16 GB RAM допустимы максимальные n_components: `efficientnet-b7` — 1536, `osnet` — 512, `vdc_type` — 512, `vdc_color` — 128. При этом не превышай 1536 для `efficientnet-b7` и не комбинируй его с другими дескрипторами.

## Технологии

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- Jupyter, pytest

## Данные

Сырые данные (изображения и pickle-дескрипторы) не входят в публичный репозиторий. См. `.gitignore`. Для воспроизведения загрузите исходный датасет в `data/IntelliVision_case/`.

## Лицензия

Проект распространяется под лицензией MIT. Подробнее см. [`LICENSE`](LICENSE).