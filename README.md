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
│   ├── 02_preprocess.py              # масштабирование + PCA до максимальных компонент (N_COMPONENTS_MAP)
│   ├── 03_cluster.py                 # кластеризация всех конфигураций с максимальными компонентами
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
│   ├── final_report.md               # итоговый отчёт по максимальным компонентам
│   ├── cluster_descriptions.md       # описание кластеров и выбросов
│   ├── clustering_max_components_report.md          # отчёт и рекомендации по максимальным компонентам
│   ├── clustering_summary_max_components.csv        # метрики индивидуальных дескрипторов с максимальными n_components
│   ├── combined_clustering_summary_max_components.csv # метрики комбинаций с максимальными n_components
│   ├── best_configurations.csv       # топ-конфигурации (максимальные компоненты)
│   ├── best_clustering.csv           # путь → кластер для лучшей конфигурации (генерируется scripts/05_best_and_outliers.py)
│   ├── figures/                      # t-SNE, сетки изображений и графики метрик
│   └── pca_variance_analysis.csv     # сохранённая дисперсия PCA при разных n_components
└── data/                             # сырые данные (не входят в git)
    └── IntelliVision_case/
        ├── descriptors/              # pickle-дескрипторы
        ├── raw_data/               # изображения
        └── images_paths.csv
```

## Эволюция исследования

### 1. Первый этап: уменьшенные дескрипторы (128 компонент)

Изначально все дескрипторы были сведены к **128 компонентам** через PCA. Это позволило быстро оценить качество кластеризации и интерпретируемость дескрипторов на локальной машине.

**Ключевые выводы этапа (128 компонент):**

| Дескриптор | Лучший скейлер | Calinski-Harabasz | Davies-Bouldin | Интерпретируемость |
|---|---|---:|---:|---|
| **vdc_type** | standard | 94 664 | **1,62** | **Высокая** — тип кузова и ракурс |
| vdc_type | minmax | **107 466** | 1,81 | Высокая — максимальное разделение |
| vdc_color | minmax | 69 147 | 2,31 | Средняя — преимущественно цвет |
| osnet | minmax | 32 412 | 2,79 | Низкая — re-identification дескриптор |
| efficientnet-b7 | standard | 20 476 | 4,26 | Низкая — ImageNet-дескриптор |

При сжатии до 128 компонент `efficientnet-b7` потерял около **50% дисперсии**, что сильно снизило его метрики. Подробные результаты первого этапа и расшифровка кластеров сохранены в отчётах: [`results/cluster_descriptions.md`](results/cluster_descriptions.md).

### 2. Решение: максимальные компоненты на 16 GB RAM

Чтобы понять, хранит ли сжатие дескрипторов потерянную информацию, было проведено исследование максимального числа компонент, которое выдерживает машина с **16 GB RAM** без свопа. Результаты исследования PCA:

- `efficientnet-b7`: 1536 компонент сохраняют ~92.5% дисперсии (пик RAM ~15.5 GB);
- `osnet`: 512 компонент сохраняют 100% дисперсии;
- `vdc_type`: 512 компонент сохраняют 100% дисперсии;
- `vdc_color`: 128 компонент сохраняют 100% дисперсии.

Подробный отчёт и рекомендации по памяти: [`results/clustering_max_components_report.md`](results/clustering_max_components_report.md).

### 3. Текущий пайплайн: `N_COMPONENTS_MAP`

На основе исследования в `src/config.py` задана карта максимальных компонент:

```python
N_COMPONENTS_MAP = {
    "efficientnet-b7": 1536,
    "osnet": 512,
    "vdc_type": 512,
    "vdc_color": 128,
}
```

Все скрипты пайплайна (`02`, `03`, `04`, `04b`, `05`, `06`, `07`, `08`) настроены на эти значения. Это даёт **максимальную сохранённую дисперсию** при гарантированном укладывании в 16 GB RAM.

## Как воспроизвести

### Системные требования

- Python 3.9+
- **16 GB RAM** — текущий пайплайн использует `N_COMPONENTS_MAP` и максимальные компоненты для каждого дескриптора
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

## Ключевые результаты (пайплайн с максимальными компонентами)

### Две лучшие конфигурации на `vdc_type` + KMeans + 5 кластеров

- **Лучшая по Calinski-Harabasz:** `vdc_type / MinMax / KMeans / 512 компонент` (CH 104 694, DB 1,83).
- **Лучшая по Davies-Bouldin:** `vdc_type / StandardScaler / KMeans / 512 компонент` (CH 91 537, DB 1,65).

### Сравнение дескрипторов (5 кластеров, KMeans, максимальные компоненты)

| Дескриптор | n_components | Лучший скейлер | Calinski-Harabasz | Davies-Bouldin | Интерпретируемость |
|---|---|---|---:|---:|---|
| `vdc_type` | 512 | standard | 91 537 | **1,65** | **Высокая** — тип кузова и ракурс |
| `vdc_type` | 512 | minmax | **104 694** | 1,83 | Высокая — максимальное разделение |
| `vdc_color` | 128 | minmax | 69 147 | 2,31 | Средняя — преимущественно цвет |
| `osnet` | 512 | minmax | 28 885 | 2,96 | Низкая — re-identification дескриптор |
| `efficientnet-b7` | 1536 | minmax | 10 527 | 5,91 | Низкая — ImageNet-дескриптор |
| `vdc_type+vdc_color` | 512+128 | standard | 63 662 | 2,03 | Высокая — тип + цвет |
| `vdc_type+osnet` | 512+512 | standard | 35 161 | 2,76 | Средняя — тип + идентификация |

### Поиск выбросов

- **Isolation Forest** находит редкие классы объектов (белые грузовики, угловатые старые автомобили).
- **Local Outlier Factor** выявляет аномальные условия съёмки (ночь, заслонение объектов, тёмные автомобили).

### Рекомендации

1. Для разметки по **типу кузова и ракурсу** используй `vdc_type` + `StandardScaler` + `KMeans` с 5 кластерами и 512 компонентами (лучший DB) или `vdc_type` + `MinMax` + `KMeans` (лучший CH).
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