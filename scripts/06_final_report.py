"""Step 6: generate final comparison report in Russian."""
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import RESULTS_DIR


def main():
    summary = pd.read_csv(RESULTS_DIR / "clustering_summary.csv")

    best_per_descriptor = summary.loc[
        summary.groupby("descriptor")["calinski_harabasz"].idxmax()
    ].reset_index(drop=True)

    best_overall = summary.loc[summary["calinski_harabasz"].idxmax()]
    best_by_db = summary.loc[summary["davies_bouldin"].idxmin()]

    lines = [
        "# Итоговый отчёт: кластеризация изображений транспортных средств IntelliVision",
        "",
        "## 1. Цель",
        "",
        "Проверить, может ли кластеризация предвычисленных дескрипторов изображений заменить дорогостоящую многослойную ручную разметку датасета из 416 314 изображений транспортных средств.",
        "",
        "## 2. Лучшая конфигурация",
        "",
        "### По Calinski-Harabasz (выше — лучше)",
        "",
        f"- **Дескриптор:** {best_overall['descriptor']}",
        f"- **Масштабирование:** {best_overall['scaler']}",
        f"- **Алгоритм:** {best_overall['method']}",
        f"- **Количество кластеров:** {int(best_overall['n_clusters'])}",
        f"- **Calinski-Harabasz:** {best_overall['calinski_harabasz']:.2f}",
        f"- **Davies-Bouldin:** {best_overall['davies_bouldin']:.4f}",
        "",
        "### По Davies-Bouldin (ниже — лучше)",
        "",
        f"- **Дескриптор:** {best_by_db['descriptor']}",
        f"- **Масштабирование:** {best_by_db['scaler']}",
        f"- **Алгоритм:** {best_by_db['method']}",
        f"- **Количество кластеров:** {int(best_by_db['n_clusters'])}",
        f"- **Calinski-Harabasz:** {best_by_db['calinski_harabasz']:.2f}",
        f"- **Davies-Bouldin:** {best_by_db['davies_bouldin']:.4f}",
        "",
        "## 3. Лучшие результаты по дескрипторам",
        "",
        "| Дескриптор | Масштабирование | Алгоритм | Кластеров | Calinski-Harabasz | Davies-Bouldin |",
        "|------------|-----------------|----------|-----------|-------------------|----------------|",
    ]
    for _, row in best_per_descriptor.iterrows():
        lines.append(
            f"| {row['descriptor']} | {row['scaler']} | {row['method']} | {int(row['n_clusters'])} | "
            f"{row['calinski_harabasz']:.2f} | {row['davies_bouldin']:.4f} |"
        )

    lines += [
        "",
        "## 4. Интерпретация лучших кластеров",
        "",
        "Подробные описания см. в `results/cluster_descriptions.md`.",
        "",
        "Краткая интерпретация лучшей конфигурации (`vdc_type`, стандартизация, KMeans, 5 кластеров):",
        "",
        "- **Кластер 4:** седаны, вид сзади.",
        "- **Кластер 3:** светлые микроавтобусы, в основном вид спереди.",
        "- **Кластер 2:** седаны, вид спереди.",
        "- **Кластер 1:** хэтчбеки и минивены, вертикальный срез сзади и наклонный срез спереди.",
        "- **Кластер 0:** в основном большие джипы, с единичным грузовиком и хэтчбеком.",
        "",
        "Основные разделения происходят по **типу кузова**, **ракурсу съёмки** и частично по **цвету**.",
        "",
        "## 5. Выбросы",
        "",
        "- **Isolation Forest:** выделил редкие белые грузовики и старые автомобили с угловатыми формами.",
        "- **Local Outlier Factor:** выделил тёмные автомобили, ночные снимки и фото с ветками, заслоняющими объект.",
        "",
        "Разные алгоритмы находят разные типы аномалий: IF — редкие классы объектов, LOF — нестандартные условия съёмки.",
        "",
        "## 6. Рекомендации",
        "",
        "- Для полуавтоматической разметки по типу ТС и ракурсу используй **дескриптор `vdc_type` со стандартизацией и KMeans**.",
        "- В качестве стартовой точки бери **5 кластеров**, при необходимости дроби самые большие кластеры на подклассы.",
        "- Для работы на машине с 16 ГБ RAM применяй **PCA до 128 компонент** — это разумный баланс памяти и дисперсии.",
        "- Для поиска редких классов используй **Isolation Forest**, а для выявления плохих или нестандартных условий съёмки — **Local Outlier Factor**.",
        "- Если позволяет память, попробуй комбинации дескрипторов: `vdc_type` + `osnet` (форма + идентификация) или `vdc_type` + `vdc_color` (тип + цвет) для более богатых многофакторных кластеров.",
        "",
        "## 7. Созданные файлы",
        "",
        "- `results/best_clustering.csv` — итоговое отображение изображение → кластер.",
        "- `results/cluster_descriptions.md` — описание кластеров и выбросов.",
        "- `results/clustering_summary.csv` — метрики всех протестированных конфигураций.",
        "- `results/figures/` — t-SNE-графики и сетки изображений по кластерам.",
        "",
    ]

    report_path = RESULTS_DIR / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Итоговый отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
