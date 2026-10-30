#!/usr/bin/env python3
"""ЛР №3, п.2: система с 1 входом (Бюджет) и 1 выходом (Класс сборки).

Находит выход на основе трёх механизмов вывода:
  1. Полная композиция (CRI), max-min
  2. Полная композиция (CRI), max-произведение
  3. Уровни истинности предпосылок правил (прямой алгоритм Мамдани)
для каждой из 4 импликаций (Мамдани, Ларсен, Лукасевич, Заде) - сравнение
для п.4 задания. Правила объединяются max для Мамдани и Ларсена и min для
Лукасевича и Заде. Дефазификация - методом центра тяжести.

При чётком входе все три механизма совпадают математически, поэтому они
сравниваются на нечётком входе.

Нечёткое отношение R(x,y) зависит только от (импликация, агрегация), а не
от конкретного входа, поэтому строится один раз на импликацию и переиспользуется
для всех входных значений (в т.ч. при развёртке по всему диапазону) - иначе
пересчёт полной NxN матрицы на каждую точку был бы неприемлемо медленным.

Запуск: uv run python scripts/run_single_input.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без дисплея - сохраняем графики в файлы

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fuzzy_lr3 import composition, truth_level, viz  # noqa: E402
from fuzzy_lr3.defuzzify import centroid  # noqa: E402
from fuzzy_lr3.implications import IMPLICATION_TITLES  # noqa: E402
from fuzzy_lr3.parser import load_rulebase  # noqa: E402

RULES_PATH = PROJECT_ROOT / "rules" / "single_input.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMPLICATIONS = list(IMPLICATION_TITLES)


def tier_label(score: float) -> str:
    """Переводит индекс класса сборки [0, 12] в привычное название из ЛР №1/2."""
    if score < 4:
        return "Базовый"
    if score < 8:
        return "Средний"
    return "Высокий"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rb = load_rulebase(RULES_PATH)
    input_var = rb.input_variable("Бюджет")
    output_var = rb.output_variable

    print(f"База правил: {RULES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Правил: {len(rb.rules)}")
    for rule in rb.rules:
        print(f"  #{rule.index + 1}: {rule.describe(rb.output_var)}")
    print()

    # Отношение R(x,y) зависит только от импликации - строим один раз
    # на каждую импликацию и переиспользуем во всём скрипте.
    relations = {impl: composition.build_relation(rb, implication=impl) for impl in IMPLICATIONS}

    # --- 1. Графики термов входной и выходной переменных -----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    viz.plot_terms(axes[0], input_var, title="Бюджет, ₽")
    viz.plot_terms(axes[1], output_var, title="Класс сборки (индекс 0..12)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_input_terms.png", dpi=150)
    plt.close(fig)

    # --- 2. Три механизма вывода для одного чёткого входа -----------------
    x0 = 92_000.0  # между термами "Ниже среднего" и "Средний" - есть интерполяция
    print(f"Чёткий вход: Бюджет = {x0:.0f} ₽\n")

    singleton0 = composition.singleton(input_var.universe, x0)
    rows = []
    for impl in IMPLICATIONS:
        title = IMPLICATION_TITLES[impl]
        relation = relations[impl]

        b_maxmin = composition.max_min_compose(singleton0, relation)
        b_maxprod = composition.max_prod_compose(singleton0, relation)
        b_truth = truth_level.infer(rb, {"Бюджет": x0}, implication=impl).output

        y_maxmin = centroid(output_var.universe, b_maxmin)
        y_maxprod = centroid(output_var.universe, b_maxprod)
        y_truth = centroid(output_var.universe, b_truth)

        rows.append([
            title,
            f"{y_maxmin:.3f} ({tier_label(y_maxmin)})",
            f"{y_maxprod:.3f} ({tier_label(y_maxprod)})",
            f"{y_truth:.3f} ({tier_label(y_truth)})",
        ])

    print_table(
        ["Импликация", "CRI max-min", "CRI max-произв.", "Уровни истинности"],
        rows,
    )
    print(
        "Примечание: при чётком (синглтон) входе все три механизма математически "
        "совпадают - сводятся к срезу отношения R(x0, y) в точке входа. Разница в "
        "третьем знаке - следствие привязки синглтона к узлу сетки. Механизмы "
        "расходятся только при по-настоящему нечётком входе (см. ниже).\n"
    )

    # --- 3. Нечёткий (не синглтон) вход - где механизмы действительно различаются
    fuzzy_x0 = 150_000.0
    spread = 20_000.0
    fuzzy_input = np.clip(1 - np.abs(input_var.universe - fuzzy_x0) / spread, 0, 1)
    crisp_singleton = composition.singleton(input_var.universe, fuzzy_x0)

    print(f"Нечёткий вход: «около {fuzzy_x0:.0f} ₽» (треугольник ± {spread:.0f} ₽)\n")
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in IMPLICATIONS:
        relation = relations[impl]
        b_fuzzy_maxmin = composition.max_min_compose(fuzzy_input, relation)
        b_fuzzy_maxprod = composition.max_prod_compose(fuzzy_input, relation)
        b_fuzzy_truth = truth_level.infer_fuzzy(rb, {"Бюджет": fuzzy_input}, implication=impl).output
        b_crisp = composition.max_min_compose(crisp_singleton, relation)

        y_fuzzy_maxmin = centroid(output_var.universe, b_fuzzy_maxmin)
        y_fuzzy_maxprod = centroid(output_var.universe, b_fuzzy_maxprod)
        y_fuzzy_truth = centroid(output_var.universe, b_fuzzy_truth)
        y_crisp = centroid(output_var.universe, b_crisp)
        rows.append([
            IMPLICATION_TITLES[impl],
            f"{y_fuzzy_maxmin:.3f}",
            f"{y_fuzzy_maxprod:.3f}",
            f"{y_fuzzy_truth:.3f}",
            f"{y_crisp:.3f}",
        ])

        if impl == "mamdani":
            viz.plot_output_set(
                ax, output_var.universe, b_fuzzy_maxmin,
                label=f"нечёткий вход, max-min ({y_fuzzy_maxmin:.2f})", crisp_value=y_fuzzy_maxmin)
            viz.plot_output_set(
                ax, output_var.universe, b_fuzzy_maxprod,
                label=f"нечёткий вход, max-произв. ({y_fuzzy_maxprod:.2f})", crisp_value=y_fuzzy_maxprod)
            viz.plot_output_set(
                ax, output_var.universe, b_crisp,
                label=f"чёткий вход ({y_crisp:.2f})", crisp_value=y_crisp)

        # Выходные множества всех трёх механизмов для одной импликации.
        if impl == "lukasiewicz":
            fig_mech, ax_mech = plt.subplots(figsize=(7, 4.5))
            for label, b, y in [
                ("CRI max-min", b_fuzzy_maxmin, y_fuzzy_maxmin),
                ("CRI max-произв.", b_fuzzy_maxprod, y_fuzzy_maxprod),
                ("Уровни истинности", b_fuzzy_truth, y_fuzzy_truth),
            ]:
                viz.plot_output_set(ax_mech, output_var.universe, b, label=f"{label} ({y:.2f})", crisp_value=y)
            ax_mech.set_title(f"«Около {fuzzy_x0:.0f} ₽», импликация Лукасевича: три механизма вывода")
            ax_mech.set_xlabel("Класс сборки")
            ax_mech.set_ylabel("μ")
            ax_mech.legend(fontsize=8)
            ax_mech.grid(alpha=0.3)
            fig_mech.tight_layout()
            fig_mech.savefig(OUTPUT_DIR / "single_input_mechanisms.png", dpi=150)
            plt.close(fig_mech)

    print_table(
        ["Импликация", "Нечёт. вход, max-min", "Нечёт. вход, max-произв.", "Нечёт. вход, уровни ист.", "Чёткий вход"],
        rows,
    )
    print(
        "Теперь механизмы действительно расходятся. Для Мамдани уровни истинности "
        "с alpha = sup min(A', A_k) математически совпадают с max-min композицией.\n"
    )

    ax.set_title("CRI max-min, импликация Мамдани: нечёткий vs чёткий вход")
    ax.set_xlabel("Класс сборки")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_input_fuzzy_vs_crisp_input.png", dpi=150)
    plt.close(fig)

    # --- 4. Развёртка по всему диапазону бюджета: выход(бюджет) ----------
    # Вход нечёткий («около x»), потому что при чётком все панели совпали бы.
    sweep_x = np.linspace(input_var.universe[0], input_var.universe[-1], 120)
    around = lambda x: np.clip(1 - np.abs(input_var.universe - x) / spread, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    mechanisms = [
        ("CRI max-min", lambda impl, s: composition.max_min_compose(s, relations[impl])),
        ("CRI max-произв.", lambda impl, s: composition.max_prod_compose(s, relations[impl])),
    ]

    for ax, (mech_title, compute) in zip(axes[:2], mechanisms):
        for impl in IMPLICATIONS:
            ys = []
            for x in sweep_x:
                ys.append(centroid(output_var.universe, compute(impl, around(x))))
            ax.plot(sweep_x, ys, label=IMPLICATION_TITLES[impl])
        ax.set_title(mech_title)
        ax.set_xlabel("Центр нечёткого бюджета, ₽")
        ax.grid(alpha=0.3)

    ax = axes[2]
    for impl in IMPLICATIONS:
        ys = [centroid(output_var.universe, truth_level.infer_fuzzy(rb, {"Бюджет": around(x)}, implication=impl).output)
              for x in sweep_x]
        ax.plot(sweep_x, ys, label=IMPLICATION_TITLES[impl])
    ax.set_title("Уровни истинности")
    ax.set_xlabel("Центр нечёткого бюджета, ₽")
    ax.grid(alpha=0.3)

    axes[0].set_ylabel("Класс сборки (дефазифицированный)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Сравнение импликаций и механизмов вывода, вход «около x» (± {spread:.0f} ₽)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_input_sweep.png", dpi=150)
    plt.close(fig)

    print(f"Графики сохранены в {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
