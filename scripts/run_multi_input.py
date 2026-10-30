#!/usr/bin/env python3
"""ЛР №3, п.3: система с 5 входами (Бюджет, Требовательность задач,
Важность видеокарты, Разрешение монитора, Желаемая компактность корпуса) и
1 выходом (Класс сборки) - тот же выходной универсум/термы, что и в
однофакторной системе (run_single_input.py), для сопоставимости.

32 правила: плотная таблица из 27 правил по всем 3x3x3 комбинациям трёх
основных входов (Бюджет, Требовательность задач, Важность видеокарты) +
5 корректирующих правил для Разрешения монитора и Компактности корпуса -
двух дополнительных непрерывных признаков, перенесённых из предметной
области ЛР №1/2 (в отличие от там же встречавшихся категориальных моделей
комплектующих, разрешение и форм-фактор естественно ложатся на шкалу).

Правила объединяются max для Мамдани и Ларсена и min для Лукасевича и Заде.

Полная композиционная схема (построение отношения на произведении всех
универсумов) здесь не используется: при 5 входах по 1001 точке дискретизации
отношение потребовало бы астрономического числа ячеек - поэтому задание
требует только механизм на основе уровней истинности предпосылок правил,
который работает напрямую с конкретными входными значениями и не строит
многомерное отношение.

Запуск: uv run python scripts/run_multi_input.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fuzzy_lr3 import truth_level, viz  # noqa: E402
from fuzzy_lr3.defuzzify import centroid  # noqa: E402
from fuzzy_lr3.implications import IMPLICATION_TITLES  # noqa: E402
from fuzzy_lr3.parser import load_rulebase  # noqa: E402

RULES_PATH = PROJECT_ROOT / "rules" / "multi_input.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMPLICATIONS = list(IMPLICATION_TITLES)

# Разрешение монитора задаём в млн пикселей кадра: 1080p=2.0736, 1440p=3.6864, 4K=8.2944.
RES_1080P, RES_1440P, RES_4K = 2.0736, 3.6864, 8.2944

SCENARIOS = [
    ("Бюджет 65000 ₽, лёгкие задачи", {
        "Бюджет": 65000.0, "Требовательность задач": 1.0, "Важность видеокарты": 1.0,
        "Разрешение монитора": RES_1080P, "Желаемая компактность корпуса": 3.0}),
    ("Бюджет 65000 ₽, тяжёлые задачи", {
        "Бюджет": 65000.0, "Требовательность задач": 9.0, "Важность видеокарты": 2.0,
        "Разрешение монитора": RES_1080P, "Желаемая компактность корпуса": 3.0}),
    ("Бюджет 80000 ₽, всё среднее", {
        "Бюджет": 80000.0, "Требовательность задач": 5.0, "Важность видеокарты": 5.0,
        "Разрешение монитора": RES_1440P, "Желаемая компактность корпуса": 5.0}),
    ("Бюджет 110000 ₽ (между термами), нужна мощная видеокарта", {
        "Бюджет": 110000.0, "Требовательность задач": 6.0, "Важность видеокарты": 9.0,
        "Разрешение монитора": RES_1440P, "Желаемая компактность корпуса": 3.0}),
    ("Бюджет 135000 ₽, тяжёлые задачи, видеокарта важна", {
        "Бюджет": 135000.0, "Требовательность задач": 9.5, "Важность видеокарты": 9.5,
        "Разрешение монитора": RES_4K, "Желаемая компактность корпуса": 2.0}),
    ("Бюджет 135000 ₽, лёгкие задачи", {
        "Бюджет": 135000.0, "Требовательность задач": 1.0, "Важность видеокарты": 2.0,
        "Разрешение монитора": RES_1080P, "Желаемая компактность корпуса": 2.0}),
    ("Задачи тяжёлые, видеокарта не нужна (рендер/компиляция на CPU)", {
        "Бюджет": 90000.0, "Требовательность задач": 9.0, "Важность видеокарты": 0.5,
        "Разрешение монитора": RES_1080P, "Желаемая компактность корпуса": 3.0}),
    ("Бюджет 65000 ₽, но нужен 4K-монитор", {
        "Бюджет": 65000.0, "Требовательность задач": 3.0, "Важность видеокарты": 5.0,
        "Разрешение монитора": RES_4K, "Желаемая компактность корпуса": 3.0}),
    ("Бюджет 135000 ₽, но обязательно компактный корпус", {
        "Бюджет": 135000.0, "Требовательность задач": 5.0, "Важность видеокарты": 5.0,
        "Разрешение монитора": RES_1440P, "Желаемая компактность корпуса": 9.5}),
]


def tier_label(score: float) -> str:
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
    output_var = rb.output_variable

    print(f"База правил: {RULES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Входные переменные: {', '.join(rb.input_vars)}")
    print(f"Правил: {len(rb.rules)}")
    for rule in rb.rules:
        print(f"  #{rule.index + 1}: {rule.describe(rb.output_var)}")
    print()

    # --- 1. Графики термов всех переменных --------------------------------
    all_vars = [*rb.input_vars, rb.output_var]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, var_name in zip(axes.flat, all_vars):
        viz.plot_terms(ax, rb.variables[var_name], title=var_name)
    for ax in axes.flat[len(all_vars):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multi_input_terms.png", dpi=150)
    plt.close(fig)

    # --- 2. Таблица сценариев x импликаций ---------------------------------
    headers = ["Сценарий", *[IMPLICATION_TITLES[i] for i in IMPLICATIONS]]
    rows = []
    csv_rows = []
    for name, inputs in SCENARIOS:
        row = [name]
        csv_row = {"Сценарий": name, **inputs}
        for impl in IMPLICATIONS:
            result = truth_level.infer(rb, inputs, implication=impl)
            y = centroid(output_var.universe, result.output)
            row.append(f"{y:.2f} ({tier_label(y)})")
            csv_row[impl] = round(y, 3)
        rows.append(row)
        csv_rows.append(csv_row)

    print_table(headers, rows)
    print()

    csv_path = OUTPUT_DIR / "multi_input_scenarios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Сценарий", *rb.input_vars, *IMPLICATIONS])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Таблица сценариев сохранена в {csv_path.relative_to(PROJECT_ROOT)}")

    # --- 3. Карта выхода: Бюджет x Требовательность, остальное зафиксировано
    fixed_rest = {
        "Важность видеокарты": 5.0,
        "Разрешение монитора": RES_1440P,
        "Желаемая компактность корпуса": 5.0,
    }
    budget_var = rb.input_variable("Бюджет")
    workload_var = rb.input_variable("Требовательность задач")

    grid_n = 45
    budgets = np.linspace(budget_var.universe[0], budget_var.universe[-1], grid_n)
    workloads = np.linspace(workload_var.universe[0], workload_var.universe[-1], grid_n)

    fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(4.2 * len(IMPLICATIONS), 4), sharey=True)
    for ax, impl in zip(axes, IMPLICATIONS):
        grid = np.zeros((grid_n, grid_n))
        for i, workload in enumerate(workloads):
            for j, budget in enumerate(budgets):
                result = truth_level.infer(
                    rb,
                    {"Бюджет": budget, "Требовательность задач": workload, **fixed_rest},
                    implication=impl,
                )
                grid[i, j] = centroid(output_var.universe, result.output)

        im = ax.pcolormesh(budgets, workloads, grid, shading="auto", vmin=0, vmax=12, cmap="viridis")
        ax.set_title(IMPLICATION_TITLES[impl])
        ax.set_xlabel("Бюджет, ₽")
    axes[0].set_ylabel("Требовательность задач")
    fig.colorbar(im, ax=axes, label="Класс сборки (дефазифицированный)", shrink=0.85)
    fig.suptitle("Выход системы по Бюджету и Требовательности (остальные входы - средние)")
    fig.savefig(OUTPUT_DIR / "multi_input_heatmap.png", dpi=150)
    plt.close(fig)

    # --- 4. Карта выхода: Разрешение монитора x Компактность корпуса ------
    # Показывает эффект двух новых (непрерывных) входов и работу
    # корректирующих правил #28-32. Основные три входа фиксированы НЕ на
    # серединах термов (там несколько ключевых термов - "Тяжёлые",
    # "Высокая" GPU, бюджет "135000 ₽" - дают нулевую степень
    # истинности, и корректирующие правила попросту не срабатывают), а на
    # значениях, уверенно активирующих термы "135000 ₽"/"Тяжёлые"/"Высокая",
    # от которых зависят правила #28 и #31.
    fixed_core = {"Бюджет": 135000.0, "Требовательность задач": 8.0, "Важность видеокарты": 9.0}
    res_var = rb.input_variable("Разрешение монитора")
    compact_var = rb.input_variable("Желаемая компактность корпуса")

    resolutions = np.linspace(res_var.universe[0], res_var.universe[-1], grid_n)
    compactness = np.linspace(compact_var.universe[0], compact_var.universe[-1], grid_n)

    fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(4.2 * len(IMPLICATIONS), 4), sharey=True)
    for ax, impl in zip(axes, IMPLICATIONS):
        grid = np.zeros((grid_n, grid_n))
        for i, compact in enumerate(compactness):
            for j, res in enumerate(resolutions):
                result = truth_level.infer(
                    rb,
                    {"Разрешение монитора": res, "Желаемая компактность корпуса": compact, **fixed_core},
                    implication=impl,
                )
                grid[i, j] = centroid(output_var.universe, result.output)

        im = ax.pcolormesh(resolutions, compactness, grid, shading="auto", vmin=0, vmax=12, cmap="viridis")
        ax.set_title(IMPLICATION_TITLES[impl])
        ax.set_xlabel("Разрешение, млн пикс.")
    axes[0].set_ylabel("Желаемая компактность корпуса")
    fig.colorbar(im, ax=axes, label="Класс сборки (дефазифицированный)", shrink=0.85)
    fig.suptitle("Выход системы по Разрешению и Компактности (Бюджет=135000 ₽, Требовательность=8, ГПУ=9)")
    fig.savefig(OUTPUT_DIR / "multi_input_heatmap_resolution_compactness.png", dpi=150)
    plt.close(fig)

    print(f"Графики сохранены в {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
