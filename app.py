"""Графическая оболочка (веб-приложение) нечёткой системы логического
вывода - ЛР №3.

Загружает любую базу правил из rules/*.json (или свой JSON-файл), даёт
интерактивно выбирать импликацию, оператор агрегации, механизм вывода и
значения входных переменных, и строит графики, в т.ч. 3D-поверхности
(нечёткое отношение для систем с одним входом; срез выхода по двум
выбранным входам для систем с несколькими) и представление N-мерной
зависимости (parallel coordinates + локальные срезы чувствительности по
каждому входу).

Запуск: uv run streamlit run app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fuzzy_lr3 import composition, truth_level  # noqa: E402
from fuzzy_lr3.aggregation import AGGREGATION_TITLES, DEFAULT_AGGREGATION  # noqa: E402
from fuzzy_lr3.defuzzify import centroid  # noqa: E402
from fuzzy_lr3.implications import IMPLICATION_TITLES  # noqa: E402
from fuzzy_lr3.parser import load_rulebase  # noqa: E402
from fuzzy_lr3.variable import RuleBase  # noqa: E402

RULES_DIR = PROJECT_ROOT / "rules"
IMPLICATIONS = list(IMPLICATION_TITLES)
AGGREGATIONS = [None, *AGGREGATION_TITLES]  # None - подходящая для импликации

st.set_page_config(page_title="Нечёткая система логического вывода", layout="wide")


# --------------------------------------------------------------------------
# Загрузка и кэш
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Загрузка базы правил...")
def load(path_str: str, universe_points: int, mtime: float) -> RuleBase:
    # mtime входит в ключ кэша: изменённый файл правил перечитывается сразу.
    return load_rulebase(path_str, universe_points=universe_points)


@st.cache_data(show_spinner="Строим нечёткое отношение...")
def relation_cached(path_str: str, universe_points: int, mtime: float, implication: str, aggregation: str | None) -> np.ndarray:
    rb = load(path_str, universe_points, mtime)
    return composition.build_relation(rb, implication=implication, aggregation=aggregation)


def rule_base_file_picker() -> Path:
    files = sorted(RULES_DIR.glob("*.json"))
    labels = {f: f.name for f in files}

    uploaded = st.sidebar.file_uploader("...или загрузить свой JSON", type="json")
    if uploaded is not None:
        custom_path = PROJECT_ROOT / "output" / f"_uploaded_{uploaded.name}"
        custom_path.parent.mkdir(exist_ok=True)
        custom_path.write_bytes(uploaded.getvalue())
        return custom_path

    choice = st.sidebar.selectbox("База правил", files, format_func=lambda f: labels[f])
    return choice


# --------------------------------------------------------------------------
# Вспомогательные вычисления
# --------------------------------------------------------------------------

def slider_step(span: float) -> float:
    """Круглый шаг ползунка: 1000 ₽ для бюджета, 0.1 для шкалы 0..10."""
    return 10.0 ** (math.floor(math.log10(span)) - 2)


def term_curves_df(variable) -> pd.DataFrame:
    rows = []
    for term, mf in variable.terms.items():
        mu = mf(variable.universe)
        rows.append(pd.DataFrame({"x": variable.universe, "mu": mu, "Терм": term}))
    return pd.concat(rows, ignore_index=True)


def truth_level_single_input(
    rb: RuleBase, x0: float, input_membership: np.ndarray, fuzzy: bool, *, implication: str, aggregation: str | None,
) -> np.ndarray:
    name = rb.input_vars[0]
    if fuzzy:
        return truth_level.infer_fuzzy(rb, {name: input_membership}, implication=implication, aggregation=aggregation).output
    return truth_level.infer(rb, {name: x0}, implication=implication, aggregation=aggregation).output


def comparison_table_single_input(
    rb: RuleBase, x0: float, input_membership: np.ndarray, fuzzy: bool, aggregation: str | None,
) -> pd.DataFrame:
    output_var = rb.output_variable

    rows = []
    for impl in IMPLICATIONS:
        relation = composition.build_relation(rb, implication=impl, aggregation=aggregation)
        b_maxmin = composition.max_min_compose(input_membership, relation)
        b_maxprod = composition.max_prod_compose(input_membership, relation)
        b_truth = truth_level_single_input(rb, x0, input_membership, fuzzy, implication=impl, aggregation=aggregation)
        rows.append({
            "Импликация": IMPLICATION_TITLES[impl],
            "CRI max-min": round(centroid(output_var.universe, b_maxmin), 3),
            "CRI max-произв.": round(centroid(output_var.universe, b_maxprod), 3),
            "Уровни истинности": round(centroid(output_var.universe, b_truth), 3),
        })
    return pd.DataFrame(rows)


def comparison_table_multi_input(rb: RuleBase, inputs: dict[str, float], aggregation: str | None) -> pd.DataFrame:
    output_var = rb.output_variable
    rows = []
    for impl in IMPLICATIONS:
        result = truth_level.infer(rb, inputs, implication=impl, aggregation=aggregation)
        rows.append({"Импликация": IMPLICATION_TITLES[impl], "Выход": round(centroid(output_var.universe, result.output), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("Нечёткая система логического вывода")
st.caption("ЛР №3 - интерактивная оболочка: импликации, композиция, агрегация, 2D/3D/nD-визуализация")

rules_path = rule_base_file_picker()
universe_points = st.sidebar.slider("Узлов сетки (точность/скорость)", 61, 801, 241, step=20)

implication = st.sidebar.selectbox("Импликация", IMPLICATIONS, format_func=lambda i: IMPLICATION_TITLES[i])
aggregation = st.sidebar.selectbox(
    "Агрегация правил", AGGREGATIONS,
    format_func=lambda a: AGGREGATION_TITLES[a] if a else f"по импликации ({DEFAULT_AGGREGATION[implication]})",
)

try:
    rules_mtime = rules_path.stat().st_mtime
    rb = load(str(rules_path), universe_points, rules_mtime)
except Exception as e:  # noqa: BLE001 - показываем любую ошибку загрузки пользователю
    st.error(f"Не удалось загрузить базу правил: {e}")
    st.stop()

output_var = rb.output_variable
n_inputs = len(rb.input_vars)

st.sidebar.markdown("---")
st.sidebar.subheader("Входные значения")

input_values: dict[str, float] = {}
for var_name in rb.input_vars:
    var = rb.variables[var_name]
    lo, hi = float(var.universe[0]), float(var.universe[-1])
    step = slider_step(hi - lo)
    input_values[var_name] = st.sidebar.slider(var_name, lo, hi, round((lo + hi) / 2 / step) * step, step=step)

if n_inputs == 1:
    st.sidebar.markdown("---")
    fuzzy_input_mode = st.sidebar.checkbox("Нечёткий вход (не точка, а «примерно»)")
    spread = None
    if fuzzy_input_mode:
        var = rb.variables[rb.input_vars[0]]
        span = float(var.universe[-1] - var.universe[0])
        spread = st.sidebar.slider("Разброс входа (±)", span * 0.01, span * 0.3, span * 0.1)

st.subheader(f"База правил: {rules_path.name}  ·  {len(rb.rules)} правил  ·  {n_inputs} вход(а/ов)")
with st.expander("Показать правила"):
    for rule in rb.rules:
        st.markdown(f"**#{rule.index + 1}:** {rule.describe(rb.output_var)}")

tab_terms, tab_output, tab_3d, tab_nd = st.tabs(["Термы", "Вывод", "3D", "nD"])

# --- Термы ------------------------------------------------------------
with tab_terms:
    cols = st.columns(n_inputs + 1)
    for col, var_name in zip(cols, [*rb.input_vars, rb.output_var]):
        var = rb.variables[var_name]
        df = term_curves_df(var)
        fig = go.Figure()
        for term in var.terms:
            sub = df[df["Терм"] == term]
            fig.add_trace(go.Scatter(x=sub["x"], y=sub["mu"], mode="lines", name=term))
        if var_name in input_values:
            fig.add_vline(x=input_values[var_name], line_dash="dash", line_color="gray")
        fig.update_layout(title=var_name, height=350, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[-0.05, 1.05])
        col.plotly_chart(fig, width="stretch")

# --- Вывод --------------------------------------------------------------
with tab_output:
    if n_inputs == 1:
        var_name = rb.input_vars[0]
        input_var = rb.variables[var_name]
        x0 = input_values[var_name]

        if fuzzy_input_mode:
            input_membership = np.clip(1 - np.abs(input_var.universe - x0) / spread, 0, 1)
        else:
            input_membership = composition.singleton(input_var.universe, x0)

        relation = relation_cached(str(rules_path), universe_points, rules_mtime, implication, aggregation)
        b_maxmin = composition.max_min_compose(input_membership, relation)
        b_maxprod = composition.max_prod_compose(input_membership, relation)
        b_truth = truth_level_single_input(
            rb, x0, input_membership, fuzzy_input_mode, implication=implication, aggregation=aggregation)

        c1, c2, c3 = st.columns(3)
        c1.metric("CRI max-min", f"{centroid(output_var.universe, b_maxmin):.2f}")
        c2.metric("CRI max-произведение", f"{centroid(output_var.universe, b_maxprod):.2f}")
        c3.metric("Уровни истинности", f"{centroid(output_var.universe, b_truth):.2f}")

        fig = go.Figure()
        for label, b in [("CRI max-min", b_maxmin), ("CRI max-произв.", b_maxprod), ("Уровни истинности", b_truth)]:
            y = centroid(output_var.universe, b)
            fig.add_trace(go.Scatter(x=output_var.universe, y=b, mode="lines", name=f"{label} ({y:.2f})"))
        fig.update_layout(title="Выходное нечёткое множество", xaxis_title=output_var.name, yaxis_title="μ", height=420)
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Сравнение по всем импликациям** (для текущего входа и агрегации):")
        st.dataframe(comparison_table_single_input(rb, x0, input_membership, fuzzy_input_mode, aggregation), width="stretch", hide_index=True)

    else:
        result = truth_level.infer(rb, input_values, implication=implication, aggregation=aggregation)
        y_crisp = centroid(output_var.universe, result.output)

        st.metric("Дефазифицированный выход", f"{y_crisp:.2f}")
        if result.output.max() < 1e-9:
            st.warning("Выходное множество пусто: правила противоречат друг другу, значение выхода не информативно.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=output_var.universe, y=result.output, mode="lines", name="B'(y)"))
        fig.add_vline(x=y_crisp, line_dash="dash", line_color="gray")
        fig.update_layout(title="Выходное нечёткое множество", xaxis_title=output_var.name, yaxis_title="μ", height=420)
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Степени истинности предпосылок правил (уровни отсечения α):**")
        strength_rows = [
            {"Правило": f"#{rule.index + 1}", "Условие": rule.describe(rb.output_var), "α": round(result.rule_strengths[rule.index], 3)}
            for rule in rb.rules
        ]
        st.dataframe(pd.DataFrame(strength_rows), width="stretch", hide_index=True)

        st.markdown("**Сравнение по всем импликациям** (для текущих входов и агрегации):")
        st.dataframe(comparison_table_multi_input(rb, input_values, aggregation), width="stretch", hide_index=True)

# --- 3D -------------------------------------------------------------------
with tab_3d:
    if n_inputs == 1:
        var_name = rb.input_vars[0]
        input_var = rb.variables[var_name]
        relation = relation_cached(str(rules_path), universe_points, rules_mtime, implication, aggregation)

        fig = go.Figure(data=[go.Surface(
            x=input_var.universe, y=output_var.universe, z=relation.T,
            colorscale="Viridis", showscale=True,
        )])
        fig.update_layout(
            title=f"Нечёткое отношение R(x, y) - импликация «{IMPLICATION_TITLES[implication]}»",
            scene=dict(xaxis_title=var_name, yaxis_title=output_var.name, zaxis_title="μ"),
            height=650, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Поверхность показывает отношение R(x, y), построенное объединением всех правил "
            "выбранной импликацией и агрегацией. Срез по x=x0 (красная линия на вкладке «Вывод» "
            "по сути) и даёт выходное множество при чётком входе x0."
        )
    else:
        c1, c2, c3 = st.columns(3)
        var_x = c1.selectbox("Ось X", rb.input_vars, index=0)
        remaining = [v for v in rb.input_vars if v != var_x]
        var_y = c2.selectbox("Ось Y", remaining, index=0)
        grid_n = c3.slider("Разрешение сетки", 15, 60, 30)

        vx, vy = rb.variables[var_x], rb.variables[var_y]
        xs = np.linspace(vx.universe[0], vx.universe[-1], grid_n)
        ys = np.linspace(vy.universe[0], vy.universe[-1], grid_n)

        fixed = {k: v for k, v in input_values.items() if k not in (var_x, var_y)}
        z = np.zeros((grid_n, grid_n))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                inputs = {**fixed, var_x: xv, var_y: yv}
                result = truth_level.infer(rb, inputs, implication=implication, aggregation=aggregation)
                z[i, j] = centroid(output_var.universe, result.output)

        fig = go.Figure(data=[go.Surface(x=xs, y=ys, z=z, colorscale="Viridis", showscale=True)])
        fixed_desc = ", ".join(f"{k}={v:.1f}" for k, v in fixed.items()) or "-"
        fig.update_layout(
            title=f"Выход({var_x}, {var_y}) при фиксированных: {fixed_desc}",
            scene=dict(xaxis_title=var_x, yaxis_title=var_y, zaxis_title=output_var.name),
            height=650, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Остальные входные переменные зафиксированы на значениях из боковой панели.")

# --- nD ---------------------------------------------------------------
with tab_nd:
    st.markdown("#### Локальная чувствительность по каждому входу")
    st.caption(
        "Для каждой входной переменной: выход при развёртке этой переменной по всему "
        "диапазону, остальные входы зафиксированы на текущих значениях с боковой панели."
    )
    fig = make_subplots(rows=1, cols=n_inputs, subplot_titles=rb.input_vars)
    for col_idx, var_name in enumerate(rb.input_vars, start=1):
        var = rb.variables[var_name]
        sweep = np.linspace(var.universe[0], var.universe[-1], 60)
        ys = []
        for xv in sweep:
            inputs = {**input_values, var_name: xv}
            result = truth_level.infer(rb, inputs, implication=implication, aggregation=aggregation)
            ys.append(centroid(output_var.universe, result.output))
        fig.add_trace(go.Scatter(x=sweep, y=ys, mode="lines", showlegend=False), row=1, col=col_idx)
        fig.add_vline(x=input_values[var_name], line_dash="dash", line_color="gray", row=1, col=col_idx)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    if n_inputs >= 2:
        st.markdown("#### Параллельные координаты (глобальная N-мерная картина)")
        st.caption(
            f"Случайная выборка точек из всего пространства входов ({n_inputs}-мерного), "
            "прогнанная через систему; каждая линия - один набор входов и соответствующий "
            "выход, цвет - величина выхода."
        )
        n_samples = st.slider("Число точек выборки", 50, 1000, 300)
        rng = np.random.default_rng(42)

        sample_rows = []
        for _ in range(n_samples):
            sampled_inputs = {
                v: float(rng.uniform(rb.variables[v].universe[0], rb.variables[v].universe[-1]))
                for v in rb.input_vars
            }
            result = truth_level.infer(rb, sampled_inputs, implication=implication, aggregation=aggregation)
            sample_rows.append({**sampled_inputs, output_var.name: centroid(output_var.universe, result.output)})

        df = pd.DataFrame(sample_rows)
        dims = [dict(label=col, values=df[col]) for col in df.columns]
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df[output_var.name], colorscale="Viridis", showscale=True),
            dimensions=dims,
        ))
        fig.update_layout(height=480, margin=dict(l=40, r=40, t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Параллельные координаты осмысленны при 2+ входных переменных - переключитесь на базу правил с несколькими входами.")
