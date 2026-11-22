"""可重复的布局压力生成器。"""
from __future__ import annotations

import numpy as np


PROFILES = {
    0: ("low_signal", 0.35),
    1: ("moderate_signal", 0.65),
    2: ("nominal_signal", 1.00),
    3: ("high_signal", 1.25),
    4: ("low_signal_shifted", 0.45),
    5: ("high_signal_shifted", 1.15),
}


def profile_values(layout: int) -> tuple[str, float]:
    """为确认集提供可重复的扩展剖面，不改变六种已冻结基本压力类型。"""
    if layout in PROFILES:
        return PROFILES[layout]
    base_name, base_scale = PROFILES[layout % 6]
    cycle = layout // 6
    multiplier = (0.90, 1.08, 0.96)[(cycle - 1) % 3]
    return f"{base_name}_cycle{cycle}", base_scale * multiplier


def make_profiled_case(layout: int, row: dict, *, seed_base: int) -> tuple[dict, bool]:
    """生成一个布局压力案例；规则对应 v6.2 计划，标签不参与生成。"""
    profile_name, noncommit_scale = profile_values(layout)
    seed = seed_base + layout
    rng = np.random.default_rng(seed + int(row["condition_id"][1:]))
    base_delta = np.array([0.20, 0.12], dtype=float) + rng.normal(0, 0.005, 2)
    eye = np.eye(4)
    observed = eye.copy()
    source_status = row["source_status"]
    if source_status == "TRANSLATION_MISMATCH":
        observed[0, 3] = 0.1
    elif source_status == "ROTATION_MISMATCH":
        observed[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    # 固定规则：只有非提交压力条件改变候选幅度；合法提交条件保持充分幅度。
    delta = base_delta.copy()
    if source_status in {"TRANSLATION_MISMATCH", "ROTATION_MISMATCH", "ID_MISMATCH"}:
        delta = base_delta * noncommit_scale
    lower, upper = [0.2, 0.2], [0.3, 0.3]
    point_margin = 0.01
    if row["interval_case"] == "cross_threshold":
        base_offsets = [-0.020, -0.005, 0.010, 0.025, -0.012, 0.018]
        offset = base_offsets[layout % 6] + 0.002 * (layout // 6)
        lower = [0.0 + offset, 0.2 + offset]
        upper = [0.1 + offset, 0.3 + offset]
    if row["view_factor"] == "occluded":
        base_levels = [0.025, 0.040, 0.060, 0.080, 0.035, 0.070]
        level = base_levels[layout % 6] * (1.0 + 0.03 * (layout // 6))
        delta = np.array([level, level * 0.8], dtype=float)
        lower, upper = [0.0, 0.0], [0.1, 0.1]
    J = np.eye(2)
    if row["rank_profile"] == "rank_deficient":
        J = np.array([[1.0, 0.0], [0.0, 0.0]])
        base_levels = [0.025, 0.040, 0.060, 0.080, 0.035, 0.070]
        delta = np.array([0.0, base_levels[layout % 6] * (1.0 + 0.03 * (layout // 6))])
    elif row["rank_profile"] == "near_null":
        J = np.array([[1.0, 0.0], [0.0, 1e-6]])
        base_levels = [0.025, 0.040, 0.060, 0.080, 0.035, 0.070]
        delta = np.array([0.0, base_levels[layout % 6] * (1.0 + 0.03 * (layout // 6))])
    harm = {"within_budget": 0.05, "over_budget": 0.12 + 0.01 * (layout % 3)}.get(row["protected_harm"], 0.0)
    case = {
        "case_id": f"v6_2_{seed_base}_{layout:02d}_{row['condition_id']}",
        "layout_profile": profile_name,
        "reference": [{"id": "f0", "Tcw": eye.tolist()}],
        "observed": [{"id": "other" if source_status == "ID_MISMATCH" else "f0", "Tcw": observed.tolist()}],
        "J": J.tolist(), "B": np.zeros((2, 0)).tolist(), "theta0": [0.0, 0.0],
        "candidate_delta": delta.tolist(), "lower": lower, "upper": upper,
        "threshold": 0.05, "point_margin": point_margin,
        "harm_upper_bound": harm, "harm_limit": 0.1,
    }
    expected = row["condition_id"] in {"C01", "C02", "C03", "C08"}
    return case, expected

