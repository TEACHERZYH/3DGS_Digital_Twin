"""可审计更新的基础实现。

本模块只实现来源资格、可辨识子空间、集合值证书和可撤销事务；
它不读取评价标签，也不修改外部数据。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


def _finite_matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限的 {shape} 矩阵")
    return array


def _rotation_distance(a, b):
    delta = np.asarray(a, dtype=np.float64)[:3, :3] @ np.asarray(b, dtype=np.float64)[:3, :3].T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _is_rigid_pose(value):
    rotation = value[:3, :3]
    return bool(
        np.allclose(value[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
        and np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-6)
    )


def qualify_sources(reference, observed, *, translation_tolerance=1e-6, rotation_tolerance=1e-6):
    """核对逐帧来源；不一致时返回 INVALID_SOURCE，不生成后验。"""
    if translation_tolerance < 0 or rotation_tolerance < 0:
        raise ValueError("来源容差不能为负")
    ref = {str(row["id"]): _finite_matrix(row["Tcw"], (4, 4), "reference Tcw") for row in reference}
    obs = {str(row["id"]): _finite_matrix(row["Tcw"], (4, 4), "observed Tcw") for row in observed}
    reasons = []
    if any(not _is_rigid_pose(value) for value in (*ref.values(), *obs.values())):
        reasons.append("invalid_pose_matrix")
    if set(ref) != set(obs):
        reasons.append("source_id_mismatch")
    common = sorted(set(ref) & set(obs))
    translation_errors = []
    rotation_errors = []
    for frame_id in common:
        translation_errors.append(float(np.linalg.norm(ref[frame_id][:3, 3] - obs[frame_id][:3, 3])))
        rotation_errors.append(_rotation_distance(ref[frame_id], obs[frame_id]))
    max_translation = max(translation_errors, default=0.0)
    max_rotation = max(rotation_errors, default=0.0)
    if max_translation > translation_tolerance:
        reasons.append("translation_mismatch")
    if max_rotation > rotation_tolerance:
        reasons.append("rotation_mismatch")
    return {
        "status": "VALID_SOURCE" if not reasons else "INVALID_SOURCE",
        "reasons": reasons,
        "accepted_ids": common if not reasons else [],
        "max_translation_error": max_translation,
        "max_rotation_error": max_rotation,
        "translation_tolerance": float(translation_tolerance),
        "rotation_tolerance": float(rotation_tolerance),
    }


def identifiable_subspace(A, B, *, rtol=1e-4):
    """返回消除线性干扰后的 H_eff、秩和零空间投影信息。"""
    design = np.asarray(A, dtype=np.float64)
    nuisance = np.asarray(B, dtype=np.float64)
    if design.ndim != 2 or nuisance.ndim != 2 or design.shape[0] != nuisance.shape[0]:
        raise ValueError("A/B 必须是具有相同行数的二维矩阵")
    if not np.isfinite(design).all() or not np.isfinite(nuisance).all() or not 0 < rtol <= 1:
        raise ValueError("A/B 必须有限，rtol 必须位于 (0,1]")
    if nuisance.shape[1]:
        u_b, s_b, _ = np.linalg.svd(nuisance, full_matrices=False)
        keep_b = s_b >= rtol * s_b[0] if s_b.size and s_b[0] > 0 else np.zeros(0, dtype=bool)
        q = u_b[:, keep_b]
        projected = design - q @ (q.T @ design) if q.size else design.copy()
    else:
        s_b = np.empty(0, dtype=np.float64)
        projected = design.copy()
    u, singular, vt = np.linalg.svd(projected, full_matrices=False)
    rank = int(np.count_nonzero(singular >= rtol * singular[0])) if singular.size and singular[0] > 0 else 0
    h_eff = projected.T @ projected
    return {
        "H_eff": h_eff,
        "singular_values": singular,
        "nuisance_singular_values": s_b,
        "rank": rank,
        "nullity": int(design.shape[1] - rank),
        "projected_design": projected,
        "identifiable_right_singular_vectors": vt[:rank],
    }


def certificate_state(*, source_status, lower, upper, threshold, model_feasible=True,
                      model_mismatch=False, harm_upper_bound=0.0, harm_limit=0.0):
    """把来源、集合区间和最坏损害映射到可审计动作状态。"""
    if source_status != "VALID_SOURCE":
        state = "INVALID_SOURCE"
    elif model_mismatch:
        state = "MODEL_MISMATCH"
    elif not model_feasible:
        state = "NO_FEASIBLE_MODEL"
    elif not np.isfinite([lower, upper, threshold, harm_upper_bound, harm_limit]).all() or lower > upper:
        raise ValueError("集合区间和损害界必须有限且 lower <= upper")
    elif lower <= threshold <= upper:
        state = "UNIDENTIFIED"
    elif harm_upper_bound > harm_limit:
        state = "SHADOW"
    else:
        state = "COMMIT"
    return {
        "state": state,
        "lower": float(lower),
        "upper": float(upper),
        "threshold": float(threshold),
        "harm_upper_bound": float(harm_upper_bound),
        "harm_limit": float(harm_limit),
        "commit_allowed": state == "COMMIT",
    }


@dataclass
class TrustedCandidateStore:
    """可信层与候选层分离的可撤销事务容器。"""

    trusted: dict
    candidate: dict = field(default_factory=dict)
    _snapshot: dict | None = field(default=None, init=False, repr=False)

    def stage(self, candidate_id, payload):
        if candidate_id in self.candidate:
            raise ValueError("candidate_id 已存在")
        self.candidate[str(candidate_id)] = deepcopy(payload)

    def commit(self, certificate, candidate_id):
        if not certificate.get("commit_allowed") or candidate_id not in self.candidate:
            raise ValueError("证书不允许提交或候选不存在")
        self._snapshot = deepcopy(self.trusted)
        self.trusted[str(candidate_id)] = deepcopy(self.candidate[candidate_id])
        return {"state": "COMMIT", "candidate_id": str(candidate_id)}

    def rollback(self):
        if self._snapshot is None:
            raise ValueError("没有可撤销提交")
        self.trusted.clear()
        self.trusted.update(deepcopy(self._snapshot))
        self._snapshot = None
        return {"state": "ROLLED_BACK", "trusted": deepcopy(self.trusted)}


