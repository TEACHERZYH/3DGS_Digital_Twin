"""证据控制写回器的最小可审计实现。

该模块用于受控写回测试，不读取评价标签。所有比较臂共享同一
候选增量、copy-on-write 执行器、哈希和回滚逻辑；模块开关只改变对应的
证据检查。它不是完整矿井部署实现，外部前端需要先通过适配器门槛。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .observable_primitives import certificate_state, identifiable_subspace, qualify_sources


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V6Modules:
    """可开关的科学模块；所有臂保持相同候选输入。"""

    source: bool = True
    certificate: bool = True
    heff: bool = True
    protected: bool = True
    shadow: bool = True
    rollback: bool = True


def _source(problem: dict, modules: V6Modules) -> dict:
    reference = problem["reference"]
    observed = problem["observed"]
    if not modules.source:
        return {
            "status": "SOURCE_CHECK_DISABLED",
            "reasons": [],
            "accepted_ids": sorted(str(row["id"]) for row in observed),
            "max_translation_error": None,
            "max_rotation_error": None,
        }
    return qualify_sources(
        reference,
        observed,
        translation_tolerance=float(problem.get("translation_tolerance", 1e-6)),
        rotation_tolerance=float(problem.get("rotation_tolerance", 1e-6)),
    )


def _certificate(problem: dict, source_status: str, modules: V6Modules) -> list[dict]:
    lower = np.asarray(problem["lower"], dtype=np.float64)
    upper = np.asarray(problem["upper"], dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower/upper 必须是一维同形数组")
    threshold = float(problem["threshold"])
    feasible = bool(problem.get("model_feasible", True))
    mismatch = bool(problem.get("model_mismatch", False))
    harm = float(problem.get("harm_upper_bound", 0.0)) if modules.protected else 0.0
    harm_limit = float(problem.get("harm_limit", 0.0)) if modules.protected else 0.0
    result = []
    cert_source_status = "VALID_SOURCE" if source_status == "SOURCE_CHECK_DISABLED" else source_status
    for slot, (lo, hi) in enumerate(zip(lower, upper)):
        if modules.certificate:
            state = certificate_state(
                source_status=cert_source_status,
                lower=float(lo),
                upper=float(hi),
                threshold=threshold,
                model_feasible=feasible,
                model_mismatch=mismatch,
                harm_upper_bound=harm,
                harm_limit=harm_limit,
            )
        else:
            point = float((lo + hi) / 2.0)
            blocked = source_status not in {"VALID_SOURCE", "SOURCE_CHECK_DISABLED"} or not feasible or mismatch
            if blocked:
                state_name = "POINT_BLOCKED"
            elif modules.protected and harm > harm_limit:
                state_name = "POINT_HARM"
            elif abs(point - threshold) < float(problem.get("point_margin", 0.0)):
                state_name = "POINT_UNCERTAIN"
            else:
                state_name = "POINT_COMMIT"
            state = {
                "state": state_name,
                "lower": float(lo),
                "upper": float(hi),
                "threshold": threshold,
                "harm_upper_bound": harm,
                "harm_limit": harm_limit,
                "commit_allowed": state_name == "POINT_COMMIT",
                "certificate_mode": "POINT_ESTIMATE",
            }
        result.append({"slot": int(slot), **state})
    return result


def _heff(problem: dict, modules: V6Modules) -> dict:
    design = np.asarray(problem["J"], dtype=np.float64)
    nuisance = np.asarray(problem.get("B", np.zeros((design.shape[0], 0))), dtype=np.float64)
    if modules.heff:
        info = identifiable_subspace(design, nuisance, rtol=float(problem.get("rank_rtol", 1e-4)))
        return {**info, "mode": "IDENTIFIABLE_SUBSPACE"}
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    return {
        "H_eff": design.T @ design,
        "singular_values": singular,
        "nuisance_singular_values": np.empty(0),
        "rank": rank,
        "nullity": int(design.shape[1] - rank),
        "projected_design": design,
        "identifiable_right_singular_vectors": np.eye(design.shape[1]),
        "mode": "FULL_SPACE",
    }


def _state_from_certificates(certs: list[dict], modules: V6Modules, delta: np.ndarray, threshold: float) -> tuple[str, np.ndarray]:
    allowed = np.array([bool(row.get("commit_allowed")) for row in certs], dtype=bool)
    if allowed.shape != delta.shape:
        allowed = np.ones(delta.shape, dtype=bool)
    masked = np.where(allowed, delta, 0.0)
    if not modules.shadow:
        # E−SHADOW 保留同一候选与审计执行器，但对非空候选强制提交。
        return ("COMMIT" if np.any(np.abs(delta) > threshold) else "NO_CHANGE"), delta.copy()
    # 集合值证书是动作级授权：任一关键槽位仍有竞争解释、模型不可行或
    # 保护区超预算，整次动作进入 SHADOW，不能把其余槽位的可行增量单独写回。
    if certs and any(row.get("state") not in {"COMMIT", "POINT_COMMIT"} for row in certs):
        return "SHADOW", np.zeros_like(delta)
    if np.any(np.abs(masked) > threshold):
        return "COMMIT", masked
    return "SHADOW", masked


def run_v6_case(problem: dict, modules: V6Modules = V6Modules(), *, arm_id: str = "FULL-N") -> dict:
    """运行一个压力案例并返回可序列化的动作、状态和审计证据。"""
    delta = np.asarray(problem["candidate_delta"], dtype=np.float64)
    if delta.ndim != 1 or not np.isfinite(delta).all():
        raise ValueError("candidate_delta 必须是一维有限数组")
    theta0 = np.asarray(problem["theta0"], dtype=np.float64)
    if theta0.shape != delta.shape:
        raise ValueError("theta0 与 candidate_delta 形状必须一致")
    source = _source(problem, modules)
    source_ok = source["status"] in {"VALID_SOURCE", "SOURCE_CHECK_DISABLED"}
    info = _heff(problem, modules)
    rank_ok = int(info["rank"]) > 0
    certs = _certificate(problem, source["status"], modules)
    threshold = float(problem["threshold"])
    candidate = delta.copy() if source_ok and rank_ok else np.zeros_like(delta)
    if modules.heff and rank_ok:
        basis = np.asarray(info["identifiable_right_singular_vectors"], dtype=np.float64)
        if basis.ndim == 2 and basis.shape[1] == candidate.size:
            candidate = basis.T @ (basis @ candidate)
    state, masked = _state_from_certificates(certs, modules, candidate, threshold)
    if modules.source and not source_ok:
        state, masked = "INVALID_SOURCE", np.zeros_like(delta)
    if modules.heff and not rank_ok:
        state, masked = ("ABSTAIN" if modules.shadow else "COMMIT"), (np.zeros_like(delta) if modules.shadow else candidate)
    if modules.protected and float(problem.get("harm_upper_bound", 0.0)) > float(problem.get("harm_limit", 0.0)):
        state, masked = ("SHADOW" if modules.shadow else "COMMIT"), (np.zeros_like(delta) if modules.shadow else candidate)
    before = {"theta": theta0.tolist()}
    trusted = copy.deepcopy(before)
    candidate_payload = {"theta_delta": masked.tolist(), "arm_id": arm_id, "case_id": str(problem.get("case_id", "unknown"))}
    candidate_hash = _json_hash(candidate_payload)
    before_hash = _json_hash(trusted)
    transaction_id = hashlib.sha256(f"{arm_id}:{problem.get('case_id', 'unknown')}:{candidate_hash}".encode()).hexdigest()
    committed = state == "COMMIT" and bool(np.any(np.abs(masked) > threshold))
    if committed:
        trusted["theta"] = (theta0 + masked).tolist()
    after_commit_hash = _json_hash(trusted)
    rollback_exact = True
    after_rollback_hash = after_commit_hash
    if modules.rollback and committed:
        trusted = copy.deepcopy(before)
        after_rollback_hash = _json_hash(trusted)
        rollback_exact = after_rollback_hash == before_hash
    elif not modules.rollback:
        after_rollback_hash = None
        rollback_exact = None
    audit = {
        "transaction_id": transaction_id,
        "candidate_layer_hash": candidate_hash,
        "trusted_layer_hash_before": before_hash,
        "trusted_layer_hash_after_commit": after_commit_hash,
        "trusted_layer_hash_after_rollback": after_rollback_hash,
        "rollback_exact": rollback_exact,
        "state_transition": ["CANDIDATE_CREATED", state] + (["ROLLED_BACK"] if modules.rollback and committed else []),
        "audit_complete": bool(modules.rollback and candidate_hash and transaction_id),
    }
    return {
        "case_id": str(problem.get("case_id", "unknown")),
        "arm_id": arm_id,
        "state": state,
        "committed": bool(committed),
        "delta": masked.tolist(),
        "source": source,
        "certificate": certs,
        "identifiability": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items() if k not in {"H_eff", "projected_design"}},
        "audit": audit,
        "modules": modules.__dict__,
    }

