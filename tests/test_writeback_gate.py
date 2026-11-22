import numpy as np

from src.minegs.writeback_gate import V6Modules, run_v6_case


def base_case(**overrides):
    eye = np.eye(4).tolist()
    data = {
        "case_id": "fixture",
        "reference": [{"id": "f0", "Tcw": eye}],
        "observed": [{"id": "f0", "Tcw": eye}],
        "J": np.eye(2).tolist(),
        "B": np.zeros((2, 0)).tolist(),
        "theta0": [0.0, 0.0],
        "candidate_delta": [0.2, 0.0],
        "lower": [0.2, 0.2],
        "upper": [0.3, 0.3],
        "threshold": 0.05,
        "harm_upper_bound": 0.0,
        "harm_limit": 0.0,
    }
    data.update(overrides)
    return data


def test_full_n_commits_and_exactly_rolls_back():
    out = run_v6_case(base_case(), arm_id="FULL-N")
    assert out["state"] == "COMMIT"
    assert out["committed"] is True
    assert out["audit"]["transaction_id"]
    assert out["audit"]["rollback_exact"] is True
    assert out["audit"]["trusted_layer_hash_before"] == out["audit"]["trusted_layer_hash_after_rollback"]


def test_source_module_blocks_translation_mismatch_but_no_source_can_leak():
    shifted = np.eye(4)
    shifted[0, 3] = 0.1
    problem = base_case(observed=[{"id": "f0", "Tcw": shifted.tolist()}])
    blocked = run_v6_case(problem, V6Modules(), arm_id="FULL-N")
    leaked = run_v6_case(problem, V6Modules(source=False), arm_id="NO-SOURCE")
    assert blocked["state"] == "INVALID_SOURCE"
    assert blocked["committed"] is False
    assert leaked["committed"] is True


def test_certificate_module_exposes_interval_crossing():
    problem = base_case(lower=[0.0, 0.2], upper=[0.1, 0.3])
    full = run_v6_case(problem, V6Modules(), arm_id="FULL-N")
    no_cert = run_v6_case(problem, V6Modules(certificate=False), arm_id="NO-CERT")
    assert any(row["state"] == "UNIDENTIFIED" for row in full["certificate"])
    assert no_cert["committed"] is True


def test_heff_blocks_rank_deficient_direction():
    problem = base_case(J=[[1.0, 0.0], [0.0, 0.0]], candidate_delta=[0.0, 0.2], lower=[0.2, 0.2], upper=[0.3, 0.3])
    full = run_v6_case(problem, V6Modules(), arm_id="FULL-N")
    no_heff = run_v6_case(problem, V6Modules(heff=False), arm_id="NO-HEFF")
    assert full["identifiability"]["nullity"] == 1
    assert full["committed"] is False
    assert no_heff["committed"] is True


def test_protected_module_blocks_over_budget_harm():
    problem = base_case(harm_upper_bound=0.2, harm_limit=0.1)
    full = run_v6_case(problem, V6Modules(), arm_id="FULL-N")
    no_protected = run_v6_case(problem, V6Modules(protected=False), arm_id="NO-PROTECTED")
    assert full["state"] == "SHADOW"
    assert full["committed"] is False
    assert no_protected["committed"] is True


def test_shadow_removal_forces_commit_for_uncertain_candidate():
    problem = base_case(lower=[0.0, 0.0], upper=[0.1, 0.1], candidate_delta=[0.2, 0.0])
    full = run_v6_case(problem, V6Modules(), arm_id="FULL-N")
    no_shadow = run_v6_case(problem, V6Modules(shadow=False), arm_id="NO-SHADOW")
    assert full["state"] == "SHADOW"
    assert no_shadow["state"] == "COMMIT"
    assert no_shadow["committed"] is True


def test_rollback_ablation_is_explicitly_incomplete():
    out = run_v6_case(base_case(), V6Modules(rollback=False), arm_id="ROLLBACK-FIXTURE")
    assert out["audit"]["rollback_exact"] is None
    assert out["audit"]["trusted_layer_hash_after_rollback"] is None
    assert out["audit"]["audit_complete"] is False

