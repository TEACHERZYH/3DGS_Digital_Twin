from __future__ import annotations

import csv
from pathlib import Path

from src.minegs.pressure_cases import make_profiled_case
from src.minegs.writeback_gate import V6Modules, run_v6_case


MATRIX = Path(__file__).parents[1] / "configs" / "case_matrix.csv"


def test_profiled_layouts_keep_full_n_truth_and_create_pressure_variation():
    rows = list(csv.DictReader(MATRIX.open(encoding="utf-8-sig", newline="")))
    wrong = []
    for layout in range(16):
        for row in rows:
            case, expected = make_profiled_case(layout, row, seed_base=740000)
            out = run_v6_case(case, V6Modules(), arm_id="FULL-N")
            assert out["committed"] == expected
            wrong.append((layout, row["condition_id"], int(out["committed"] and not expected)))
    assert sum(x[2] for x in wrong) == 0


def test_profiled_case_input_changes_by_layout():
    rows = list(csv.DictReader(MATRIX.open(encoding="utf-8-sig", newline="")))
    cases = [make_profiled_case(i, rows[6], seed_base=720000)[0] for i in range(4)]
    assert len({tuple(c["candidate_delta"]) for c in cases}) > 1

