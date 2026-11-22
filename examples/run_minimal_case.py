from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.minegs.writeback_gate import run_v6_case

def main() -> None:
    eye = np.eye(4).tolist()
    case = {
        "case_id": "minimal-example",
        "reference": [{"id": "frame-0", "Tcw": eye}],
        "observed": [{"id": "frame-0", "Tcw": eye}],
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
    output = run_v6_case(case, arm_id="FULL")
    print(json.dumps({
        "state": output["state"],
        "committed": output["committed"],
        "certificate_states": [row["state"] for row in output["certificate"]],
        "rollback_exact": output["audit"]["rollback_exact"],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()


