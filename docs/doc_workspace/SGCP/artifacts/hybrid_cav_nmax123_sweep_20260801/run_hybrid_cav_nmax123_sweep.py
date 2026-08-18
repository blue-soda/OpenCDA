# -*- coding: utf-8 -*-
"""Run the paired hybrid CAV-count/Nmax sweep: (5,1), (10,2), (15,3)."""

import importlib.util
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
BASE_SCRIPT = (
    REPO / "docs/doc_workspace/SGCP/artifacts"
    / "hybrid_cav_nmax_sweep_20260801/run_hybrid_cav_nmax_sweep.py")
ARTIFACT = (
    REPO / "docs/doc_workspace/SGCP/artifacts"
    / "hybrid_cav_nmax123_sweep_20260801")


def main():
    spec = importlib.util.spec_from_file_location("base_sweep", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARTIFACT = ARTIFACT
    module.NMAX_PAIRS = [(5, 1), (10, 2), (15, 3)]
    module.N_VALUES = [item[0] for item in module.NMAX_PAIRS]
    module.main()


if __name__ == "__main__":
    main()
