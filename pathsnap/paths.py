"""Machine-specific locations.

Set PATHSNAP_DATA_ROOT to the folder that contains the `train/` and `student_test/` partitions
(the "Student folder/Student folder" directory of the dataset). Default: the dataset folder next to this repository.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", ROOT.parent / "Student folder" / "Student folder"))
