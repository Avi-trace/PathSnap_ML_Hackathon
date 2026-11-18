"""Assemble the four teammate zips from this repository. Re-run it whenever results change.

usage: python team_packages/build_zips.py [output_dir]        (default: <repo parent>/team_zips)
Each zip holds START_HERE.md (that piece's brief), TEAM_README.md and the files the piece needs.
Large regenerable artifacts (cached tables, trained models) and node_modules are never included.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "team_zips"
SKIP_DIRS = {"__pycache__", "node_modules", "dist", ".git", ".vite"}
SKIP_PREFIX = ("artifacts/data/", "artifacts/models/", "team_packages/", "ui/", "client_handoff.zip", "gemini-code")
SKIP_SUFFIX = (".pyc", ".db", ".tmp")
SKIP_NAMES = ("oof_p_", "student_p_")

PYTHON_CODE = ["pathsnap", "scripts", "reports", "requirements.txt", "README.md", "PROGRESS.md", ".gitignore"]
PIECES = {
    "pathsnap_piece1_data_features": ("PIECE_1_DATA_FEATURES.md", PYTHON_CODE),
    "pathsnap_piece2_models_tuning": ("PIECE_2_MODELS_TUNING.md", PYTHON_CODE),
    "pathsnap_piece3_hmm_server": ("PIECE_3_HMM_SERVER.md", PYTHON_CODE + ["artifacts/final", "client_handoff/CLIENT_README.md"]),
    "pathsnap_piece4_client_ui": ("PIECE_4_CLIENT_UI.md", ["web", "client_handoff", "PROGRESS.md"]),
}


def wanted(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
        return False
    if rel.startswith(SKIP_PREFIX) or rel.endswith(SKIP_SUFFIX) or any(n in path.name for n in SKIP_NAMES):
        return False
    return path.is_file()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for name, (readme, includes) in PIECES.items():
        files = []
        for inc in includes:
            p = REPO / inc
            files += [p] if p.is_file() else sorted(p.rglob("*"))
        files = [f for f in files if wanted(f)]
        target = OUT / f"{name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f"{name}/{f.relative_to(REPO).as_posix()}")
            z.write(REPO / "team_packages" / readme, f"{name}/START_HERE.md")
            z.write(REPO / "team_packages" / "TEAM_README.md", f"{name}/TEAM_README.md")
            z.writestr(f"{name}/ZIP_INFO.txt", f"built {stamp} from the team lead's repository\nfiles: {len(files) + 3}\n"
                       "numbers are development results on student_test from the final bundle; ask for refreshed reports/ if anything is re-run\n")
        print(f"{target.name:42s} {len(files) + 3:4d} files  {target.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
