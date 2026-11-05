from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_validator_accepts_repository() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_repo.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_tracked_large_artifact_extensions_in_scaffold() -> None:
    forbidden = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".zip"}
    found = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix in forbidden]
    assert found == []
