"""Run every test from the project folder, with src/ importable, as the scripts expect."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.chdir(ROOT)
