"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_DB = ROOT / "pricematch.db"

# User agent for the collector. Identify yourself honestly; do not rotate.
USER_AGENT = "pricematch-student-project/0.1 (+respectful research collector)"

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)
