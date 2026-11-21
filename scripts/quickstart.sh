#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pricematch.pipeline seed --db pricematch.db
python -m pricematch.pipeline eval --compare
python -m pricematch.pipeline agent --db pricematch.db "boAt airdopes under 1500"
pytest -q
echo "All good. Try the demo: pip install -r requirements-ui.txt && python -m pricematch.pipeline demo"
