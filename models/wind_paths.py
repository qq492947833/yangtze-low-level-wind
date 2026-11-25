import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('YANGTZE_DATA_DIR', BASE_DIR / 'data'))
MODEL_DIR = Path(os.environ.get('YANGTZE_MODEL_DIR', BASE_DIR / 'model'))
RESULT_DIR = Path(os.environ.get('YANGTZE_RESULTS_DIR', BASE_DIR / 'results'))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
