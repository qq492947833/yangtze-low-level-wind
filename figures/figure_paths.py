import os
from pathlib import Path


def setting(name, default):
    return Path(os.environ.get(name, default)).expanduser()


DATA = setting('YANGTZE_DATA_DIR', 'data')
RESULTS = setting('YANGTZE_RESULTS_DIR', 'results')
WORK = setting('YANGTZE_WORK_DIR', 'work')
FIGURES = setting('YANGTZE_FIGURE_DIR', WORK / 'figures')
FIGURES.mkdir(parents=True, exist_ok=True)
