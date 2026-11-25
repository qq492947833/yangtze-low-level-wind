from pathlib import Path
import os

CODE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("YANGTZE_WORK_DIR", "work")).resolve()
ROOT = WORK / "zwd"
OLD = WORK / "gradient" / "source_cache"
DATA = Path(os.environ.get("YANGTZE_DATA_DIR", "data")).resolve()
ZWD = Path(os.environ.get("YANGTZE_ZWD_FILE", str(DATA / "GPC_ZWD_2010_2025_hourly_global_0p25deg.nc")))
DEVICE = os.environ.get("YANGTZE_DEVICE", "cuda:0")
RUN = {"run_id": "paired_zwd"}
