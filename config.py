from pathlib import Path
import os
from ultralytics import settings

ROOT=Path(__file__).resolve().parent
MODEL_DIR=ROOT/"models"; RUNS_DIR=ROOT/"runs"; DATASETS_DIR=ROOT/"datasets"
MODEL_DIR.mkdir(exist_ok=True); RUNS_DIR.mkdir(exist_ok=True); DATASETS_DIR.mkdir(exist_ok=True)
try:
    settings.update({"datasets_dir": str(DATASETS_DIR)})
except Exception:
    pass
DEFAULT_DATA = str(DATASETS_DIR / "smart_glasses.yaml") if (DATASETS_DIR / "smart_glasses.yaml").exists() else "coco128.yaml"
DATA_YAML=os.getenv("DATA_YAML", DEFAULT_DATA)
BASE_MODEL=os.getenv("BASE_MODEL","yolo11n.pt")
EPOCHS=int(os.getenv("EPOCHS","20")); IMAGE_SIZE=int(os.getenv("IMAGE_SIZE","640"))
BATCH_SIZE=int(os.getenv("BATCH_SIZE","8")); DEVICE=os.getenv("DEVICE","")
CONFIDENCE=float(os.getenv("CONFIDENCE","0.35"))
BEST_WEIGHTS=MODEL_DIR/"best.pt"
