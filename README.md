# AI Smart Glasses starter
Prototype: YOLO object detection, rough direction/distance, experimental traffic-light color heuristic, OCR, speech, weather lookup, Streamlit UI.

## Run
Python 3.10/3.11/3.13:
```powershell
pip install -r requirements.txt
python train.py        # Train model with datasets/smart_glasses.yaml (Auto-deploys to models/best.pt)
streamlit run app.py   # Run web app with Live Webcam + Upload + OCR modes
python live_detect.py  # High-FPS direct live camera detection with voice guidance
```

### Dataset Configuration
The project is configured to train using `datasets/smart_glasses.yaml`:
- **Training Images:** `datasets/smart_glasses/images/train`
- **Validation Images:** `datasets/smart_glasses/images/val`
- **Labels:** `datasets/smart_glasses/labels/` (COCO 80 classes)
- **Deployment:** When training completes, `train.py` automatically deploys the best checkpoint to `models/best.pt`, which is instantly loaded by `app.py` and `live_detect.py`.

Dataset layout:
```
dataset/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml
```
Labels: `class_id x_center y_center width height`, normalized 0..1. Verify class mapping and image-label pairs. Do not merge unrelated datasets without harmonizing labels.

## Critical limitations
This is not a certified mobility aid. Never rely on it for road crossing. Single-camera distance is approximate and requires camera calibration. COCO detects traffic-light objects, not red/yellow/green state; included color heuristic is experimental and not validated. A web-hosted Space cannot access/control the user's cellular modem or AirPods; audio follows browser/OS output routing. Real cellular connectivity needs a physical LTE/4G modem or cellular phone and device integration. Train on Colab/local GPU; free hosting is for demo inference and may be slow/sleep.
