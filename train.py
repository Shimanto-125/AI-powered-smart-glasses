import argparse
from pathlib import Path
from ultralytics import YOLO
from config import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Model for AI Smart Glasses")
    parser.add_argument("--data", type=str, default=DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default=BASE_MODEL, help="Base YOLO model weights (e.g. yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Image size (e.g. 640 or 416)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run on (e.g. '0', 'cpu', or '')")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 is best on Windows CPU)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g. 0.1 for quick test, 1.0 for full)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print(" 🚀 AI SMART GLASSES - MODEL TRAINING")
    print("=" * 60)
    print(f"[*] Dataset config : {args.data}")
    print(f"[*] Base model     : {args.model}")
    print(f"[*] Epochs         : {args.epochs}")
    print(f"[*] Batch size     : {args.batch}")
    print(f"[*] Image size     : {args.imgsz}")
    print(f"[*] Device         : {args.device or 'CPU (auto)'}")
    print(f"[*] Data fraction  : {args.fraction}")
    print(f"[*] Target deploy  : {BEST_WEIGHTS}")
    print("=" * 60)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(RUNS_DIR),
        name="smart_glasses",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        patience=8,
        fraction=args.fraction,
        # Keep colour augmentation moderate: traffic signal colour matters.
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=3,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        plots=True,
        workers=args.workers,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (RUNS_DIR / "smart_glasses")
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        BEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        BEST_WEIGHTS.write_bytes(best.read_bytes())
        print("\n" + "=" * 60)
        print(" ✅ TRAINING & DEPLOYMENT COMPLETE!")
        print(f"[*] Trained model saved to: {best}")
        print(f"[*] Deployed to app path  : {BEST_WEIGHTS}")
        print("=" * 60)
        print("Now you can launch the app:")
        print("  streamlit run app.py")
        print("  python live_detect.py")
        print("=" * 60)
    else:
        print(f"[!] Warning: best.pt not found at {best}; inspect {save_dir}")

if __name__ == "__main__":
    main()
