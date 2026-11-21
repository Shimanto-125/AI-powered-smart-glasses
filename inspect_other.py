from pathlib import Path

label_dir = Path("datasets/smart_glasses/labels/train")

prefixes = [
    "animal-dataset-for-yolo-model_",
    "animals10_",
    "coco-minitrain-25k_",
    "solesensei_bdd100k_",
    "open-images-2019-object-detection_",
    "traffic-sign-object-detection-challenge_",
]

other = [
    f for f in label_dir.glob("*.txt")
    if not any(f.name.startswith(p) for p in prefixes)
]

print("OTHER labeled files:", len(other))

print("\nFirst 50 OTHER files:")
for f in other[:50]:
    print(f.name)

print("\nFirst 10 annotations:")
for f in other[:10]:
    print(f"\n--- {f.name} ---")
    print(f.read_text(errors="ignore")[:500])