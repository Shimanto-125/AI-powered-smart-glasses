
from pathlib import Path
from collections import Counter

image_dir = Path("datasets/smart_glasses/images/train")
label_dir = Path("datasets/smart_glasses/labels/train")

prefixes = {
    "Animal YOLO": "animal-dataset-for-yolo-model_",
    "Animals10": "animals10_",
    "COCO": "coco-minitrain-25k_",
    "BDD100K": "solesensei_bdd100k_",
    "Open Images": "open-images-2019-object-detection_",
    "Traffic Signs": "traffic-sign-object-detection-challenge_",
}

counts = Counter()

for image in image_dir.iterdir():
    if not image.is_file():
        continue

    if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    if (label_dir / f"{image.stem}.txt").exists():
        continue

    source = "OTHER"

    for name, prefix in prefixes.items():
        if image.name.startswith(prefix):
            source = name
            break

    counts[source] += 1

print("UNLABELED IMAGES BY SOURCE")
for source, count in counts.most_common():
    print(f"{source}: {count}")
