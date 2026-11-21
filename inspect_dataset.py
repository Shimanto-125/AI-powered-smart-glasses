
from pathlib import Path
from collections import defaultdict, Counter

label_dir = Path("datasets/smart_glasses/labels/train")

groups = {
    "Animal YOLO": "animal-dataset-for-yolo-model_",
    "Animals10": "animals10_",
    "COCO": "coco-minitrain-25k_",
    "BDD100K": "solesensei_bdd100k_",
    "Open Images": "open-images-2019-object-detection_",
    "Traffic Signs": "traffic-sign-object-detection-challenge_",
}

class_counts = defaultdict(Counter)
file_counts = Counter()

for file in label_dir.glob("*.txt"):
    group = "OTHER"

    for name, prefix in groups.items():
        if file.name.startswith(prefix):
            group = name
            break

    file_counts[group] += 1

    for line in file.read_text(errors="ignore").splitlines():
        parts = line.split()

        if len(parts) >= 5:
            try:
                class_id = int(parts[0])
                class_counts[group][class_id] += 1
            except ValueError:
                continue

print("\n========== DATASET SOURCE REPORT ==========")

for group in sorted(file_counts):
    print(f"\nSource: {group}")
    print("Label files:", file_counts[group])
    print("Class IDs:", sorted(class_counts[group]))

print("\n========== NUMERIC LABEL EXAMPLES ==========")

for filename in ["00000.txt", "00001.txt", "00003.txt"]:
    file = label_dir / filename

    if file.exists():
        print(f"\n{filename}")
        print(file.read_text(errors="ignore")[:300])
