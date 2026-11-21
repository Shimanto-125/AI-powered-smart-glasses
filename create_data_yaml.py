
from pathlib import Path
from collections import Counter
import re

# Your dataset directory
DATASET_DIR = Path(
    r"E:\Academic project\AI_Smart_Glasses_Full_Starter_Project\AI_Smart_Glasses_Full_Starter_Project\datasets"
)

# Find all YOLO label files
label_files = list(DATASET_DIR.rglob("*.txt"))

class_counts = Counter()

for file in label_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    except (ValueError, UnicodeDecodeError):
        continue

if not class_counts:
    raise RuntimeError(
        "No valid YOLO labels found. Check your dataset folder."
    )

# Try to find class names from metadata files
class_names = {}

metadata_files = [
    "classes.txt",
    "obj.names",
    "names.txt",
    "class_names.txt"
]

for filename in metadata_files:
    for file in DATASET_DIR.rglob(filename):
        lines = file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

        if len(lines) > max(class_counts):
            class_names = {
                i: line.strip()
                for i, line in enumerate(lines)
                if line.strip()
            }
            break

    if class_names:
        break

# If names are unavailable, use honest placeholders
for class_id in range(max(class_counts) + 1):
    if class_id not in class_names:
        class_names[class_id] = f"class_{class_id}"

# Generate data.yaml
yaml_content = [
    f"path: {DATASET_DIR.as_posix()}",
    "",
    "train: images/train",
    "val: images/val",
    "",
    "names:"
]

for class_id in sorted(class_names):
    name = class_names[class_id]
    yaml_content.append(f"  {class_id}: {name}")

yaml_path = DATASET_DIR / "data.yaml"
yaml_path.write_text(
    "\n".join(yaml_content) + "\n",
    encoding="utf-8"
)

# Generate class report
report_path = DATASET_DIR / "class_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("YOLO DATASET CLASS REPORT\n")
    f.write("=" * 35 + "\n")

    for class_id, count in sorted(class_counts.items()):
        f.write(
            f"Class ID: {class_id} | "
            f"Name: {class_names[class_id]} | "
            f"Annotations: {count}\n"
        )

print("\nDataset scan completed!")
print("Total label files:", len(label_files))
print("\nDetected classes:")

for class_id, count in sorted(class_counts.items()):
    print(
        f"ID {class_id}: "
        f"{class_names[class_id]} "
        f"({count} annotations)"
    )

print("\nCreated:", yaml_path)
print("Created:", report_path)
