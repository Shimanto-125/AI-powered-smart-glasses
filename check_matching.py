
from pathlib import Path

image_dir = Path("datasets/smart_glasses/images/train")
label_dir = Path("datasets/smart_glasses/labels/train")

images = {
    p.stem for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".webp"
    }
}

labels = {p.stem for p in label_dir.glob("*.txt")}

matched = images & labels
images_without_labels = images - labels
labels_without_images = labels - images

print("========== IMAGE-LABEL MATCH REPORT ==========")
print("Total images:", len(images))
print("Total labels:", len(labels))
print("Matched pairs:", len(matched))
print("Images without labels:", len(images_without_labels))
print("Labels without images:", len(labels_without_images))

print("\nUnmatched image examples:")
for name in sorted(images_without_labels)[:10]:
    print(name)

print("\nUnmatched label examples:")
for name in sorted(labels_without_images)[:10]:
    print(name)
