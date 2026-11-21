
from pathlib import Path
from PIL import Image

image_dir = Path("datasets/smart_glasses/images/train")
label_dir = Path("datasets/smart_glasses/labels/train")

extensions = {".jpg", ".jpeg", ".png", ".webp"}

print("========== NUMERIC LABEL IMAGE MATCHES ==========")

for label in sorted(label_dir.glob("*.txt")):
    if not label.stem.isdigit():
        continue

    matches = [
        image for image in image_dir.glob(label.stem + ".*")
        if image.suffix.lower() in extensions
    ]

    for image in matches:
        try:
            with Image.open(image) as im:
                print(
                    f"{label.name} -> {image.name} | "
                    f"Size: {im.size} | Mode: {im.mode}"
                )
        except Exception as e:
            print(image.name, "Image error:", e)
