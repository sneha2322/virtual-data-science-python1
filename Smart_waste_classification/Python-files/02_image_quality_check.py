
from pathlib import Path
from PIL import Image
from collections import Counter

dataset_path = Path("Smart_waste_classification/dataset-resized/dataset-resized")

image_extensions = {".jpg", ".jpeg", ".png"}

total_images = 0
corrupted_images = []
image_sizes = Counter()

for category in dataset_path.iterdir():
    if category.is_dir():

        for file in category.iterdir():

            if file.suffix.lower() in image_extensions:
                total_images += 1

                try:
                    with Image.open(file) as img:
                        img.verify()

                    with Image.open(file) as img:
                        image_sizes[img.size] += 1

                except Exception:
                    corrupted_images.append(str(file))

print("IMAGE QUALITY REPORT")
print("--------------------")
print("Total Images:", total_images)
print("Corrupted Images:", len(corrupted_images))

print("\nMost Common Image Dimensions:")
for size, count in image_sizes.most_common(10):
    print(size, ":", count)

if corrupted_images:
    print("\nCorrupted Image Files:")
    for file in corrupted_images:
        print(file)
