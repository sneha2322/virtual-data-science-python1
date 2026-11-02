
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

dataset_path = Path("Smart_waste_classification/dataset-resized/dataset-resized")

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for index, category in enumerate(categories):
    folder = dataset_path / category

    image_files = [
        file for file in folder.iterdir()
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    image_path = random.choice(image_files)
    img = Image.open(image_path).convert("RGB")

    ax = axes[index // 3, index % 3]
    ax.imshow(img)
    ax.set_title(category.capitalize(), fontsize=12)
    ax.axis("off")

fig.suptitle(
    "Sample Images from Each Waste Category",
    fontsize=16,
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("sample_waste_images.png", dpi=300)
plt.show()

