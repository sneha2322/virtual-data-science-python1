
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Dataset folder
dataset_path = Path("Smart_waste_classification/dataset-resized/dataset-resized")

# Check dataset folder
if not dataset_path.exists():
    print("Dataset folder not found! Check the folder location.")
    raise SystemExit

# Count images in each category
image_extensions = {".jpg", ".jpeg", ".png"}
class_counts = {}

for category in sorted(dataset_path.iterdir()):
    if category.is_dir():
        images = [
            file for file in category.iterdir()
            if file.suffix.lower() in image_extensions
        ]
        class_counts[category.name] = len(images)

# Display dataset information
df = pd.DataFrame(
    list(class_counts.items()),
    columns=["Category", "Number of Images"]
)

print("DATASET SUMMARY")
print(df)
print("\nTotal Images:", df["Number of Images"].sum())
print("Total Categories:", len(df))

# Visualize class distribution
plt.figure(figsize=(9, 5))
plt.bar(df["Category"], df["Number of Images"])
plt.title("Number of Images in Each Waste Category")
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
