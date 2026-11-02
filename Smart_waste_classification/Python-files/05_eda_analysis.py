
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset location
dataset_path = Path("Smart_waste_classification/dataset-resized/dataset-resized")

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

extensions = {".jpg", ".jpeg", ".png"}

results = []

# Analyse each category
for category in categories:

    folder = dataset_path / category

    brightness_values = []
    red_values = []
    green_values = []
    blue_values = []

    image_files = [
        file for file in folder.iterdir()
        if file.suffix.lower() in extensions
    ]

    for file in image_files:

        with Image.open(file) as img:
            img = img.convert("RGB")
            img = img.resize((128, 128))

            image_array = np.array(img, dtype=np.float32)

            # Average brightness
            grayscale = np.mean(image_array, axis=2)
            brightness_values.append(np.mean(grayscale))

            # Average RGB values
            red_values.append(np.mean(image_array[:, :, 0]))
            green_values.append(np.mean(image_array[:, :, 1]))
            blue_values.append(np.mean(image_array[:, :, 2]))

    results.append({
        "Category": category,
        "Image Count": len(image_files),
        "Average Brightness": np.mean(brightness_values),
        "Red": np.mean(red_values),
        "Green": np.mean(green_values),
        "Blue": np.mean(blue_values)
    })

# Create DataFrame
df = pd.DataFrame(results)

# Calculate percentages
df["Image Percentage"] = (
    df["Image Count"] / df["Image Count"].sum()
) * 100

print("\nCATEGORY-WISE EDA REPORT")
print(df.round(2).to_string(index=False))

# Save results
df.to_csv("eda_results.csv", index=False)

# Graph 1: Category-wise image percentage
plt.figure(figsize=(9, 5))
plt.bar(df["Category"], df["Image Percentage"])
plt.title("Percentage Distribution of Waste Categories")
plt.xlabel("Waste Category")
plt.ylabel("Percentage of Images (%)")
plt.tight_layout()
plt.savefig("category_percentage.png", dpi=300)
plt.show()

# Graph 2: Average brightness
plt.figure(figsize=(9, 5))
plt.bar(df["Category"], df["Average Brightness"])
plt.title("Average Image Brightness by Waste Category")
plt.xlabel("Waste Category")
plt.ylabel("Average Brightness (0-255)")
plt.tight_layout()
plt.savefig("category_brightness.png", dpi=300)
plt.show()

# Graph 3: Average RGB colour distribution
df.plot(
    x="Category",
    y=["Red", "Green", "Blue"],
    kind="bar",
    figsize=(10, 5)
)

plt.title("Average RGB Colour Distribution by Category")
plt.xlabel("Waste Category")
plt.ylabel("Average Pixel Intensity (0-255)")
plt.xticks(rotation=30)
plt.legend(title="Colour Channel")
plt.tight_layout()
plt.savefig("rgb_distribution.png", dpi=300)
plt.show()

print("\nEDA analysis completed successfully!")


# Graph 4: Image count with percentage labels

plt.figure(figsize=(9, 5))

bars = plt.bar(
    df["Category"],
    df["Image Count"]
)

plt.title("Image Count and Distribution Across Waste Categories")
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")

for bar, percentage in zip(bars, df["Image Percentage"]):
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 8,
        f"{percentage:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("class_imbalance_analysis.png", dpi=300)
plt.show()

