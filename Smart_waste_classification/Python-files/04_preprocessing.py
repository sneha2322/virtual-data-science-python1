
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Dataset path
dataset_path = Path("Smart_waste_classification/dataset-resized/dataset-resized")

# 2. Image categories
categories = sorted([
    folder.name
    for folder in dataset_path.iterdir()
    if folder.is_dir()
])

image_extensions = {".jpg", ".jpeg", ".png"}

# 3. Collect image paths and labels
image_paths = []
image_labels = []

for label, category in enumerate(categories):
    category_path = dataset_path / category

    for image in category_path.iterdir():
        if image.suffix.lower() in image_extensions:
            image_paths.append(str(image))
            image_labels.append(label)

image_paths = np.array(image_paths)
image_labels = np.array(image_labels)

print("Total Images:", len(image_paths))
print("Categories:", categories)

# 4. Split into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    image_labels,
    test_size=0.30,
    random_state=42,
    stratify=image_labels
)

# 5. Split temporary set into validation and testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDATASET SPLIT")
print("Training Images:", len(X_train))
print("Validation Images:", len(X_val))
print("Testing Images:", len(X_test))

# 6. Image preprocessing settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

def preprocess_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label

# 7. Create TensorFlow datasets
def create_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=42
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = create_dataset(X_train, y_train, training=True)
val_ds = create_dataset(X_val, y_val)
test_ds = create_dataset(X_test, y_test)

# 8. Display one batch information
for images, labels in train_ds.take(1):
    print("\nIMAGE PREPROCESSING DETAILS")
    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)
    print("Minimum Pixel Value:", float(tf.reduce_min(images)))
    print("Maximum Pixel Value:", float(tf.reduce_max(images)))

# 9. Save split information
split_data = pd.DataFrame({
    "Image_Path": list(X_train) + list(X_val) + list(X_test),
    "Label": list(y_train) + list(y_val) + list(y_test),
    "Set": (
        ["Training"] * len(X_train)
        + ["Validation"] * len(X_val)
        + ["Testing"] * len(X_test)
    )
})

split_data.to_csv("dataset_split.csv", index=False)

print("\nDataset split information saved successfully!")
