
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Basic settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 6

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

print("Loading dataset...")

# 2. Read dataset split
df = pd.read_csv("dataset_split.csv")

train_df = df[df["Set"] == "Training"].copy()
val_df = df[df["Set"] == "Validation"].copy()
test_df = df[df["Set"] == "Testing"].copy()

print("Training Images:", len(train_df))
print("Validation Images:", len(val_df))
print("Testing Images:", len(test_df))


# 3. Create TensorFlow datasets
def load_image(path, label):

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


def create_dataset(dataframe, shuffle=False):

    paths = dataframe["Image_Path"].astype(str).values
    labels = dataframe["Label"].astype("int32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df, shuffle=True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)

# 4. Calculate class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["Label"]),
    y=train_df["Label"]
)

class_weight_dict = {
    int(label): float(weight)
    for label, weight in zip(
        np.unique(train_df["Label"]),
        class_weights
    )
}

print("\nClass Weights:", class_weight_dict)


# 5. Build CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    # Data augmentation
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),

    # First convolution block
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu",
        padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu",
        padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    # Classification layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# 6. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN MODEL SUMMARY")
model.summary()


# 7. Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# 8. Train CNN
print("\nStarting CNN Training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)


# 9. Evaluate model
print("\nEvaluating CNN Model...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nFINAL CNN RESULTS")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")


# 10. Predict test images
y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])

y_pred_prob = model.predict(test_ds)

y_pred = np.argmax(y_pred_prob, axis=1)


# 11. Classification report
print("\nCNN CLASSIFICATION REPORT")

print(
    classification_report(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
        target_names=categories,
        zero_division=0
    )
)


# 12. Training and validation accuracy graph
plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cnn_accuracy.png", dpi=300)
plt.show()


# 13. Training and validation loss graph
plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cnn_loss.png", dpi=300)
plt.show()


# 14. Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(NUM_CLASSES)
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categories,
    yticklabels=categories
)

plt.title("CNN Waste Classification Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png", dpi=300)
plt.show()


# 15. Save trained model
model.save("waste_classification_cnn.keras")

print("\nCNN model saved successfully!")
print("CNN training and evaluation completed!")
