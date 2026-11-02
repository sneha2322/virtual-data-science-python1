
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset split
df = pd.read_csv("dataset_split.csv")

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

# 2. Extract image features
def extract_features(image_path):

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((32, 32))
        image = np.array(img)

    features = []

    # RGB colour histograms
    for channel in range(3):
        hist, _ = np.histogram(
            image[:, :, channel],
            bins=16,
            range=(0, 256),
            density=True
        )
        features.extend(hist)

    # Grayscale features
    gray = np.array(
        Image.fromarray(image).convert("L").resize((16, 16))
    )

    features.extend(gray.flatten() / 255.0)

    return features


print("Extracting image features...")

X = np.array([
    extract_features(path)
    for path in df["Image_Path"]
])

y = df["Label"].astype(int).to_numpy()

print("Feature Matrix Shape:", X.shape)

# 3. Split features using existing dataset split
train_mask = df["Set"].str.lower() == "training"
val_mask = df["Set"].str.lower() == "validation"
test_mask = df["Set"].str.lower() == "testing"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nDATASET SPLIT")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# 4. Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Train SVM model
print("\nTraining SVM Model...")

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_model.fit(X_train_scaled, y_train)

svm_val_pred = svm_model.predict(X_val_scaled)

print(
    "SVM Validation Accuracy:",
    round(accuracy_score(y_val, svm_val_pred) * 100, 2),
    "%"
)

# 6. Train Random Forest model
print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_val_pred = rf_model.predict(X_val)

print(
    "Random Forest Validation Accuracy:",
    round(accuracy_score(y_val, rf_val_pred) * 100, 2),
    "%"
)

# 7. Final test evaluation
svm_test_pred = svm_model.predict(X_test_scaled)
rf_test_pred = rf_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_test_pred)
rf_accuracy = accuracy_score(y_test, rf_test_pred)

print("\nFINAL TEST RESULTS")

print("\nSVM Accuracy:", round(svm_accuracy * 100, 2), "%")
print("\nSVM Classification Report:")
print(
    classification_report(
        y_test,
        svm_test_pred,
        labels=range(6),
        target_names=categories,
        zero_division=0
    )
)

print("\nRandom Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_test_pred,
        labels=range(6),
        target_names=categories,
        zero_division=0
    )
)

# 8. Compare model accuracy
comparison = pd.DataFrame({
    "Model": ["SVM", "Random Forest"],
    "Test Accuracy": [
        svm_accuracy * 100,
        rf_accuracy * 100
    ]
})

print("\nMODEL COMPARISON")
print(comparison.round(2))

comparison.to_csv("supervised_model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["Test Accuracy"])

plt.title("SVM vs Random Forest - Test Accuracy")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig("supervised_model_comparison.png", dpi=300)
plt.show()

# 9. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, predictions, title in [
    (axes[0], svm_test_pred, "SVM Confusion Matrix"),
    (axes[1], rf_test_pred, "Random Forest Confusion Matrix")
]:

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=range(6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")

plt.tight_layout()
plt.savefig("supervised_confusion_matrices.png", dpi=300)
plt.show()

print("\nSupervised Learning Completed Successfully!")
