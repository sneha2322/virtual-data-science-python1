
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset split
SPLIT_FILE = "dataset_split.csv"

df = pd.read_csv(SPLIT_FILE)

print("CSV Columns:", df.columns.tolist())
print("Total Images:", len(df))

# 2. Waste categories
categories = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# 3. Image feature extraction
def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = []

    # RGB histogram features
    for channel in range(3):

        hist = cv2.calcHist(
            [image],
            [channel],
            None,
            [16],
            [0, 256]
        )

        features.extend(hist.flatten())

    # Grayscale features
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(gray, (16, 16))

    features.extend(gray.flatten())

    return features


# 4. Extract all image features
X = []
y = []
splits = []

for _, row in df.iterrows():

    image_path = str(row["Image_Path"])

    # Handle paths stored relative to the parent folder
    if not os.path.exists(image_path):
        image_path = os.path.normpath(
            os.path.join("..", image_path)
        )

    features = extract_features(image_path)

    if features is not None:

        X.append(features)
        y.append(int(row["Label"]))
        splits.append(row["Set"])

X = np.array(X)
y = np.array(y)
splits = np.array(splits)

print("\nFeature Matrix Shape:", X.shape)

# 5. Separate training and testing data
X_train = X[splits == "Training"]
y_train = y[splits == "Training"]

X_test = X[splits == "Testing"]
y_test = y[splits == "Testing"]

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))

# 6. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Classification report
print("\nRANDOM FOREST ERROR ANALYSIS")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(6),
        target_names=categories,
        zero_division=0
    )
)

# 10. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(6)
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

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()

plt.savefig(
    "random_forest_error_analysis.png",
    dpi=300
)

plt.show()

# 11. Save incorrect predictions
errors = pd.DataFrame({
    "Actual Category": [
        categories[i] for i in y_test
    ],
    "Predicted Category": [
        categories[i] for i in y_pred
    ]
})

errors = errors[
    errors["Actual Category"] != errors["Predicted Category"]
]

errors.to_csv(
    "random_forest_misclassifications.csv",
    index=False
)

# 12. Final results
print("\nFINAL ERROR ANALYSIS")

print("Total Test Images:", len(y_test))
print("Correct Predictions:", np.sum(y_test == y_pred))
print("Incorrect Predictions:", len(errors))

print("\nError analysis graph saved successfully.")
print("Misclassification CSV saved successfully.")
