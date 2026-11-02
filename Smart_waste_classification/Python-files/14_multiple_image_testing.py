
import os
import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score

# Class names
class_names = {
    0: "Cardboard",
    1: "Glass",
    2: "Metal",
    3: "Paper",
    4: "Plastic",
    5: "Trash"
}

# Load trained model
model = joblib.load("waste_classification_model.pkl")

# Feature extraction
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image")

    image = cv2.resize(image, (128, 128))
    features = []

    for channel in range(3):
        hist = cv2.calcHist(
            [image], [channel], None, [16], [0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    features.extend(gray.flatten() / 255.0)

    return np.array(features)


# Load testing dataset
df = pd.read_csv("dataset_split.csv")
test_df = df[df["Set"] == "Testing"].copy()

# Select up to 10 images from each class
sample_df = (
    test_df.groupby("Label", group_keys=False)
    .head(10)
    .reset_index(drop=True)
)

print("Total images selected:", len(sample_df))

actual_labels = []
predicted_labels = []
results = []

# Predict each image
for _, row in sample_df.iterrows():

    image_path = row["Image_Path"]

    if not os.path.exists(image_path):
        image_path = os.path.join("..", image_path)

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        continue

    features = extract_features(image_path).reshape(1, -1)

    prediction = model.predict(features)[0]

    actual = int(row["Label"])

    actual_labels.append(actual)
    predicted_labels.append(prediction)

    result = "Correct" if actual == prediction else "Incorrect"

    results.append({
        "Image": os.path.basename(image_path),
        "Actual Category": class_names[actual],
        "Predicted Category": class_names[prediction],
        "Result": result
    })

# Display results
results_df = pd.DataFrame(results)

print("\nMULTIPLE IMAGE PREDICTION RESULTS")
print(results_df.to_string(index=False))

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)

correct = sum(
    a == p for a, p in zip(actual_labels, predicted_labels)
)

incorrect = len(actual_labels) - correct

print("\n========== FINAL SUMMARY ==========")
print("Images Tested:", len(actual_labels))
print("Correct Predictions:", correct)
print("Incorrect Predictions:", incorrect)
print("Testing Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    actual_labels,
    predicted_labels,
    labels=list(class_names.keys()),
    target_names=list(class_names.values()),
    zero_division=0
))

# Save results
results_df.to_csv("multiple_prediction_results.csv", index=False)

print("\nResults saved to multiple_prediction_results.csv")
