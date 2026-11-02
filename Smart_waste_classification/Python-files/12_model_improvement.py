
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load dataset
df = pd.read_csv("dataset_split.csv")

print("Total Images:", len(df))

# Feature extraction function
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (128, 128))
    features = []

    # RGB colour histograms
    for channel in range(3):
        hist = cv2.calcHist(
            [image], [channel], None, [16], [0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Grayscale pixel features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    features.extend(gray.flatten() / 255.0)

    return features


# Extract features
X = []
y = []
sets = []

for _, row in df.iterrows():
    image_path = row["Image_Path"]

    if not os.path.exists(image_path):
        image_path = os.path.join("..", image_path)

    X.append(extract_features(image_path))
    y.append(row["Label"])
    sets.append(row["Set"])

X = np.array(X)
y = np.array(y)
sets = np.array(sets)

# Split using existing dataset split
X_train = X[sets == "Training"]
y_train = y[sets == "Training"]

X_val = X[sets == "Validation"]
y_val = y[sets == "Validation"]

X_test = X[sets == "Testing"]
y_test = y[sets == "Testing"]

print("\nTraining:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# Three model configurations
models = {
    "Baseline RF": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Improved RF 1": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Improved RF 2": RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

# Train and compare on validation data
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    weighted_f1 = f1_score(
        y_val, predictions, average="weighted"
    )

    results.append({
        "Model": name,
        "Validation Accuracy": accuracy,
        "Weighted F1": weighted_f1
    })

    print("Validation Accuracy:", round(accuracy * 100, 2), "%")
    print("Weighted F1:", round(weighted_f1, 3))

# Select best model using validation accuracy
best_result = max(results, key=lambda x: x["Validation Accuracy"])
best_name = best_result["Model"]
best_model = models[best_name]

# Final evaluation on testing data
test_predictions = best_model.predict(X_test)

print("\nBEST MODEL:", best_name)
print("\nFINAL TEST ACCURACY:",
      round(accuracy_score(y_test, test_predictions) * 100, 2), "%")

print("\nFINAL CLASSIFICATION REPORT:")
print(classification_report(y_test, test_predictions))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_improvement_results.csv", index=False)

print("\nAll results saved successfully.")
