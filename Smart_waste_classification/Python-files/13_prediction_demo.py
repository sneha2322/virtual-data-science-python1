
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Class names
class_names = {
    0: "Cardboard",
    1: "Glass",
    2: "Metal",
    3: "Paper",
    4: "Plastic",
    5: "Trash"
}

# Feature extraction function
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded!")

    image = cv2.resize(image, (128, 128))
    features = []

    # RGB colour histogram features
    for channel in range(3):
        hist = cv2.calcHist(
            [image], [channel], None, [16], [0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # Grayscale features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    features.extend(gray.flatten() / 255.0)

    return np.array(features).reshape(1, -1)


# Load dataset
df = pd.read_csv("dataset_split.csv")

X_train = []
y_train = []

print("Preparing training data...")

for _, row in df[df["Set"] == "Training"].iterrows():

    image_path = row["Image_Path"]

    if not os.path.exists(image_path):
        image_path = os.path.join("..", image_path)

    X_train.append(extract_features(image_path).flatten())
    y_train.append(row["Label"])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train Random Forest model
print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=2,
    min_samples_split=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save trained model
import joblib
joblib.dump(model, "waste_classification_model.pkl")

print("\nModel trained and saved successfully!")

# Enter image path
image_path = input(
    "\nEnter the path of your waste image: "
).strip().strip('"')

if not os.path.exists(image_path):
    print("Image file not found. Please check the path.")

else:
    features = extract_features(image_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print("\n========== PREDICTION RESULT ==========")
    print("Predicted Waste Category:",
          class_names[prediction])

    print("\nClass-wise Prediction Confidence:")

    for class_id, probability in zip(
        model.classes_, probabilities
    ):
        print(
            f"{class_names[class_id]}: "
            f"{probability * 100:.2f}%"
        )

    print("======================================")
