
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("dataset_split.csv")

print("Total Images:", len(df))
print("Dataset Columns:", df.columns.tolist())

# 2. Extract image features
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
            [image], [channel], None,
            [16], [0, 256]
        )

        features.extend(hist.flatten())

    # Grayscale features
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (16, 16))

    features.extend(gray.flatten())

    return features


# 3. Prepare features and labels
X = []
y = []
splits = []

for _, row in df.iterrows():

    image_path = str(row["Image_Path"])

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

# 4. Split training and testing data
X_train = X[splits == "Training"]
y_train = y[splits == "Training"]

X_test = X[splits == "Testing"]
y_test = y[splits == "Testing"]

print("Training Images:", len(X_train))
print("Testing Images:", len(X_test))

# 5. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# 7. Calculate feature importance
importance = model.feature_importances_

feature_names = []

# RGB histogram feature names
for channel in ["Red", "Green", "Blue"]:
    for i in range(16):
        feature_names.append(
            f"{channel}_Histogram_{i}"
        )

# Grayscale feature names
for i in range(256):
    feature_names.append(
        f"Gray_Pixel_{i}"
    )

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))

# 8. Group importance by feature type
importance_groups = {
    "Red Histogram": importance[:16].sum(),
    "Green Histogram": importance[16:32].sum(),
    "Blue Histogram": importance[32:48].sum(),
    "Grayscale Features": importance[48:].sum()
}

print("\nFEATURE GROUP IMPORTANCE")

for name, value in importance_groups.items():
    print(f"{name}: {value:.4f}")

# 9. Plot top 15 features
top_features = importance_df.head(15)

plt.figure(figsize=(11, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features in Random Forest")
plt.xlabel("Feature Importance Score")
plt.ylabel("Image Features")
plt.tight_layout()

plt.savefig(
    "top_feature_importance.png",
    dpi=300
)

plt.show()

# 10. Plot feature group importance
plt.figure(figsize=(9, 6))

sns.barplot(
    x=list(importance_groups.keys()),
    y=list(importance_groups.values())
)

plt.title("Importance of Different Image Feature Groups")
plt.xlabel("Feature Groups")
plt.ylabel("Total Importance")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "feature_group_importance.png",
    dpi=300
)

plt.show()

# 11. Save results
importance_df.to_csv(
    "feature_importance_results.csv",
    index=False
)

print("\nFeature importance analysis completed.")
print("All results saved successfully.")
