
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load training image paths
split_df = pd.read_csv("dataset_split.csv")
train_df = split_df[split_df["Set"] == "Training"].copy()

categories = sorted(train_df["Label"].unique())
category_names = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

# 2. Extract visual features
def extract_features(image_path):

    with Image.open(image_path) as img:
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

    # Grayscale image features
    gray = np.array(
        Image.fromarray(image).convert("L").resize((16, 16))
    )

    features.extend(gray.flatten() / 255.0)

    return features


print("Extracting image features...")

X = np.array([
    extract_features(path)
    for path in train_df["Image_Path"]
])

y = train_df["Label"].to_numpy()

print("Feature Matrix Shape:", X.shape)

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("PCA Shape:", X_pca.shape)
print(
    "Explained Variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

# 5. Compare different K values
k_values = [2, 3, 4, 5, 6, 7, 8]
silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(X_pca)

    score = silhouette_score(
        X_pca,
        cluster_labels
    )

    silhouette_scores.append(score)

    print(f"K={k}, Silhouette Score={score:.4f}")

# 6. Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_silhouette_scores.png", dpi=300)
plt.show()

# 7. Select best K from tested values
best_k = k_values[np.argmax(silhouette_scores)]

print("\nSelected K:", best_k)

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_clusters = final_model.fit_predict(X_pca)

# 8. Cluster composition
results = pd.DataFrame({
    "Actual_Category": [
        category_names[label] for label in y
    ],
    "Cluster": final_clusters
})

composition = pd.crosstab(
    results["Cluster"],
    results["Actual_Category"]
)

print("\nCLUSTER COMPOSITION")
print(composition)

composition.to_csv("kmeans_cluster_composition.csv")

# 9. Visualize clusters using PCA


# 9. Visualize cluster composition using heatmap

import seaborn as sns

plt.figure(figsize=(10, 6))

sns.heatmap(
    composition,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Waste Category Distribution Across K-Means Clusters")
plt.xlabel("Actual Waste Category")
plt.ylabel("K-Means Cluster")

plt.tight_layout()
plt.savefig("kmeans_cluster_heatmap.png", dpi=300)
plt.show()

print("\nK-Means cluster heatmap generated successfully!")



print("\nK-Means clustering completed!")
