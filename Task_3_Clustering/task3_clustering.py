
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_student_data.csv",sep=";")

# Display the first five rows
print("First five rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# Select numerical features for clustering
features = ["studytime", "failures", "absences", "G1", "G2"]

X = df[features].copy()

print("\nSelected features:")
print(X.head())

print("\nSelected data shape:")
print(X.shape)

print("\nData types:")
print(X.dtypes)

print("\nMissing values in selected features:")
print(X.isnull().sum())


# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled feature data (first 5 rows):")
print(X_scaled[:5])

print("\nScaled data shape:")
print(X_scaled.shape)


# Elbow Method to find the optimal number of clusters

inertia_values = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "elbow_method.png",
    dpi=300
   
)
plt.show()

print("\nInertia values:")
for k, inertia in zip(k_values, inertia_values):
    print(f"K = {k}: Inertia = {inertia:.2f}")


# Silhouette Score Evaluation

silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"K = {k}: Silhouette Score = {score:.4f}")

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_range),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "silhouette_scores.png",
    dpi=300
    
)
plt.show()


# Final K-Means Model
optimal_k = 5

final_kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    n_init=10,
    random_state=42
)

# Create cluster labels using selected features only
df["Cluster"] = final_kmeans.fit_predict(X_scaled)

print("\nCluster labels assigned successfully!")
print(df["Cluster"].value_counts().sort_index())

# Cluster profile and final-grade comparison
profile = df.groupby("Cluster").agg(
    Student_Count=("G3", "size"),
    Average_Study_Time=("studytime", "mean"),
    Average_Failures=("failures", "mean"),
    Average_Absences=("absences", "mean"),
    Average_G1=("G1", "mean"),
    Average_G2=("G2", "mean"),
    Average_G3=("G3", "mean")
)

print("\nCluster Profile:")
print(profile.round(2))


# Visualization 1: Clusters using G1 and G2

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="G1",
    y="G2",
    hue="Cluster",
    palette="viridis",
    s=70
)

plt.title("Student Clusters Based on G1 and G2")
plt.xlabel("First Period Grade (G1)")
plt.ylabel("Second Period Grade (G2)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("student_clusters_g1_g2.png",dpi=300)
plt.show()


# Visualization 2: Number of Students in Each Cluster

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster",
    hue="Cluster",
    palette="viridis",
    legend=False
)

plt.title("Number of Students in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("cluster_sizes.png", dpi=300)

plt.show()


# Visualization 3: Average Final Grade (G3) by Cluster

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Cluster",
    y="G3",
    hue="Cluster",
    palette="viridis",
    legend=False,
    errorbar=None
)

plt.title("Average Final Grade by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Final Grade (G3)")
plt.tight_layout()
plt.savefig("average_g3_by_cluster.png", dpi=300)
plt.show()
