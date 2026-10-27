
# Task 4: Breast Cancer Diagnostic Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
column_names = [
    "ID", "Diagnosis",
    "Radius_Mean", "Texture_Mean", "Perimeter_Mean",
    "Area_Mean", "Smoothness_Mean", "Compactness_Mean",
    "Concavity_Mean", "Concave_Points_Mean",
    "Symmetry_Mean", "Fractal_Dimension_Mean",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE",
    "Smoothness_SE", "Compactness_SE", "Concavity_SE",
    "Concave_Points_SE", "Symmetry_SE", "Fractal_Dimension_SE",
    "Radius_Worst", "Texture_Worst", "Perimeter_Worst",
    "Area_Worst", "Smoothness_Worst", "Compactness_Worst",
    "Concavity_Worst", "Concave_Points_Worst",
    "Symmetry_Worst", "Fractal_Dimension_Worst"
]

df = pd.read_csv(
    "supervised_learning_model/wdbc.data",
    header=None,
    names=column_names
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Remove ID column
df = df.drop(columns=["ID"])

# Separate features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
print("\nDiagnosis counts:")
print(y.value_counts())

# 4. Check missing values
print("\nMissing values:", df.isnull().sum().sum())

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing completed!")

# 7. Train the classification model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

print("Model training completed!")

# 8. Make predictions
y_pred = model.predict(X_test_scaled)

# 9. Evaluate the model
print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(
    precision_score(y_test, y_pred, pos_label="M"), 4
))
print("Recall:", round(
    recall_score(y_test, y_pred, pos_label="M"), 4
))
print("F1-score:", round(
    f1_score(y_test, y_pred, pos_label="M"), 4
))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["B", "M"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign (B)", "Malignant (M)"]
)
disp.plot(cmap="Blues")
plt.title("Breast Cancer Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("supervised_learning_model/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Compare actual and predicted diagnoses
results = pd.DataFrame({
    "Actual Diagnosis": y_test.values,
    "Predicted Diagnosis": y_pred
})

print("\nActual vs Predicted (first 15 records):")
print(results.head(15).to_string(index=False))
