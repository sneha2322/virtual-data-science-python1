
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model performance results
results = {
    "Model": ["SVM", "Random Forest", "CNN"],
    "Test Accuracy": [62.89, 68.95, 58.16],
    "Weighted F1-Score": [0.63, 0.69, 0.58]
}

# Create DataFrame
df = pd.DataFrame(results)

print("\nFINAL MODEL PERFORMANCE COMPARISON")
print(df.to_string(index=False))

# Identify best model
best_model = df.loc[df["Test Accuracy"].idxmax()]

print("\nBEST PERFORMING MODEL")
print("Model:", best_model["Model"])
print("Test Accuracy:", best_model["Test Accuracy"], "%")
print("Weighted F1-Score:", best_model["Weighted F1-Score"])

# Create comparison graph
comparison_data = df.melt(
    id_vars="Model",
    value_vars=["Test Accuracy", "Weighted F1-Score"],
    var_name="Metric",
    value_name="Score"
)

# Convert accuracy into decimal scale
comparison_data.loc[
    comparison_data["Metric"] == "Test Accuracy", "Score"
] /= 100

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison_data,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Final Model Performance Comparison")
plt.xlabel("Machine Learning and Deep Learning Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.legend(title="Evaluation Metric")
plt.tight_layout()

plt.savefig("final_model_comparison.png", dpi=300)
plt.show()

# Save comparison results
df.to_csv("final_model_comparison.csv", index=False)

print("\nComparison graph saved successfully.")
print("Comparison results saved successfully.")
