import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_student_data.csv",sep=None,engine="python")

# Display the first five rows
print("First 5 Rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types and non-null counts
print("\nDataset Information:")
df.info()

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 1. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Check missing values percentage
print("\nMissing Values Percentage:")
print((df.isnull().mean() * 100).round(2))

# 3. Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 4. Check data types
print("\nData Types:")
print(df.dtypes)

# 5. Check numerical summary
print("\nStatistical Summary:")
print(df.describe())

# 6. Check categorical columns
print("\nCategorical Summary:")
print(df.describe(include=["object","str"]))

# Graph 1: Distribution of Final Grades

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="G3",
    bins=11,
    kde=True
)

plt.title("Distribution of Students' Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("final_grade_distribution.png", dpi=300)
plt.show()

# Graph 2: Study Time vs Final Grades

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="studytime",
    y="G3"
)

plt.title("Study Time vs Final Grades")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()
plt.savefig("studytime_vs_grades.png", dpi=300)
plt.show()

# Graph 3: Absences vs Average Final Grades

df["absence_group"] = pd.cut(
    df["absences"],
    bins=[-1, 0, 5, 10, float("inf")],
    labels=["0 absences", "1–5 absences",
            "6–10 absences", "More than 10"]
)

absence_grades = df.groupby(
    "absence_group", observed=False
)["G3"].mean()

plt.figure(figsize=(9, 5))

absence_grades.plot(kind="bar", color="skyblue")

plt.title("Average Final Grades by Absence Group")
plt.xlabel("Number of Absences")
plt.ylabel("Average Final Grade (G3)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("absences_vs_grades.png", dpi=300)
plt.show()

# Graph 4: Average Final Grade by Gender

gender_grades = df.groupby("sex")["G3"].mean()

plt.figure(figsize=(7, 5))

gender_grades.plot(kind="bar")

plt.title("Average Final Grade by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Grade")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("gender_vs_grades.png", dpi=300)
plt.show()

# Graph 5: Average Grades Comparison

average_grades = df[["G1", "G2", "G3"]].mean()

plt.figure(figsize=(8, 5))
average_grades.plot(kind="bar", color="skyblue")

plt.title("Comparison of Average Student Grades")
plt.xlabel("Grade Period")
plt.ylabel("Average Grade")
plt.xticks(
    [0, 1, 2],
    ["G1 (First Period)", "G2 (Second Period)", "G3 (Final)"],
    rotation=0
)

plt.tight_layout()
plt.savefig("average_grades_comparison.png", dpi=300)
plt.show()

# Step 6: Numerical Analysis and Findings

print("\n--- Average Final Grade by Study Time ---")
print(
    df.groupby("studytime")["G3"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\n--- Average Final Grade by Gender ---")
print(
    df.groupby("sex")["G3"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\n--- Average Final Grade by Absence Group ---")
print(
    df.groupby("absence_group", observed=False)["G3"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\n--- Correlation Between Grades ---")
grade_columns = ["G1", "G2", "G3"]
print(df[grade_columns].corr().round(2))

print("\n--- EDA Completed Successfully ---")


