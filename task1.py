import pandas as pd

df = pd.read_csv("dataset/student-mat.csv", sep=";")

print("First 5 rows:")
print(df.head())

print("\nDataset rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nUnique values in key columns:")
for col in ["age", "studytime", "absences", "G1", "G2", "G3"]:
    print(f"\n{col}:")
    print(sorted(df[col].unique()))

print("\nCheck valid ranges:")
print("Age range:", df["age"].min(), "-", df["age"].max())
print("Study time range:", df["studytime"].min(), "-", df["studytime"].max())
print("Absences range:", df["absences"].min(), "-", df["absences"].max())
print("Final grade range:", df["G3"].min(), "-", df["G3"].max())

# Save the preprocessed dataset
df.to_csv("dataset/cleaned_student_data.csv", sep=";", index=False)

print("\nCleaned dataset saved successfully!")

print("Rows:", len(df))
print("Empty rows:", df.isna().all(axis=1).sum())