Task 4: Supervised Learning Model

Project Overview

This project focuses on building a supervised machine learning model using Python. The model classifies breast tumour records into two categories: Benign (B) and Malignant (M), using the Wisconsin Breast Cancer Diagnostic dataset.

Objectives

- Understand supervised machine learning.
- Prepare and process the dataset.
- Split the dataset into training and testing sets.
- Train a Logistic Regression classification model.
- Evaluate model performance using different metrics.
- Visualize the results using a confusion matrix.

Dataset

The project uses the Wisconsin Breast Cancer Diagnostic dataset from the UCI Machine Learning Repository.

- Total records: 569
- Total original columns: 32
- Input features used: 30
- Target column: Diagnosis
- Benign cases: 357
- Malignant cases: 212
- Missing values: 0

Dataset link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Visual Studio Code

Methodology

1. Loaded the dataset using Pandas.
2. Checked the dataset structure and missing values.
3. Removed the ID column and separated features from the target.
4. Split the data into 80% training data and 20% testing data.
5. Applied StandardScaler to scale the features.
6. Trained a Logistic Regression model.
7. Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix.

Model Performance

The model achieved the following results on the test dataset:

- Accuracy: 96.49%
- Precision for Malignant class: 97.50%
- Recall for Malignant class: 92.86%
- F1-score for Malignant class: 95.12%
- Test records: 114

The model correctly classified 110 out of 114 test records.

Project Files

- "task_4.py" — Python source code.
- "wdbc.data" — Dataset file.
- "wdbc.names" — Dataset information.
- "confusion_matrix.png" — Confusion matrix visualization.
- "Task_4_Report.docx" — Detailed project report.

Conclusion

This project provided practical experience in supervised learning, data preprocessing, model training, and model evaluation. Logistic Regression achieved 96.49% accuracy on the test dataset. The project helped demonstrate how machine learning can be used for classification tasks.

Note: This project is for educational purposes only and must not be used for real-world medical diagnosis.

References

1. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. Pandas Documentation: https://pandas.pydata.org/docs/
4. Matplotlib Documentation: https://matplotlib.org/stable/
