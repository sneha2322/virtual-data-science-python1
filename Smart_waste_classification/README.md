 Smart Waste Classification and Recycling Analysis

 Project Overview
This project focuses on waste classification using Machine Learning and Deep Learning techniques. It uses the TrashNet dataset to classify waste images into six categories: cardboard, glass, metal, paper, plastic, and trash.

The main objective is to explore waste data, analyze image features, apply different classification algorithms, and evaluate their performance.

Dataset
The project uses the TrashNet dataset containing 2,527 images across six waste categories.

- Cardboard: 403 images
- Glass: 501 images
- Metal: 410 images
- Paper: 594 images
- Plastic: 482 images
- Trash: 137 images

 Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow and Keras
- OpenCV

 Project Workflow
1. Data Acquisition and Inspection
2. Image Preprocessing
3. Exploratory Data Analysis
4. K-Means Clustering
5. Supervised Learning using SVM and Random Forest
6. CNN-based Image Classification
7. Model Comparison
8. Error Analysis
9. Feature Importance
10. Hyperparameter Tuning
11. Individual and Multiple Image Prediction
 Model Performance

| Model | Test Accuracy |
|---|---:|
| SVM | 62.89% |
| Random Forest | 68.95% |
| CNN | 58.16% |

A separate diagnostic Random Forest run achieved 70.00% accuracy. A balanced practical sample of 60 images achieved 81.67% accuracy. These results represent different experiments and evaluation settings.

Project Findings
Random Forest achieved the highest test accuracy among the three models in the main comparison. The experiments also showed that some waste categories are more difficult to distinguish because of similarities in their visual features.

Project Report
The complete project report is included in this repository.

 Note
The original TrashNet image dataset is not included in this repository. Please download it separately and arrange it according to the six category folders before running the scripts.

 Conclusion
This project demonstrates how machine learning and deep learning techniques can be applied to waste image classification. It also highlights the challenges of class imbalance and visual similarities between waste categories.
