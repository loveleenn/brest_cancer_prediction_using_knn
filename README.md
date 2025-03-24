# Breast Cancer Prediction Using K-Nearest Neighbors (KNN)

## Overview

Breast cancer is one of the most common types of cancer, and early detection is crucial for effective treatment. This project utilizes the **K-Nearest Neighbors (KNN) algorithm** to predict whether a tumor is **benign or malignant** based on a set of features extracted from cell nuclei.

## Dataset

The dataset used for this project is the **Breast Cancer Wisconsin (Diagnostic) dataset**, which includes features computed from digitized images of fine needle aspirate (FNA) biopsies. Each instance in the dataset has 30 numerical attributes, including:

- Mean radius, texture, perimeter, area, smoothness, compactness, etc.
- A target variable: **0 (benign) or 1 (malignant)**

## Requirements

Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Implementation Steps

1. **Data Loading:** Load the Breast Cancer dataset from Scikit-learn.
2. **Data Preprocessing:** Handle missing values, normalize features, and split data into training and testing sets.
3. **Model Training:** Implement KNN for classification, selecting an optimal value for `k` using cross-validation.
4. **Model Evaluation:** Assess model accuracy using metrics like confusion matrix, precision, recall, and F1-score.
5. **Visualization:** Plot accuracy trends for different `k` values.

## How to Run

Run the following Python script to train and evaluate the KNN model:

```bash
python breast_cancer_knn.py
```

## Results

- Achieved high accuracy in classifying tumors as **benign or malignant**.
- Identified optimal `k` value for the best performance.
- Demonstrated the effectiveness of KNN in medical diagnosis.

## Conclusion

This project showcases how **machine learning can assist in early breast cancer detection**, potentially improving diagnosis and treatment. Further improvements can be made by experimenting with different distance metrics and feature selection techniques.

## Author

Developed by a **Data Analyst & Machine Learning Enthusiast**.



