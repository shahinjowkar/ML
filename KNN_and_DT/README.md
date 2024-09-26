# K-Nearest Neighbors and Decision Tree Classifiers

This repository contains an implementation of two machine learning algorithms: **K-Nearest Neighbors (KNN)** and **Decision Trees (DT)**. These models are trained and evaluated on two health-related datasets to classify age prediction and breast cancer diagnosis.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Implementation Details](#implementation-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview
This project focuses on building machine learning models from scratch in Python, specifically implementing KNN and Decision Trees. The goal is to classify data into categories by training the models on two health-related datasets and comparing their performance using evaluation metrics.

## Datasets
The two datasets used in this project are:
1. **[NHANES Age Prediction Subset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset)**: A subset of the National Health and Nutrition Examination Survey (NHANES), used to predict the age of individuals based on health-related features.
2. **[Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)**: A dataset used to classify breast cancer cases as malignant or benign based on various clinical measurements.

## Implementation Details
### 1. K-Nearest Neighbors (KNN)
- **Distance Metric**: The model uses **Euclidean distance** to calculate the distance between points. 
- **K Value**: Various values of K were tested to find the optimal number of neighbors. 
- **Hyperparameter Tuning**: K values were tuned using a train/test split to ensure the best accuracy without overfitting.
- **Normalization**: Feature scaling (normalization) was applied to ensure that features with larger ranges don’t dominate the Euclidean distance calculation.
- **Predict Function**: For a given test sample, the K-nearest neighbors are found, and the most common label among them is assigned as the prediction.

### 2. Decision Tree (DT)
- **Splitting Criterion**: The Gini Impurity is used to evaluate how well a potential split separates the classes.
- **Tree Depth**: Controlled through hyperparameter tuning to prevent overfitting. The maximum depth of the tree was adjusted to improve performance.
- **Recursive Binary Splitting**: The model recursively splits data into subgroups until each node contains only one class or meets a specified maximum depth.
- **Feature Importance**: After training, the importance of each feature is computed based on how frequently and effectively it was used to split the data.

### 3. Data Preprocessing
- **Handling Missing Data**: Missing or malformed features were removed using `dropna()`. Other imputation methods were explored but not applied in the final implementation.
- **Feature Scaling**: Normalization was performed on both datasets to ensure that features are on a comparable scale, which is critical for distance-based algorithms like KNN.
- **Class Distribution**: Basic statistics were computed, such as the mean and distribution of the target labels, to better understand the data.

### 4. Model Training and Testing
- **Train-Test Split**: Both datasets were split into training and testing sets to evaluate the model’s generalizability. A 70-30 split ratio was used to provide enough data for training while keeping sufficient data for testing.
  
## Evaluation Metrics
The performance of the models is evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions out of total predictions.
- **AUROC (Area Under the Receiver Operating Characteristic Curve)**: Measures the model's ability to distinguish between classes.
- **Confusion Matrix**: Generated to visualize the model's performance across different classes (i.e., true positives, true negatives, false positives, and false negatives).
  
Plots for ROC curves are generated to visualize the performance of KNN and Decision Tree models on the test sets.

## Usage
To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/shahinjowkar/ML.git
    ```
2. Navigate to the project directory and open the Jupyter notebook:
    ```bash
    jupyter notebook KNN_and_Decision_tree.ipynb
    ```
3. Ensure the required libraries (e.g., NumPy, Pandas, Matplotlib) are installed:
    ```bash
    pip install numpy pandas matplotlib
    ```
4. Run the notebook to train the models and evaluate their performance.

## Results
The results of the experiments show the comparative performance of KNN and Decision Tree models on the two datasets. Key findings include:

- **KNN**: Performed better with a lower K value on the NHANES dataset for age prediction.
- **Decision Trees**: Achieved higher accuracy on the Breast Cancer dataset due to the structure of the features.

Plots for accuracy and ROC curves are available in the notebook.

## References
- NHANES Age Prediction Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset)
- Breast Cancer Wisconsin Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

