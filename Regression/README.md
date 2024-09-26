# Logistic and Multi-Class Regression Models

This repository contains implementations of **binary logistic regression** for sentiment classification of IMDB movie reviews and **multi-class regression** for topic classification using the 20-newsgroups dataset. The models are developed from scratch using Python, with a focus on gradient descent optimization and feature selection techniques to improve performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Implementation Details](#implementation-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Project Overview
This project implements **logistic regression** for binary classification (IMDB sentiment analysis) and **multi-class regression** for topic classification (20-newsgroups dataset). The main objectives are to implement these models from scratch, perform feature selection, and optimize using gradient descent. The performance of the models is compared against Decision Trees.

## Datasets
The two datasets used in this project are:

1. **[IMDB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)**:
   - **Task**: Binary sentiment classification (positive/negative).
   - **Preprocessing**: Feature selection based on word frequency and regression coefficients.

2. **[20 Newsgroups Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)**:
   - **Task**: Multi-class topic classification (5 selected categories).
   - **Preprocessing**: Mutual information and other feature selection techniques to identify the most relevant words for each category.

## Implementation Details
### 1. Binary Logistic Regression (IMDB)
- **Objective**: Classify movie reviews into positive or negative sentiment.
- **Model**: Logistic regression with gradient descent optimization.
- **Feature Selection**: Words were filtered based on their frequency in documents, followed by selecting the top features using Simple Linear Regression coefficients.
- **Optimization**: Gradient descent was used to minimize the cost function (cross-entropy).

### 2. Multi-Class Regression (20 Newsgroups)
- **Objective**: Classify documents into one of five distinct topics.
- **Model**: Multiclass logistic regression, implemented with the one-vs-rest (OvR) strategy.
- **Feature Selection**: Mutual Information (MI) was used to select the top features for each class, reducing the dimensionality of the dataset.
- **Optimization**: Gradient descent was employed for parameter learning.

### 3. Decision Trees (Comparison)
- As a baseline for comparison, Decision Trees from scikit-learn were applied to both the binary and multi-class classification tasks to assess performance differences with logistic regression.

### 4. Gradient Checking
- Implemented a small perturbation technique to check the correctness of gradient computations.

## Evaluation Metrics
- **Binary Logistic Regression (IMDB)**:
  - **Accuracy**: Proportion of correctly predicted reviews.
  - **AUROC**: Area Under the Receiver Operating Characteristic curve to measure model performance.

- **Multi-Class Regression (20 Newsgroups)**:
  - **Accuracy**: Proportion of correctly classified documents.
  - **Confusion Matrix**: Used to visualize the classification results across multiple categories.

## Usage
To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/shahinjowkar/ML.git
    ```
2. Navigate to the project directory and open the Jupyter notebook:
    ```bash
    jupyter notebook logistic_and_multiClass_Regression.ipynb
    ```
3. Ensure the required libraries (e.g., NumPy, Pandas, Matplotlib) are installed:
    ```bash
    pip install numpy pandas matplotlib
    ```
4. Run the notebook to train the models and evaluate their performance.

## Results
The results of the experiments show the comparative performance of logistic regression and Decision Trees on both binary and multi-class classification tasks. Key findings include:

- **Logistic Regression (IMDB)**: Achieved higher accuracy and AUROC compared to Decision Trees for sentiment classification.
- **Multi-Class Regression (20 Newsgroups)**: Logistic regression showed competitive performance, with feature selection techniques improving the model’s efficiency.

Plots for accuracy, ROC curves, and confusion matrices are available in the notebook.

## References
- IMDB Movie Reviews: [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)
- 20 Newsgroups Dataset: [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)

