# Imbalance-Learning

# Real-Fake Job Prediction Model 

This repository contains a Jupyter notebook that develops and evaluates machine learning models to classify job postings as **real** or **fraudulent**. The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction), presents a significant **class imbalance**, with fake job postings comprising only about 5% of the total data.

## Objective
To tackle imbalanced classification in fake job detection dataset using various resampling techniques and model evaluations.


<img src='https://github.com/MusadiqPasha/Imbalance-Learning/blob/main/dist.png'>

## Dataset Summary
- **Rows**: ~17,880
- **Columns**: 17
- **Target Variable**: `fraudulent` (0: real, 1: fake)
- **Imbalance**: ~5% fraudulent entries

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualized missing values and class distribution.
- Explored feature distributions using plots.
- Analyzed relationships using scatter plot matrix.

### 2. Data Preprocessing
- Handled missing values (mode fill and text defaults).
- Dropped high-null columns (e.g., `department`).
- Encoded categorical features.
- Split data into training and testing sets.

### 3. Tackling Imbalance
Applied various sampling techniques:
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Random Oversampling**
- **Random Undersampling**

### 4. Modeling
Evaluated several models:
- Logistic Regression
- Decision Tree Classifier
- Random Forest
- XGBoost
- SVM (Support Vector Machine)

### 5. Evaluation Metrics
Models were assessed using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

## Results (SMOTE-enhanced dataset)
| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 78.92%   | 77.74%    | 82.21% | 79.91%   |
| Decision Tree       | 97.26%   | 96.79%    | 97.87% | 97.33%   |
| Random Forest       | 99.15%   | 99.20%    | 99.13% | 99.17%   |
| XGBoost             | 99.11%   | 98.75%    | 99.53% | 99.13%   |
| SVM                 | 75.53%   | 78.01%    | 72.03% | 74.90%   |

## Files
- `RealFakeJobPrediction.ipynb`: Main notebook containing all steps from preprocessing to model evaluation.

---

Feel free to explore the notebook and reach out with suggestions or improvements!
