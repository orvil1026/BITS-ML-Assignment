# Machine Learning Assignment 2  
## Online Shoppers Purchasing Intention Prediction

---

## a. Problem Statement

The objective of this project is to predict whether an online website visitor will make a purchase during their browsing session.

This is a **binary classification problem**, where the target variable `Revenue` indicates whether the session resulted in a transaction (`True`) or not (`False`).

Accurate prediction of purchasing intention can help businesses:
- Improve targeted marketing strategies  
- Optimize customer engagement  
- Increase conversion rates  

---

## b. Dataset Description

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

### Dataset Overview

- **Total Instances:** 12,330  
- **Total Input Features:** 17  
- **Target Variable:** `Revenue` (True / False)  
- **Problem Type:** Binary Classification  

### Feature Categories

#### Numerical Features:
- Administrative  
- Administrative_Duration  
- Informational  
- Informational_Duration  
- ProductRelated  
- ProductRelated_Duration  
- BounceRates  
- ExitRates  
- PageValues  
- SpecialDay  

#### Categorical Features:
- Month  
- OperatingSystems  
- Browser  
- Region  
- TrafficType  
- VisitorType  
- Weekend  

### Class Distribution

- Non-Purchasing Sessions (False): 10,422 (~84.5%)  
- Purchasing Sessions (True): 1,908 (~15.5%)  

The dataset is **imbalanced**, therefore evaluation metrics beyond accuracy such as **AUC, F1-score, and Matthews Correlation Coefficient (MCC)** were emphasized.

---

## c. Models Used and Evaluation Metrics

The following six machine learning models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble Boosting)  

### Evaluation Metrics Used

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.849959 | 0.896244 | 0.510714 | 0.748691 | 0.607219 | 0.533041 |
| Decision Tree        | 0.854420 | 0.717176 | 0.530831 | 0.518325 | 0.524503 | 0.438614 |
| KNN                  | 0.868613 | 0.772459 | 0.636792 | 0.353403 | 0.454545 | 0.408458 |
| Naive Bayes          | 0.676399 | 0.797426 | 0.297665 | 0.801047 | 0.434043 | 0.333605 |
| Random Forest        | 0.896594 | 0.918795 | 0.743295 | 0.507853 | 0.603421 | 0.559490 |
| XGBoost              | 0.881184 | 0.922418 | 0.591376 | 0.753927 | 0.662831 | 0.598422 |

---

## Observations on Model Performance

## Observations on Model Performance

| ML Model Name        | Observation about Model Performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Performed strongly as a baseline model with high recall and strong AUC. It handled class imbalance reasonably well and showed balanced performance across evaluation metrics. |
| Decision Tree        | Achieved moderate accuracy but relatively low AUC, indicating weaker ranking capability. Performance suggests possible overfitting and limited generalization compared to ensemble methods. |
| K-Nearest Neighbors  | Demonstrated good precision but low recall, meaning many purchasing sessions were missed. Less suitable for imbalanced data without additional tuning. |
| Naive Bayes          | Achieved very high recall but low precision, resulting in many false positives. Although it captured most purchasing sessions, it lacked predictive precision. |
| Random Forest        | Delivered the highest accuracy and strong AUC. Provided stable and reliable performance, though recall was moderate compared to XGBoost. |
| XGBoost              | Achieved the best overall performance with the highest AUC, F1-score, and MCC. Effectively handled class imbalance and captured complex feature interactions, making it the most suitable model for this dataset. |

---



