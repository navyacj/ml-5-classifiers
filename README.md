# Dry Bean Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to classify different varieties of dry beans using machine learning classification algorithms and compare their performance using multiple evaluation metrics.

## b. Dataset Description

The **Dry Bean Dataset** is a multi-class classification dataset containing numerical measurements describing the shape, size, and geometric properties of dry beans.

The target variable is `Class`.

The dataset was divided into training and testing data using an **80:20 train-test split** with stratification.

StandardScaler was applied before training the **Logistic Regression** and **K-Nearest Neighbors (KNN)** models.

## c. GitHub Repository Link

**GitHub Repository:**
https://github.com/navyacj/ml-5-classifiers.git

## d. Models Used and Evaluation

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

### Comparison Table

| ML Model Name            | Accuracy |      AUC | Precision |   Recall |       F1 |      MCC |
| ------------------------ | -------: | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression      | 0.919158 | 0.993449 |  0.919158 | 0.919158 | 0.919290 | 0.902314 |
| Decision Tree            | 0.895533 | 0.935674 |  0.895533 | 0.895533 | 0.895289 | 0.873720 |
| KNN                      | 0.915467 | 0.981051 |  0.915467 | 0.915467 | 0.915652 | 0.897791 |
| Naive Bayes              | 0.038391 | 0.500000 |  0.038391 | 0.038391 | 0.002839 | 0.000000 |
| Random Forest (Ensemble) | 0.916944 | 0.990544 |  0.916944 | 0.916944 | 0.916887 | 0.899533 |

### Observations on Model Performance

| ML Model Name            | Observation about model performance                                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved the highest Accuracy (0.919158) among the five models and also achieved the highest AUC (0.993449). It performed consistently well across all evaluation metrics. |
| Decision Tree            | Achieved an Accuracy of 0.895533. Its performance was lower than Logistic Regression, KNN and Random Forest across most metrics.                                           |
| KNN                      | Achieved an Accuracy of 0.915467 and performed well across all metrics. Its performance was close to Logistic Regression and Random Forest.                                |
| Naive Bayes              | Performed very poorly on this dataset, with an Accuracy of only 0.038391, F1 Score of 0.002839 and MCC of 0.000000.                                                        |
| Random Forest (Ensemble) | Achieved an Accuracy of 0.916944 and an AUC of 0.990544. It performed strongly and was close to Logistic Regression and KNN.                                               |

### Overall Winner

**Logistic Regression**

Logistic Regression is the overall winner for this dataset based on the obtained results. It achieved the highest **Accuracy (0.919158)**, **AUC (0.993449)**, **F1 Score (0.919290)** and **MCC (0.902314)** among the five models.
