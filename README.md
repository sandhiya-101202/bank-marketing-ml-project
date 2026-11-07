# Bank Marketing Campaign Analysis and Prediction Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop a machine learning-based classification system to predict whether a customer will subscribe to a term deposit based on the customer's demographic details and previous marketing campaign information.

The project uses the Bank Marketing dataset and applies five classification algorithms: Logistic Regression, Decision Tree, K-Nearest Neighbors (kNN), Naive Bayes, and Support Vector Machine (SVM).

The performance of these models is evaluated using Accuracy, AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC). The main objective is to compare the models and identify the most suitable classifier for the selected dataset.

## 2. Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Dataset Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The Bank Marketing dataset contains information about a Portuguese bank's direct marketing campaigns. The data was collected to understand whether customers would subscribe to a term deposit after being contacted during a marketing campaign.

**Dataset Details:**
- Total records: 45,211
- Original attributes: 17
- Target variable: `y`
- Target classes: `yes` and `no`
- Input features used: 15

The dataset includes customer-related information such as age, job, marital status, education, balance, housing loan, personal loan, contact type, campaign details, and previous campaign information.

The `duration` feature was removed during preprocessing because call duration is known only after the customer interaction. This helps avoid using information that would not be available before making a prediction.

The target variable `y` indicates whether the customer subscribed to a term deposit:
- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe to a term deposit.

The dataset was divided into 80% training data and 20% testing data using a stratified train-test split with `random_state=42`.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/sandhiya-101202/bank-marketing-ml-project

The repository contains the project source code, preprocessing and model training files, saved machine learning models, evaluation results, application code, test dataset, and project documentation.

## 4. Models Used and Performance Comparison

Five classification algorithms were implemented and evaluated using the same Bank Marketing dataset.

### 4.1 Evaluation Metrics

The following evaluation metrics were used to compare the performance of the classification models:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between the positive and negative classes.
- **Precision:** Measures how many predicted positive instances are actually positive.
- **Recall:** Measures how many actual positive instances are correctly identified.
- **F1-Score:** Represents the harmonic mean of Precision and Recall.
- **MCC:** Measures the overall quality of binary classification predictions, considering all four confusion matrix outcomes.

### 4.2 Model Performance Comparison

The following table presents the evaluation results obtained from the test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8305 | 0.7222 | 0.3503 | 0.5255 | 0.4204 | 0.3349 |
| kNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

### 4.3 Observations on Individual Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved an AUC of 0.7722 and the highest Recall of 0.6238 among the five models. It identifies a larger proportion of actual customers who subscribed, but its Precision is relatively low at 0.2662. |
| Decision Tree | Achieved an Accuracy of 0.8305, an F1-Score of 0.4204, and an MCC of 0.3349. It provides the highest F1-Score and MCC among the five models, showing a better balance between Precision and Recall. |
| kNN | Achieved the highest Accuracy of 0.8874 and the highest Precision of 0.5529. However, its Recall of 0.1975 is the lowest, indicating that it misses many actual customers who subscribed to the term deposit. |
| Naive Bayes | Achieved an Accuracy of 0.8452 and an F1-Score of 0.4123. It provides a relatively balanced performance between Precision and Recall and has the second-highest F1-Score among the five models. |
| Support Vector Machine | Achieved an AUC of 0.7718 and a Recall of 0.6200. Its performance is similar to Logistic Regression, with relatively good Recall but lower Precision. |

### 4.4 Overall Winner for the Dataset

**Overall Winner: Decision Tree**

Based on the evaluation results, Decision Tree is selected as the overall winner for this dataset.

The Decision Tree model achieved the highest F1-Score of **0.4204** and the highest MCC of **0.3349** among all five classification models. These results indicate a better balance between Precision and Recall and better overall classification quality compared with the other evaluated models.

Although kNN achieved the highest Accuracy of 0.8874, its low Recall of 0.1975 indicates that it fails to identify many actual positive customers. Therefore, Accuracy alone is not sufficient to select the best model for this dataset.

Considering the F1-Score and MCC, Decision Tree is selected as the most suitable model among the five algorithms evaluated.

## 5. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Saving and Loading:** Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Version Control:** Git and GitHub

## 6. Project Structure

```text
bank-marketing-ml-project/
│
├── app.py
├── data_exploration.py
├── data_preprocessing.py
├── model_training.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k-nearest_neighbors.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
│
├── results/
│   ├── model_comparison.csv
│   └── confusion_matrix/
│
└── data/
    └── processed/
        ├── train_data.csv
        └── test_data.csv
```

## 7. Streamlit Application

A web application was developed using Streamlit to demonstrate the trained classification models.

The application provides the following features:

- Upload a CSV test dataset.
- Select a machine learning model using a dropdown menu.
- View the evaluation metrics: Accuracy, AUC, Precision, Recall, F1-Score, and MCC.
- View the confusion matrix and classification report.
- Compare the performance of all five classification models.
- View the prediction summary for the uploaded dataset.

The application allows users to evaluate the selected trained model using the uploaded test dataset.

## 8. Conclusion

This project compares five machine learning classification algorithms using the Bank Marketing dataset to predict customer subscription to a term deposit.

The experimental results show that each model performs differently across the evaluation metrics. kNN achieved the highest Accuracy, while Logistic Regression achieved the highest AUC and Recall. Decision Tree achieved the highest F1-Score and MCC.

Based on the overall evaluation, Decision Tree was selected as the winner because it provides a better balance between Precision and Recall and the highest MCC among the evaluated models.

The project demonstrates how machine learning classification algorithms can be applied to marketing data to support customer subscription prediction.