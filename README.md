# Bank Marketing Campaign Analysis and Prediction

## 1. Project Overview
This project uses machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit based on customer information and previous marketing campaign details.

## 2. Objectives
- Perform exploratory data analysis.
- Preprocess the Bank Marketing dataset.
- Train five machine learning classification models.
- Compare model performance using evaluation metrics.
- Predict customer subscription outcomes using an interactive Streamlit application.

## 3. Dataset
The project uses the Bank Marketing dataset.

- Training dataset: `train_data.csv`
- Testing dataset: `test_data.csv`
- Input features: 15
- Target variable: Term deposit subscription (Yes/No)

## 4. Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes
5. Support Vector Machine

## 5. Evaluation Metrics
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 6. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git and GitHub

## 7. Project Structure
```text
ML_Assignment_2/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── results/
├── app.py
├── data_exploration.py
├── data_preprocessing.py
├── model_training.py
├── requirements.txt
└── test_data.csv
```

## 8. Installation and Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 9. Results
The application provides model comparison, classification reports, confusion matrices, and prediction summaries.

## 10. Conclusion
This project demonstrates the application of machine learning classification techniques to bank marketing campaign analysis and customer subscription prediction.