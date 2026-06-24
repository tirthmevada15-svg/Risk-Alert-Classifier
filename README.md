# Risk Alert Classifier

Project Overview

The objective of this project is to develop a Risk Alert Classification System that identifies whether a customer is High Risk or Low Risk based on customer profile, transaction history, financial behavior, and account details.

The project applies various machine learning classification techniques and data balancing methods to improve prediction performance on an imbalanced dataset.

Dataset Information

Dataset Name: Risk_Alert_Classifier_Dataset_4600.xlsx

Total Records: 4600

Target Variable:

risk_status
0 = Low Risk
1 = High Risk

Features Used:

customer_id
age
gender
region
employment_type
annual_income_inr
credit_score
credit_utilization_ratio
missed_payments_12m
avg_late_payment_days
monthly_transaction_count
monthly_spend_inr
cash_advance_count_6m
complaints_last_6m
failed_login_attempts_3m
account_tenure_months
last_transaction_date
debt_balance_inr
Project Workflow
Step 1: Data Loading

The dataset was loaded using Pandas and checked for missing values and data types.

Step 2: Data Preprocessing

The following preprocessing steps were performed:

Converted date column into numerical format.
Encoded categorical variables using Label Encoder.
Handled missing values using KNN Imputer.
Standardized numerical features using StandardScaler.
Step 3: Train-Test Split

Dataset was divided into:

Training Data: 80%
Testing Data: 20%

using train_test_split().

Models Implemented
1. Logistic Regression

Used as the baseline classification model.

Evaluation Metrics:

Accuracy
Precision
Recall
F1 Score
AUC ROC
Confusion Matrix
2. Decision Tree Classifier

Decision Tree was implemented to compare performance with Logistic Regression.

Advantages:

Easy to understand
Handles non-linear relationships
Requires less preprocessing
3. Random Forest Classifier

Random Forest combines multiple decision trees to improve prediction performance and reduce overfitting.

Advantages:

Better accuracy
More robust
Handles complex datasets effectively
Imbalanced Data Handling Techniques

The dataset contains class imbalance. To improve prediction performance, the following techniques were applied:

Under Sampling

Reduces majority class samples.

Over Sampling

Duplicates minority class samples.

SMOTE

Creates synthetic minority class samples.

ADASYN

Generates adaptive synthetic samples for difficult minority observations.

Hyperparameter Tuning
RandomizedSearchCV

Used to find good parameter combinations efficiently.

Parameters tuned:

n_estimators
max_depth
min_samples_split
GridSearchCV

Applied after Randomized Search to find the optimal parameter combination.

Performance Metrics

The following metrics were used to evaluate the models:

Accuracy

Measures overall prediction correctness.

Precision

Measures how many predicted positive cases are actually positive.

Recall

Measures how many actual positive cases are correctly identified.

F1 Score

Balances Precision and Recall.

ROC-AUC Score

Measures model performance across different classification thresholds.

Type I and Type II Errors
Type I Error (False Positive)

Low-Risk customer classified as High-Risk.

Type II Error (False Negative)

High-Risk customer classified as Low-Risk.

In risk prediction systems, reducing Type II Error is generally more important.

ROC Curve Analysis

ROC Curves were generated for:

Logistic Regression
Decision Tree
Random Forest
Tuned Random Forest

The model with the highest AUC score was selected as the best-performing classifier.

Libraries Used
Pandas
NumPy
Matplotlib
Scikit-Learn
Imbalanced-Learn
Conclusion

This project successfully developed a Risk Alert Classification System using Machine Learning techniques. Various classification models and data balancing methods were evaluated. Logistic Regression, Decision Tree, and Random Forest were compared using multiple performance metrics. Hyperparameter tuning further improved model performance. The final model provides an effective solution for identifying high-risk customers and supports better decision-making in financial risk management.

Future Scope
Use larger real-world datasets.
Implement advanced ensemble models.
Deploy the model as a web application.
Integrate real-time risk monitoring.
Build a dashboard for risk visualization.
