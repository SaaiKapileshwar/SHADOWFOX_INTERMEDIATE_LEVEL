===============================
LOAN APPROVAL PREDICTION (ML + STREAMLIT)
===============================

1) PROJECT OVERVIEW
-------------------
This project predicts whether a loan application will be Approved or Rejected
based on applicant details such as income, credit history, education, employment,
property area, and other factors.

It is built as a Machine Learning pipeline:
- Data preprocessing (missing value handling + encoding)
- Model training (Random Forest Classifier)
- Model evaluation (accuracy, confusion matrix, classification report)
- Deployment (Streamlit web app with HTML-like form inputs)

2) DATASET
----------
File: loan_prediction.csv

Important columns:
- Loan_Status (Target): Y = Approved, N = Rejected
- Feature columns: ApplicantIncome, LoanAmount, Credit_History, etc.

The script automatically detects:
- Numerical columns (int/float)
- Categorical columns (object/text)

3) TECHNOLOGIES USED
--------------------
Language: Python 3.x

Libraries:
- pandas
- scikit-learn
- joblib
- streamlit

Model:
- RandomForestClassifier (with class_weight="balanced")

Preprocessing:
- SimpleImputer (median for numeric, most_frequent for categorical)
- OneHotEncoder (for categorical features)
- Pipeline + ColumnTransformer

4) PROJECT FILES
----------------
Recommended folder structure:

LoanApprovalProject/
│
├── loan_prediction.csv          (Dataset)
├── train_model.py               (Training + Evaluation + Save model)
├── app.py                       (Streamlit web app for prediction)
├── loan_model.pkl               (Saved trained model - generated)
├── columns.pkl                  (Saved column info - generated)
└── README.txt                   (This file)

Note:
loan_model.pkl and columns.pkl are created after running train_model.py.

5) HOW TO SET UP AND RUN
------------------------

STEP A: Install dependencies
----------------------------
Open terminal in the project folder and run:

pip install streamlit pandas numpy scikit-learn joblib

STEP B: Train the ML model
--------------------------
Run:

python train_model.py

Output:
- Prints Accuracy, Confusion Matrix, Classification Report
- Creates:
  - loan_model.pkl
  - columns.pkl

STEP C: Run the Streamlit Web App
---------------------------------
Run:

streamlit run app.py

Then open the link shown in the terminal (usually):
http://localhost:8501

6) HOW THE TRAINING WORKS (SHORT EXPLANATION)
---------------------------------------------
1. Loads the CSV dataset.
2. Drops Loan_ID if present (ID is not useful for prediction).
3. Separates the target column Loan_Status (Y/N) and converts it to numeric (1/0).
4. Splits dataset into train and test sets (80% train, 20% test) with stratification.
5. Preprocessing:
   - Numerical: fill missing values with median
   - Categorical: fill missing values with most frequent value, then OneHotEncode
6. Trains Random Forest model.
7. Evaluates on test set using Accuracy + Confusion Matrix + Classification Report.
8. Saves the trained pipeline to loan_model.pkl for deployment.

7) HOW THE STREAMLIT APP WORKS
------------------------------
1. Loads the saved model (loan_model.pkl).
2. Shows an HTML-like form interface for user input.
3. Collects user inputs and creates a DataFrame.
4. Uses model.predict() to predict loan approval.
5. Uses model.predict_proba() to show approval probability.

8) CUSTOMIZATION IDEAS
----------------------
You can improve or modify the project by:
- Trying different models (Logistic Regression, Gradient Boosting, etc.)
- Adding feature engineering (example: TotalIncome = ApplicantIncome + CoapplicantIncome)
- Hyperparameter tuning (max_depth, n_estimators, min_samples_split)
- Adding more evaluation metrics in the Streamlit app (F1-score, confusion matrix display)
- Saving and loading user predictions for history tracking

9) AUTHOR / NOTES
-----------------
This project is created for Machine Learning learning purposes.
It demonstrates the complete ML pipeline + simple web deployment.

===============================
END OF README
===============================
