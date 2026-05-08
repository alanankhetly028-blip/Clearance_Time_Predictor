import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load REAL DATASET ---
def load_data():
    # Load your CSV file
    df = pd.read_csv("StudentFile.Csv")
    return df

# --- 1. REGRESSION PROBLEM ---
# Problem: Predict Total_Days based on other factors
def run_regression(df):
    # Make a copy to avoid changing original
    data = df.copy()

    # Encode categorical columns
    le_course = LabelEncoder()
    le_year = LabelEncoder()
    le_sem = LabelEncoder()
    
    data['Course_enc'] = le_course.fit_transform(data['Course'])
    data['Year_enc'] = le_year.fit_transform(data['Year_Level'])
    data['Semester_enc'] = le_sem.fit_transform(data['Semester'])

    # Select features (all your attributes)
    features = [
        'No. Departments', 'Payment_Status', 'Payment_Minutes_Verify_Time',
        'Department_Count', 'Report_Mat', 'Prev.Avg_Approval',
        'Course_enc', 'Year_enc', 'Semester_enc'
    ]
    
    X = data[features]
    y = data['Total_Days']  # Target: Predict number of days

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results = {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2_Score': round(r2, 2)
    }

    return results, y_test, y_pred

# --- 2. CLASSIFICATION PROBLEM ---
# Problem: Predict Payment_Status (1 = Paid, 0 = Not Paid)
def run_classification(df):
    data = df.copy()

    # Encode categorical columns
    le_course = LabelEncoder()
    le_year = LabelEncoder()
    le_sem = LabelEncoder()
    
    data['Course_enc'] = le_course.fit_transform(data['Course'])
    data['Year_enc'] = le_year.fit_transform(data['Year_Level'])
    data['Semester_enc'] = le_sem.fit_transform(data['Semester'])

    # Select features
    features = [
        'No. Departments', 'Department_Count', 'Report_Mat',
        'Prev.Avg_Approval', 'Total_Days',
        'Course_enc', 'Year_enc', 'Semester_enc'
    ]
    
    X = data[features]
    y = data['Payment_Status']  # Target: 0 or 1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1_Score': round(f1, 2)
    }

    class_labels = ['Not Paid (0)', 'Paid (1)']
    
    return results, cm, class_labels
