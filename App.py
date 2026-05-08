import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_model import load_data, run_regression, run_classification

# --- Page Title ---
st.set_page_config(page_title="E-Clearance System", layout="wide")
st.title("📄 Automated Online Clearance Processing System")
st.subheader("Northern Iloilo State University - Lemery Campus")

# --- Sidebar Menu ---
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Student Clearance", "Payment Status", "Approval Tracking", "🤖 Machine Learning Evaluation"]
)

# --- Home Page ---
if menu == "Home":
    st.write("### Welcome to the Digital Clearance System")
    st.write("This system allows students to process clearance online with digital signatures and payment integration.")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", "1,500+")
    with col2:
        st.metric("Clearances Processed", "80-1000 per semester")
    with col3:
        st.metric("Processing Time", "Faster & Automated")

# --- Student Clearance Page ---
elif menu == "Student Clearance":
    st.write("### Student Clearance Form")
    
    student_id = st.text_input("Student ID")
    name = st.text_input("Full Name")
    course = st.selectbox("Course", ["BSIT", "BSBA", "BSN", "Others"])
    year_level = st.selectbox("Year Level", ["1st", "2nd", "3rd", "4th"])
    semester = st.selectbox("Semester", ["1st", "2nd"])
    dept_count = st.number_input("Number of Departments", min_value=1, max_value=10)
    
    if st.button("Submit Clearance Request"):
        st.success(f"✅ Clearance request submitted for {name} ({student_id})!")
        st.info("Your request will be reviewed by the departments.")

# --- Payment Status ---
elif menu == "Payment Status":
    st.write("### Payment Verification")
    
    payment_method = st.radio("Payment Method", ["GCash", "Maya", "Online Banking"])
    reference_number = st.text_input("Reference Number")
    
    if st.button("Verify Payment"):
        st.success(f"✅ Payment verified via {payment_method}!")
        st.info("Receipt generated successfully.")

# --- Approval Tracking ---
elif menu == "Approval Tracking":
    st.write("### Clearance Approval Status")
    
    data = {
        'Department': ['Registrar', 'Dean', 'Accounting', 'Library'],
        'Status': ['Approved', 'Pending', 'Approved', 'Approved'],
        'Date': ['2026-05-05', '---', '2026-05-06', '2026-05-04']
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# --- MACHINE LEARNING EVALUATION ---
elif menu == "🤖 Machine Learning Evaluation":
    st.write("## 🤖 Machine Learning Model Evaluation")
    st.write("### Lab Activity: Classification and Regression Problems")
    
    # Load your real data
    df = load_data()
    st.write("#### 📊 Your Dataset")
    st.dataframe(df, use_container_width=True)
    
    # --- Regression Section ---
    st.write("---")
    st.write("### 1. Regression Problem")
    st.write("*Goal:* Predict *Total_Days* based on student details and requirements.")
    
    reg_results, y_test, y_pred = run_regression(df)
    
    st.write("#### Evaluation Metrics:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", reg_results['MAE'])
    col2.metric("MSE", reg_results['MSE'])
    col3.metric("RMSE", reg_results['RMSE'])
    col4.metric("R² Score", reg_results['R2_Score'])
    
    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Days")
    ax.set_ylabel("Predicted Days")
    ax.set_title("Regression: Actual vs Predicted Processing Time")
    st.pyplot(fig)
    
    # --- Classification Section ---
    st.write("---")
    st.write("### 2. Classification Problem")
    st.write("*Goal:* Predict *Payment_Status* (1 = Paid, 0 = Not Paid).")
    
    class_results, cm, classes = run_classification(df)
    
    st.write("#### Evaluation Metrics:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{class_results['Accuracy']*100:.0f}%")
    col2.metric("Precision", f"{class_results['Precision']*100:.0f}%")
    col3.metric("Recall", f"{class_results['Recall']*100:.0f}%")
    col4.metric("F1 Score", f"{class_results['F1_Score']*100:.0f}%")
    
    # Confusion Matrix
    st.write("#### Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --- Footer ---
st.write("---")
st.write("© 2026 Group 7 - BSIT Capstone Project")
