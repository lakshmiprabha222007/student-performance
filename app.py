import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# App title
st.title("🎓 Student Performance Prediction")
st.write("Predict student final marks using Machine Learning")

# Upload Excel file
uploaded_file = st.file_uploader(
    "https://github.com/lakshmiprabha222007/student-performance/blob/main/student_performance_dataset.xlsx",
    type=["xlsx"]
)

if uploaded_file is not None:
    # Read dataset
    df = pd.read_excel("https://github.com/lakshmiprabha222007/student-performance/blob/main/student_performance_dataset.xlsx")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    # Check required columns
    required_cols = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Final_Marks']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Dataset must contain: Study_Hours, Attendance, Previous_Marks, Final_Marks")
    else:
        # Split input and output
        X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
        y = df['Final_Marks']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("✅ Machine Learning model trained successfully!")

        st.subheader("🧑‍🎓 Enter Student Details")

        study_hours = st.number_input("Study Hours per Day", min_value=0, max_value=12, value=5)
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
        previous_marks = st.number_input("Previous Marks", min_value=0, max_value=100, value=60)

        if st.button("Predict Final Marks"):
            # Create DataFrame for prediction (IMPORTANT FIX)
            new_data = pd.DataFrame(
                [[study_hours, attendance, previous_marks]],
                columns=['Study_Hours', 'Attendance', 'Previous_Marks']
            )

            prediction = model.predict(new_data)

            st.success(f"📌 Predicted Final Marks: {prediction[0]:.2f}")
