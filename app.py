import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# App title
st.title("🎓 Student Performance Prediction App")
st.write("Predict student final marks using Machine Learning")

# Upload dataset
uploaded_file = st.file_uploader("https://github.com/lakshmiprabha222007/student-performance/blob/main/student_performance_dataset.xlsx", type=["xlsx"])

if uploaded_file is not None:
    # Load data
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    # Input and output
    X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
    y = df['Final_Marks']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    st.subheader("🧑‍🎓 Enter Student Details")

    study_hours = st.number_input("Study Hours per Day", 0, 12, 5)
    attendance = st.number_input("Attendance (%)", 0, 100, 75)
    previous_marks = st.number_input("Previous Marks", 0, 100, 60)

    if st.button("Predict Final Marks"):
        prediction = model.predict([[study_hours, attendance, previous_marks]])
        st.success(f"📌 Predicted Final Marks: {prediction[0]:.2f}")
