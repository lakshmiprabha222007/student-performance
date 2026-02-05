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
    "Upload Student Performance Excel File (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df = pd.read_excel(upload file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    required_cols = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Final_Marks']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Dataset must contain: Study_Hours, Attendance, Previous_Marks, Final_Marks")
    else:
        X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
        y = df['Final_Marks']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("✅ Machine Learning model trained successfully!")

        st.subheader("🧑‍🎓 Enter Student Details")

        study_hours = st.number_input("Study Hours per Day", 0, 12, 5)
        attendance = st.number_input("Attendance (%)", 0, 100, 75)
        previous_marks = st.number_input("Previous Marks", 0, 100, 60)

        if st.button("Predict Final Marks"):
            new_data = pd.DataFrame(
                [[study_hours, attendance, previous_marks]],
                columns=['Study_Hours', 'Attendance', 'Previous_Marks']
            )

            predicted_marks = model.predict(new_data)[0]

            st.success(f"📌 Predicted Final Marks: {predicted_marks:.2f}")
            st.info(f"📘 Previous Marks: {previous_marks}")

            # 🔍 Comparison logic
            if predicted_marks > previous_marks:
                st.success("📈 Performance Increased compared to previous marks")
            elif predicted_marks < previous_marks:
                st.warning("📉 Performance Decreased compared to previous marks")
            else:
                st.info("➖ Performance Remains the Same")
