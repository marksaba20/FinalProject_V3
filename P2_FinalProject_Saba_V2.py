import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

s = pd.read_csv('social_media_usage.csv')


def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss['sm_li'] = clean_sm(ss['web1h'])

ss = ss[(ss['income'] <= 9) & (ss['educ2'] <= 8)]
ss['parent'] = clean_sm(ss['par'])
ss['married'] = clean_sm(ss['marital'])
ss['female'] = np.where(ss['gender'] == 2, 1, 0)
ss = ss[ss['age'] <= 98]
ss = ss[['sm_li', 'income', 'educ2', 'parent', 'married', 'female', 'age']]
ss = ss.dropna()


users = ss[ss['sm_li'] == 1]
non_users = ss[ss['sm_li'] == 0]

X = ss[['income', 'educ2', 'parent', 'married', 'female', 'age']]
y = ss['sm_li']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm, 
                     index=['Actual: Non-User', 'Actual: User'], 
                     columns=['Predicted: Non-User', 'Predicted: User'])



income_map = {
    1: "Less than $10,000",
    2: "10 to under $20,000",
    3: "20 to under $30,000",
    4: "30 to under $40,000",
    5: "40 to under $50,000",
    6: "50 to under $75,000",
    7: "75 to under $100,000",
    8: "100 to under $150,000",
    9: "$150,000 or more"
}

education_map = {
    1: "Less than high school (Grades 1-8 or no formal schooling)",
    2: "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "High school graduate (Grade 12 with diploma or GED certificate)",
    4: "Some college, no degree (includes some community college)",
    5: "Two-year associate degree from a college or university",
    6: "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    7: "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    8: "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
}

st.header("Mark's Handy Dandy LinkedIn Usage Identifier")

age = st.number_input("Enter your age", min_value=18, max_value=98, value=30)

income = st.selectbox("Select your income range", 
                      options=list(income_map.keys()), 
                      format_func=lambda x: income_map[x])

educ2 = st.selectbox("Select your education level", 
                     options=list(education_map.keys()), 
                     format_func=lambda x: education_map[x])

marital = st.radio("Are you married?", ("Yes", "No"))
parent = st.radio("Are you a parent?", ("Yes", "No"))
gender = st.radio("Gender", ("Female", "Male"))

user_input = pd.DataFrame({
    'income': [income],
    'educ2': [educ2],
    'parent': [1 if parent == "Yes" else 0],
    'married': [1 if marital == "Yes" else 0],
    'female': [1 if gender == "Female" else 0],
    'age': [age]
})

prediction = model.predict(user_input)
probability = model.predict_proba(user_input)[:, 1]

if prediction[0] == 1:
    st.write("You are classified as a LinkedIn user.")
else:
    st.write("You are classified as a non-LinkedIn user.")

st.write(f"Probability of using LinkedIn: {probability[0]:.4f}")

