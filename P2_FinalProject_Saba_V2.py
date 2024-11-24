import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

s = pd.read_csv('social_media_usage.csv')

st.write("Dataset Dimensions:", s.shape)

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

st.write("Mean of Variables by LinkedIn Usage (1 = LinkedIn User):")
st.write(ss.groupby('sm_li').mean())

users = ss[ss['sm_li'] == 1]
non_users = ss[ss['sm_li'] == 0]

# Income 
plt.figure(figsize=(6, 4))
plt.boxplot([non_users['income'], users['income']], labels=['Non-Users', 'Users'])
plt.title("LinkedIn Users Tend to Have Higher Income than Non-Users")
plt.ylabel("Income Level")
st.pyplot(plt)
plt.clf()

# Education 
plt.figure(figsize=(6, 4))
plt.hist(non_users['educ2'], bins=8, alpha=0.5, label='Non-Users', color='blue')
plt.hist(users['educ2'], bins=8, alpha=0.5, label='Users', color='orange')
plt.title("LinkedIn Users are More Likely to Have a Higher Education Level than Non-Users")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.legend()
st.pyplot(plt)
plt.clf()

# Age 
plt.figure(figsize=(6, 4))
plt.hist(non_users['age'], bins=15, alpha=0.5, label='Non-Users', color='blue')
plt.hist(users['age'], bins=15, alpha=0.5, label='Users', color='orange')
plt.title("Both LinkedIn Users and Non-Users Tend to Have Similar Age Trends")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
st.pyplot(plt)
plt.clf()

# Parent 
plt.figure(figsize=(6, 4))
parent_counts = ss.groupby(['parent', 'sm_li']).size().unstack()
parent_counts.plot(kind='bar', figsize=(6, 4), color=['blue', 'orange'])
plt.title("Parents have a near equal chance of being LinkedIn Users as Non-Users but Non-Parents are over twice as likely to be Non-Users")
plt.xlabel("Parent (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(['Non-Users', 'Users'], title="LinkedIn Usage")
plt.xticks(rotation=0)
st.pyplot(plt)
plt.clf()

# Female 
plt.figure(figsize=(6, 4))
female_counts = ss.groupby(['female', 'sm_li']).size().unstack()
female_counts.plot(kind='bar', figsize=(6, 4), color=['blue', 'orange'])
plt.title("Female and Non-Female Have Similar Trends of LinkedIn Usage")
plt.xlabel("Female (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(['Non-Users', 'Users'], title="LinkedIn Usage")
plt.xticks(rotation=0)
st.pyplot(plt)
plt.clf()

X = ss[['income', 'educ2', 'parent', 'married', 'female', 'age']]
y = ss['sm_li']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Training Features (X_train) shape:", X_train.shape)
st.write("Test Features (X_test) shape:", X_test.shape)
st.write("Training Target (y_train) shape:", y_train.shape)
st.write("Test Target (y_test) shape:", y_test.shape)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
cm_df = pd.DataFrame(cm, 
                     index=['Actual: Non-User', 'Actual: User'], 
                     columns=['Predicted: Non-User', 'Predicted: User'])
st.write(cm_df)

st.write("\nClassification Report:")
st.text(classification_report(y_test, y_pred))

st.header("Predict LinkedIn Usage")
age = st.number_input("Enter your age", min_value=18, max_value=98, value=30)
income = st.selectbox("Select your income range", [1, 2, 3, 4, 5, 6, 7, 8, 9])
educ2 = st.selectbox("Select your education level", [1, 2, 3, 4, 5, 6, 7, 8])
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
    st.write("You are classified as a LinkedIn user!")
else:
    st.write("You are classified as a non-LinkedIn user.")

st.write(f"Probability of using LinkedIn: {probability[0]:.4f}")
