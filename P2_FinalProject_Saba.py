#Q1
import pandas as pd

s = pd.read_csv('social_media_usage.csv')

print(s.shape)

import numpy as np

#Q2
def clean_sm(x):
    return np.where(x == 1, 1, 0)

data = {'col1': [1, 2, 2], 'col2': [1, 1, 9]}
df = pd.DataFrame(data)

print(clean_sm(df))

#Q3
import matplotlib.pyplot as plt
import seaborn as sns

ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

ss['sm_li'] = clean_sm(ss['web1h'])

ss = ss[(ss['income'] <= 9) & (ss['educ2'] <= 8)]

ss['parent'] = clean_sm(ss['par'])
ss['married'] = clean_sm(ss['marital'])
ss['female'] = np.where(ss['gender'] == 2, 1, 0)

ss = ss[ss['age'] <= 98]

ss = ss[['sm_li', 'income', 'educ2', 'parent', 'married', 'female', 'age']]

ss = ss.dropna()

print("Mean of Variables by LinkedIn Usage (1 = LinkedIn User)")
print(ss.groupby('sm_li').mean())

# Assign LinkedIn Users
users = ss[ss['sm_li'] == 1]
non_users = ss[ss['sm_li'] == 0]

# Income
plt.figure(figsize=(6, 4))
plt.boxplot([non_users['income'], users['income']], labels=['Non-Users', 'Users'])
plt.title("LinkedIn Users Tend to Have Higher Income than Non-Users")
plt.ylabel("Income Level")
plt.show()

# Education
plt.figure(figsize=(6, 4))
plt.hist(non_users['educ2'], bins=8, alpha=0.5, label='Non-Users', color='blue')
plt.hist(users['educ2'], bins=8, alpha=0.5, label='Users', color='orange')
plt.title("LinkedIn Users are More Likely to have a Higher Education Level than Non-Users")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.legend()
plt.show()

# Age
plt.figure(figsize=(6, 4))
plt.hist(non_users['age'], bins=15, alpha=0.5, label='Non-Users', color='blue')
plt.hist(users['age'], bins=15, alpha=0.5, label='Users', color='orange')
plt.title("Both LinkedIn Users and Non-Users Tend to Have Similar Age Trends")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# Parent
plt.figure(figsize=(6, 4))
parent_counts = ss.groupby(['parent', 'sm_li']).size().unstack()
parent_counts.plot(kind='bar', figsize=(6, 4), color=['blue', 'orange'])
plt.title("Parents have a near equal chance of being LinkedIn Users as Non-Users but Non-Parents are over twice as likely to be Non-Users")
plt.xlabel("Parent (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(['Non-Users', 'Users'], title="LinkedIn Usage")
plt.xticks(rotation=0)
plt.show()

# Female
plt.figure(figsize=(6, 4))
parent_counts = ss.groupby(['female', 'sm_li']).size().unstack()
parent_counts.plot(kind='bar', figsize=(6, 4), color=['blue', 'orange'])
plt.title("Female and Non-Female Have Similar Trends of LinkedIn Usage")
plt.xlabel("Female (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(['Non-Users', 'Users'], title="LinkedIn Usage")
plt.xticks(rotation=0)
plt.show()

#Q4
from sklearn.model_selection import train_test_split

X = ss[['income', 'educ2', 'parent', 'married', 'female', 'age']]
y = ss['sm_li']

#Q5
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Features (X_train) shape:", X_train.shape)
print("Test Features (X_test) shape:", X_test.shape)
print("Training Target (y_train) shape:", y_train.shape)
print("Test Target (y_test) shape:", y_test.shape)

#Q6
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

#Q7
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

#Q8
cm_df = pd.DataFrame(cm, 
                     index=['Actual: Non-User', 'Actual: User'], 
                     columns=['Predicted: Non-User', 'Predicted: User'])

print(cm_df)

#Q9
from sklearn.metrics import classification_report

TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Manual Calculation of Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Q10
# Income 2 vs 5 vs 8
c1 = pd.DataFrame({
    'income': [2],
    'educ2': [6],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [30]
})

c2 = pd.DataFrame({
    'income': [5],
    'educ2': [6],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [30]
})

c3 = pd.DataFrame({
    'income': [8],
    'educ2': [6],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [30]
})


prob_c1 = model.predict_proba(c1)[:, 1]
prob_c2 = model.predict_proba(c2)[:, 1]
prob_c3 = model.predict_proba(c3)[:, 1]

# Display the results
print(f"Probability of using LinkedIn (C1 - Low Income): {prob_c1[0]:.4f}")
print(f"Probability of using LinkedIn (C2 - Medium Income): {prob_c2[0]:.4f}")
print(f"Probability of using LinkedIn (C3 - High Income): {prob_c3[0]:.4f}")

print("\nWhen evaluating probability for 30-year-old married, non-parent females with a four-year college or university degree/Bachelor’s degree across three different income levels, we find that those with an income of 10 to under $20,000 have a 35.6% probability of using LinkedIn, those with an income of 40 to under $50,000 have a 55.7% probability of using LinkedIn, and those with an income of $150,000 or more have a 74.1% probability of using LinkedIn.")