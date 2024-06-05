# Spam Email Detection

# Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

# Import Dataset
filepath = 'insert file path'
spam = pd.read_csv(filepath)

# Perform General Check on Data (Missing Values and Target Label Distribution)
spam.info()
spam['Category'].value_counts()
spam = spam.loc[(spam['Category'].isin(['ham', 'spam']))]

# Visualize Distribution of Target Label (Spam vs Ham)
color = ['green', 'orange']
sns.countplot(data=spam, x='Category', palette=color)
plt.show()

# Transform Target Label (0 - Ham & 1 - Spam)
spam.loc[spam['Category'] == 'ham', 'Category'] = 0
spam.loc[spam['Category'] == 'spam', 'Category'] = 1

# Split Data into Feature and Target
X = spam['Message']
y = spam['Category']

# Create Training and Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 420)

# Check Target Label in Both Training and Test Datasets
y_train.value_counts()
y_test.value_counts()

# Convert Target Labels into Integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Transform Text Data into Vectors for Model Input
# Vectorize the Text Inputs
vectorizer = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

# Transform Feature Texts into Vector Input
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Fit Model on Training Data
# Instantiate Model for Classification
model = LogisticRegressionCV(max_iter=500)

# Fit Model to Training Data
model.fit(X_train, y_train)

# Model Performance on Training Data
y_pred = model.predict(X_train)
report = metrics.classification_report(y_train, y_pred)
print(report)

f1 = metrics.f1_score(y_train, y_pred)
acc = metrics.accuracy_score(y_train, y_pred)

print(f"F1 Score: {f1}")
print(f"Accuracy Score: {acc}")

# Visualize Model Performance via the Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_train, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Training Data Confusion Matrix')
plt.show()

# Model Performance on Test Data
y_test_pred = model.predict(X_test)
report = metrics.classification_report(y_test, y_test_pred)
print(report)

f1 = metrics.f1_score(y_test,y_test_pred)
acc = metrics.accuracy_score(y_test,y_test_pred)

print(f"F1 Score: {f1}")
print(f"Accuracy Score: {acc}")

# Visualize Model Performance via the Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Data Confusion Matrix')
plt.show()
