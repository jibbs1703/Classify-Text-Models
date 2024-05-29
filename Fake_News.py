# Import Necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Import Dataset (Adjust File Path)
true = pd.read_csv(filepath)
fake = pd.read_csv(filepath)

# Create Labels in Both Fake and True Datasets
fake['label'] = 0
true['label'] = 1

# Concatenate Both Datasets into One
data = pd.concat([true, fake])

# Check Full Dataset Features (Columns and Missing Observations)
data.info()

# Resample Dataset to Distribute Labels Randomly
data = data.sample(frac=1, random_state = 420).reset_index(drop=True)
X = data[ "text"]
y = data["label"]

# Split Features and Target into Test and Training Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 420)

# Check for Distribution of Fake and True Labels
X_train.value_counts()
X_test.value_counts()

# Create Pipeline with Vectorizer and ML Algorithm (Fit on Training Data)
vec = CountVectorizer()
clf = LogisticRegressionCV(max_iter=1000)
pipe = make_pipeline(vec,clf)
pipe.fit(X_train,y_train)

# Check Pipeline Performance on Training Data and Plot Confusion Matrix
# Model Performance Check
y_pred = pipe.predict(X_train)
report = metrics.classification_report(y_train, y_pred)
print(report)
print(metrics.accuracy_score(y_train,y_pred))

# Confusion Matrix of Model Performance
conf_matrix = metrics.confusion_matrix(y_train, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Model on Training Data)')
plt.show()

# Check Pipeline Performance on Test Data and Plot Confusion Matrix
# Model Performance Check
y_test_pred = pipe.predict(X_test)
report = metrics.classification_report(y_test, y_test_pred)
print(report)
print(metrics.accuracy_score(y_test,y_test_pred))

# Confusion Matrix of Model Performance
conf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Model on Testing Data)')
plt.show()