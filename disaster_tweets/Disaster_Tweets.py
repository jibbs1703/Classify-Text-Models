# DISASTER TWEETS CLASSIFICATION

<<<<<<< Updated upstream
## Import Necessary Libraries, Load and Inspect Dataset

# Data Analysis and Visualization Libraries
=======
# Import Necessary Libraries, Load and Inspect Dataset
# Data Analysis and Visualization
>>>>>>> Stashed changes
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

<<<<<<< Updated upstream
# Text Processing Libraries
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from collections import Counter

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
=======
# Text Processing and Machine Learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
>>>>>>> Stashed changes

# Specify Dataset Path and Import Datsaets (Replace File Path as Needed)
train_path = "insert path"
test_path = "insert path"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"The train data has {train.shape[0]} rows and {train.shape[1]} columns ")
print(f"The test data has {test.shape[0]} rows and {test.shape[1]} columns ")

# Check General Condition of Datasets (Missing Values and Datatypes)
train.info()
test.info()

# Check How Disaster Tweets Look Compared to Non-Disaster Tweets
nondisaster_tweet = train[train["target"] == 0]["text"].values[np.random.randint(0, 100)]
disaster_tweet = train[train["target"] == 1]["text"].values[np.random.randint(0, 100)]
print(nondisaster_tweet)
print(disaster_tweet)

## Simple Visualizations on Dataset

# Visualize the Distribution of the Target
sns.countplot(data = train, x = 'target')
plt.xticks([0,1], ['Non-disaster Tweet', 'Disaster Tweet'])
plt.ylabel('Frequency');

# Visualize the Top 10 Tweet Locations
train['location'].value_counts().head(10).plot(kind = 'bar')
plt.xlabel('Locations')
plt.ylabel('Frequency');

# Visualize the Top 10 Most Used Keywords
train['keyword'].value_counts().head(10).plot(kind = 'bar')
plt.xlabel('Keywords')
plt.ylabel('Frequency');

## Cleaning and Preprocess Dataset Features
def nan_remover(df):
    '''
    This function removes all nan values and replaces them with spaces and then concatenates
    the three columns in the dataset into one
    '''
    for col in df.columns:
        df[col] = df[col].fillna('')

    df['input'] = df['text'] + df['keyword'] + df['location']

    return df

train = nan_remover(train)
test = nan_remover(test)

# Create Features and Target from Datasets
X_train = train['input']
X_test = test['input']
y_train = train['target']

# Create Pipeline with Vectorizer and ML Algorithm (Fit on Training Data)
vec = CountVectorizer() #  Comment out other vectorizer to Use
vec = TfidfVectorizer(stop_words="english") #  Comment out other vectorizer to Use

clf = LogisticRegressionCV(max_iter=1000) # Comment out other classifier to Use
clf = MultinomialNB() # Comment out other classifier to Use

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

# Make Prediction on Test Data, Export Prediction and Save to Output File
predictions = pipe.predict(X_test)
results = pd.DataFrame({'id': test['id'], 'target': predictions})
results.to_csv('insert path to save/submission.csv', index=False)