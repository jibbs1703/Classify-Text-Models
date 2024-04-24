# DISASTER TWEETS CLASSIFICATION

## Import Necessary Libraries, Load and Inspect Dataset

# Data Analysis and Visualization Libraries
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Text Processing Libraries
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from collections import Counter

# Machine Learning Libraries
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Specify Dataset Path and Import Datsaets (Replace File Path as Needed)
train_path = "C:/Users/New/GitProjects/MyProjects/Text-Classification/Disaster Tweets Classification/train.csv"
test_path = "C:/Users/New/GitProjects/MyProjects/Text-Classification/Disaster Tweets Classification/test.csv"
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

# Cleaning the Dataset Features

# Remove Unwanted Characters from Text Using Text Cleaner Function
def text_cleaner(text):
    '''
    This text cleaner function removes unwanted special characters, whitespaces
    and links from texts and returns them cleaner and devoid of these special characters
    '''
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters like '?', '#', '@', etc.
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Turns Text into Lower Case
    text = text.lower()
    
    # Remove Punctuations
    text = text.replace('[{}]'.format(string.punctuation), '')
    
    return text

# Use Text Cleaner on Datsets
train['text'] = train['text'].apply(text_cleaner)
test['text'] = test['text'].apply(text_cleaner)

# Create Function to Tokenize and Stem Text Data
def func_token(df):
    '''
    This function creates the tokens column, scans and takes out the stopwords and
    goes on to stem the tokenized text and update the token count.
    '''
    # Create Token Count Column
    df['tokens'] = df['text'].apply(word_tokenize)
    
    # Apply StopWorks Method on Tokens Column
    stop_words = set(stopwords.words('english'))   
    df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

    # Update the Word Token Count
    word_counts = Counter()
    df['tokens'].apply(word_counts.update)
    
    rare_words = set(word for word, count in word_counts.items() if count < 5)
    df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in rare_words])
        
    return df

# Use Function on Train and Test Dataset
trr = func_token(train)
tee = func_token(test)

## Split Datasets into Target/Features and Vectorize Features

# Separate features and target variable
X_train = trr["text"]
y_train = trr["target"]
X_test = tee["text"]

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform on training data
X_train = tfidf_vectorizer.fit_transform(X_train)

# Transform test data
X_test = tfidf_vectorizer.fit_transform(X_test)