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

## Cleaning and Preprocess Dataset Features
def nan_remover(df):
    '''
    This function removes all nan values and replaces them with spaces and then concatenates
    the three columns in the dataset into one
    '''
    for col in df.columns:
        df[col] = df[col].fillna('')
        
    df['input'] = df['keyword'].astype(str) + '_' + df['location'] + '_' + df['text']
    
    return df
​
train = nan_remover(train)
test = nan_remover(test)

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

# Use Text Cleaner on Datasets
train['input'] = train['input'].apply(text_cleaner)
test['input'] = test['input'].apply(text_cleaner)

# Create Tokens in Dataset Feature
def func_token(df):
    '''
    This function creates the tokens column, scans and takes out the stopwords and
    goes on to stem the tokenized text and update the token count.
    '''
    # Create Token Count Column
    df['tokens'] = df['input'].apply(word_tokenize)
    
    # Apply StopWorks Method on Tokens Column
    stop_words = set(stopwords.words('english'))   
    df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

    # Update the Word Token Count
    word_counts = Counter()
    df['tokens'].apply(word_counts.update)
    
    rare_words = set(word for word, count in word_counts.items() if count < 5)
    df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in rare_words])
    df['tokens'] = df.tokens.apply(', '.join)
        
    return df
trr = func_token(train)
tee = func_token(test)

## Split Datasets into Target/Features and Vectorize Features

# Separate features and target variable
X_train = trr["tokens"]
y_train = trr["target"]
X_test = tee["tokens"]
​
# Initialize Vectorizer (TF-IDF or Count)
vectorizer = CountVectorizer()
​
# Fit and transform on training data
X_train = vectorizer.fit_transform(X_train)
​
# Transform test data
X_test = vectorizer.transform(X_test)
​
# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

## Logistic Regression Model
# Instantiate Logistic Regression Model
model = LogisticRegression()

# Train the Model
model.fit(X_train_split, y_train_split)
accuracy = model.score(X_train_split, y_train_split)
print(f" Training Accuracy: {accuracy}")

# Make Predictions on the validation set
val_predictions = model.predict(X_val_split)

# Calculate F1 and Accuracy score on the validation set
f1 = f1_score(y_val_split, val_predictions)
print(f" Validation F1 Score: {f1}")
accuracy = accuracy_score(y_val_split, val_predictions)
print(f" Validation Accuracy: {accuracy}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Export Prediction to Dataframe 
results = pd.DataFrame({'id': test['id'], 'target': predictions})

## Naive Bayes Model (Complement Naive Bayes)
# Instantiate the Complement Naive Bayes Model
model = ComplementNB()

# Train the Model
model.fit(X_train_split, y_train_split)
accuracy = model.score(X_train_split, y_train_split)
print(f" Training Accuracy: {accuracy}")

# Make Predictions on the validation set
val_predictions = model.predict(X_val_split)

# Calculate F1 and Accuracy score on the validation set
f1 = f1_score(y_val_split, val_predictions)
print(f" Validation F1 Score: {f1}")
accuracy = accuracy_score(y_val_split, val_predictions)
print(f" Validation Accuracy: {accuracy}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Export Prediction to Dataframe 
results = pd.DataFrame({'id': test['id'], 'target': predictions})