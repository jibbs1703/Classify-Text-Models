# Text Classification 

## Overview
This repository contains projects that classify texts data using a variety of machine learning and deep 
learning models. The projects show the real world use-cases of Natural Language Processing. The tasks completed
cover disaster tweet classification, spam message detection and fake news recognition.

## Data
**Disaster Tweets Classification** : The dataset for the classification of disaster tweets was obtained from
[Kaggle](https://www.kaggle.com/competitions/nlp-getting-started). The dataset contains five columns - id (a unique identifier for each tweet), text (the text of 
the tweet), location (the location the tweet was sent from, which may be blank if the tweet is sent without its
corresponding location), keyword (a particular keyword from the tweet, which may also be blank) and the target 
(labeled as 1 - for disaster tweets and 0 - for non-disaster tweets). Disaster tweets take a prediction of 1 
while non-disaster tweets take a prediction of 0. Each model used is then scored on the F1 precision scores 
of the predictions. 

**Fake News Recognition** : The dataset for the recognition of fake news texts was obtained from 
[Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data). The dataset comes
separated in two files - one containing 23502 fake news article and the other containing 21417 true news
articles. Each file contains 4 columns - Title (title of news article), Text (body text of news article),
Subject (subject of news article) and the Date (publish date of news article). Labels are created for each
category of the news - True/Real news texts take a label of 1 while fake news texts take a label of 0. The 
model(s) used would predict the validity label of news texts. 

## Chronology of Analysis
- Import necessary libraries and datasets.
- Make general check on data for missing values or inappropriate datatypes present.
- Preprocess and feature engineer dataset using text processing methods.
- Split data into the features and target.
- Split data into training and test datasets (not necessary for Kaggle datasets).
- Training machine learning models on training dataset and check training accuracy.
- Tune Hyperparameters of the model (if necessary).
- Use trained model on test dataset and classify text based on features.
- Save predictions to a desired file format.

## Results
**Disaster Tweets** : Supervised Machine learning models were used on the disaster tweets and the F1 accuracy 
scores were recorded. The Logistic Regression Cross Validation Model correctly predicted the validity category
of 79.25%  of the tweets in the test dataset (0.7925 F1 score) while the Complement Naive Bayes Model attained 
correctly predicted the validity of 78.42%  of tweets in the test dataset (0.7842 F1 score).

**Fake News** : The Logistic Regression Cross Validation Model was used to classify the validity of news texts 
and the F1 scores were recorded.The Logistic Regression Cross Validation Model correctly predicted the validity
category of 99.51%  of the news texts in the test dataset (0.9951 F1 score).


## Authors
- [Abraham Ajibade](https://www.linkedin.com/in/abraham-olakunle-1b90bb310) 
- [Boluwatife Ajibade](https://www.linkedin.com/in/ajibade-bolu/)