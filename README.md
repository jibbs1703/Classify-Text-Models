# Text Classification 

## Overview
This repository contains projects that classify texts data using a variety of machine learning and deep learning models. The projects show the real world use-cases of Natural Language Processing. The tasks completed cover disaster tweet classification, spam message detection and 

## Data
**Disaster Tweets Classification** : The dataset for the classification of disaster tweets was obtained from [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started). The dataset contains five columns - id (a unique identifier for each tweet), text (the text of the tweet), location (the location the tweet was sent from, which may be blank if the tweet is sent without its corresponding location), keyword (a particular keyword from the tweet, which may also be blank) and the target (labeled as 1 - for disaster tweets and 0 - for non-disaster tweets). 
The target is dropped in test dataset. With the test dataset, predictions are made on whether a given tweet is about a real disaster or not. Disaster tweets take a prediction of 1 while non-disaster tweets take a prediction of 0. Each model used is then scored on the f1 precision scores of the predictions. 

## Chronology of Analysis
- Import necessary libraries and datasets.
- Make general check on data for missing values or inappropriate datatypes present.
- Preprocess and feature engineer dataset using text processing methods (tokenize, stem, vectorize).
- Split data into the features and target.
- Split data into training and test datasets (not necessary for Kaggle datasets).
- Training machine learning models on training dataset and check training accuracy.
- Tune Hyperparameters of the model (if necessary).
- Use trained model on test dataset and classify text based on features.
- Save predictions to a desired file format.

## Results
- **Disaster Tweets** : Supervised Machine learning models were used on the disaster tweets and the f1 accuracy scores were recorded after the [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started) submission. The Logistic Regression Model attained a 90.05% accuracy on the Training Dataset and correctly predicted the validity of 76.65%  of tweets in the test dataset (0.7665 F1 score). The Complement Naive Bayes Model attained a 84.20% accuracy on the Training Dataset and correctly predicted the validity of 76.98%  of tweets in the test dataset (0.7698 F1 score).

- **Spam Detection**: 

- **Yelp Review**:

## Authors
- [Abraham Ajibade](https://www.linkedin.com/in/abraham-ajibade-759772117/) 
- [Boluwatife Ajibade](https://www.linkedin.com/in/ajibade-bolu/)