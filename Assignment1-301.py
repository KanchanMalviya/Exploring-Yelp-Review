#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:14:34 2024

@author: kanchan
"""

#Imports
import numpy as np
import pandas as pd

# df=pd.read_csv('/Users/kanchan/Downloads/203-Data/Office.csv')

#read data

yelp = pd.read_csv('/Users/kanchan/Downloads/yelp.csv')


#Check the head, info , and describe methods on yelp
yelp.describe()
yelp.head()
yelp.shape
yelp.info()

# Use a method called value_counts to see the count distribution of stars”
yelp.stars.value_counts()
yelp.columns

# Create a new column called "text length" which is the number of words in the text column
yelp['test_length']=yelp['text'].str.len()


# df["len"] = df["text"].str.len()
# Import the data visualization libraries if you haven't done so already.

from matplotlib import pyplot as plt
import seaborn as sns

# Create a boxplot of text length for each star category.

sns.boxplot(x=yelp['stars'],y=yelp['test_length'])
                               
yelp.dtypes                             
                               

yelp.groupby('user_id')


# Use groupby to get the mean values of the numerical columns.
stars = yelp.groupby('stars')[['cool','useful','funny']].mean()
stars


# Use the corr() method on that groupby dataframe to produce this dataframe
stars.corr()


# Then use seaborn to create a heatmap based off that .corr() dataframe.
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)



#classification
# Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


# ** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**
X = yelp_class['text']
y = yelp_class['stars']

# Import CountVectorizer and create a CountVectorizer object.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X
X = cv.fit_transform(X)


#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# Import MultinomialNB and create an instance of the estimator and call is nb

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

# Use the predict method off of nb to predict labels from X_test
predictions = nb.predict(X_test)


# Create a confusion matrix and classification report using these predictions and y_test
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))



#Using text processing


from sklearn.feature_extraction.text import  TfidfTransformer

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])




X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)




pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




