import pandas as pd

df = pd.read_csv('balanced_reviews.csv')

df.head()

df['reviewText'].head()

df['overall'].value_counts()

df.isnull().any(axis = 0)

df.dropna (inplace = True)

df['overall'] != 3

df = df[df['overall'] != 3]

df['overall'].value_counts()


import numpy as np
df['Positivity'] = np.where (df['overall'] > 3, 1, 0)

df['Positivity'].value_counts()

#reviewText - features
#Positivity - labels

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test  = train_test_split(df['reviewText'], df['Positivity'], random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(features_train)

len(vect.get_feature_names())

vect.get_feature_names()[10000:10010]


features_train_vectorized = vect.transform(features_train)

#Classification
#KNN, SVM, Naive Bayes, Logistic Regression, Decision Tree, Random Forest
#

#model 01
from sklearn.linear_model import LogisticRegression


model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix

confusion_matrix(labels_test, predictions)


from sklearn.metrics import roc_auc_score

roc_auc_score(labels_test, predictions)


#model version 02
#tf-idf
#term frequency inverse document frequency

features_train, features_test, labels_train, labels_test  = train_test_split(df['reviewText'], df['Positivity'], random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer


vect = TfidfVectorizer(min_df = 5).fit(features_train)

len(vect.get_feature_names())

vect.get_feature_names()[10000:10010]


features_train_vectorized = vect.transform(features_train)

#features_train_vectorized.toarray()

#Classification
# Logistic Regression

#model 01
from sklearn.linear_model import LogisticRegression


model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix


confusion_matrix(labels_test, predictions)


from sklearn.metrics import roc_auc_score

roc_auc_score(labels_test, predictions)



import pickle

pkl_filename = "pickle_model.pkl"


file = open(pkl_filename, 'wb')

pickle.dump(model, file)

file.close()


#-------------------------
#sylvester machine has to run below code to load the model

file = open(pkl_filename, 'rb')
recreated_model = pickle.load(file)

predictions = recreated_model.predict(vect.transform(features_test))


from sklearn.metrics import roc_auc_score

roc_auc_score(labels_test, predictions)