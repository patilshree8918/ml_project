import pandas as pd

df = pd.read_csv('balanced_reviews.csv')

df.dropna (inplace = True)

df = df[df['overall'] != 3]

import numpy as np
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)


df['reviewText'].head()

df['reviewText'][0]

df.iloc[0,1]

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 527383):
    review = re.sub('[^a-zA-Z]', ' ', df.iloc[i,1])
    
    review = review.lower()
    
    review = review.split()
    
    #stopwords removal
    review = [word for word in review if not word in stopwords.words('english')]
    
    #steming
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review]
    
    review = " ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

features = CountVectorizer().fit_transform(corpus)

#train test split
#import class
#create
#fit
#evaluate
