# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
testDf = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
trainDf = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
# I need to combine the dfs for this one 
df = testDf.append(trainDf)

# %% [code]
print("Missing values in data: ", df.isnull().sum())

# %% [code]
# There are quite a few missing in the location. For now I will just focus on using the tweets themselves


# %% [code]
df = df[['id', 'text', 'target']]

# %% [code]
# Luckily, there is a dedicated package for tokenizing tweets
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()
df['text_tok'] = df['text'].apply(tt.tokenize)

# %% [code]
# Removing stopwords 
from nltk.corpus import stopwords
stopWords = set(stopwords.words("english"))

df['text_tok'] = df['text_tok'].apply(lambda x: [item for item in x if item not in stopWords])

# %% [code]
# Stemming words 
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = nltk.PorterStemmer()

def stemming(tokenized_text):
    stemmed_text=[ps.stem(word) for word in tokenized_text]
    return stemmed_text

df['text_stem']=df['text_tok'].apply(lambda row : stemming(row))

# %% [code]
# Joining list to get final text 
def get_final_text(stemmed_text):
    final_text=" ".join([word for word in stemmed_text])
    return final_text

df['text_final']=df['text_stem'].apply(lambda row : get_final_text(row))

# %% [code]
# Two morew steps necessary, before model creation. 
# 1. TF-IDF calculation 
# 2. Bag-of-words data structure 
# Fortunately, there is a package for this, too: 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_model=TfidfVectorizer()
tfidf_vec=tfidf_model.fit_transform(df['text_final'])
tfidf_data=pd.DataFrame(tfidf_vec.toarray())


# %% [code]
# For the creation of the bag of words I had to compute train and test-data together. Therefore
# Workaround train-test split: 
xTrain = tfidf_data.iloc[3263:10876,]
xTest = tfidf_data.iloc[0:3263,]

# %% [code]
yTrain = df.loc[df['target'].notna()]
yTrain = yTrain['target']
#xTrain.shape, yTrain.shape, xTest.shape

# %% [code]
# Calculating models 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf_model = rf.fit(xTrain, yTrain)
rf_prediction=rf_model.predict(xTest)
rf_model.score(xTrain, yTrain)
acc_rf = round(rf_model.score(xTrain, yTrain) * 100, 2)
acc_rf

# %% [code]
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(xTrain, yTrain)
Y_pred = logreg.predict(xTest)
acc_log = round(logreg.score(xTrain, yTrain) * 100, 2)
acc_log

# %% [code]
ids = df['id'].iloc[0:3263,]

# %% [code]
submission = pd.DataFrame({
        "id": ids,
        "target": rf_prediction
    })
submission.to_csv('tweets_submission.csv', index=False)