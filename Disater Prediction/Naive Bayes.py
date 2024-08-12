import pandas as pd
import re
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

def clean_text(rawText):
    rawText = [re.sub(r'[^a-zA-Z\s]', "", text) for text in rawText]
clean_text(data_train['text'])
clean_text(data_test['text'])

count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(data_train["text"])
test_vectors = count_vectorizer.fit_transform(data_test["text"])
print(train_vectors.shape)

Y = data_train['target']
X_train, X_test, y_train, y_test = train_test_split(train_vectors, Y, test_size=0.2, random_state=41)
model = BernoulliNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_pred,y_test))