import sklearn
from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
import glob, os
import scipy
from sklearn import metrics
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

onlyfiles = [f for f in listdir("C:/Users/nikhi/aclImdb/train/pos")
             if isfile(join("C:/Users/nikhi/aclImdb/train/pos", f))]

print(len(onlyfiles))

# removing .txt from names
name = []
for path in onlyfiles:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

len(name)

# splitting movie id and rating
movieId = []
rating = []
for feat in name:
    movieId.append(feat.rsplit("_", 1)[0])
    rating.append(feat.rsplit("_", 1)[1])

review = []
for path in onlyfiles:
    txt = open("C:/Users/nikhi/aclImdb/train/pos/" + path, encoding="utf8")
    review.append(txt.read())

movie_dict = {"review": review, "target": 1}
movie_df = pd.DataFrame(data=movie_dict)

onlyfilesNeg = [f for f in listdir("C:/Users/nikhi/aclImdb/train/neg") if
                isfile(join("C:/Users/nikhi/aclImdb/train/neg", f))]
print(len(onlyfilesNeg))

name = []
for path in onlyfilesNeg:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

movieId = []
rating = []
for feat in name:
    movieId.append(feat.rsplit("_", 1)[0])
    rating.append(feat.rsplit("_", 1)[1])

review = []
for path in onlyfilesNeg:
    txt = open("C:/Users/nikhi/aclImdb/train/neg/" + path, encoding="utf8")
    review.append(txt.read())

movie_dict_neg = {"review": review, "target": 0}
movie_df1 = pd.DataFrame(data=movie_dict_neg)

movieDF = movie_df1.append(movie_df, ignore_index=True)
train = movieDF.review
target = movieDF.target

print(movieDF)


stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
X = vectorizer.fit_transform(movieDF.review)

#split arrays into random train and test sets

X_train, X_test,y_train, y_test = train_test_split(X, target)

clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

testonlyfiles = [f for f in listdir("C:/Users/nikhi/aclImdb/test/pos")
                 if isfile(join("C:/Users/nikhi/aclImdb/test/pos", f))]

print(len(testonlyfiles))

# removing .txt from names
name = []
for path in testonlyfiles:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])


# splitting movie id and rating
movieId = []
rating = []
for feat in name:
    movieId.append(feat.rsplit("_", 1)[0])
    rating.append(feat.rsplit("_", 1)[1])

review = []
for path in testonlyfiles:
    txt = open("C:/Users/nikhi/aclImdb/test/pos/" + path, encoding="utf8")
    review.append(txt.read())

test_movie_dict = {"review": review, "target": 1}
test_movie_df = pd.DataFrame(data=movie_dict)
testonlyfilesNeg = [f for f in listdir("C:/Users/nikhi/aclImdb/test/neg")
                    if isfile(join("C:/Users/nikhi/aclImdb/test/neg", f))]

print(len(testonlyfilesNeg))

name = []
for path in testonlyfilesNeg:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

movieId = []
rating = []
for feat in name:
    movieId.append(feat.rsplit("_", 1)[0])
    rating.append(feat.rsplit("_", 1)[1])

review = []
for path in testonlyfilesNeg:
    txt = open("C:/Users/nikhi/aclImdb/test/neg/" + path, encoding="utf8")
    review.append(txt.read())

test_movie_dict_neg = {"review": review, "target": 0}
test_movie_df1 = pd.DataFrame(data=movie_dict_neg)

test_movieDF = test_movie_df1.append(test_movie_df, ignore_index=True)
test = test_movieDF.review
test_target = test_movieDF.target

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
X1 = vectorizer.fit_transform(test)

print(X1.shape)

clf = naive_bayes.MultinomialNB()
clf.fit(X, target)
k = clf.predict(X1)
acc = sklearn.metrics.accuracy_score(test_target, k)
print('accuracy:')
print(round(acc,2))
f_scc = sklearn.metrics.f1_score(test_target, k)
print('f1_score:')
print(round(f_scc,2))
fpr, tpr, threshold = metrics.roc_curve(test_target, k)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
