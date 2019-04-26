import sklearn
from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
import glob, os
import scipy
from sklearn import metrics

import matplotlib.pyplot as plt

onlyfiles = [f for f in listdir("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/train/pos")
             if isfile(join("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/train/pos", f))]
len(onlyfiles)

# removing .txt from names
name = []
for path in onlyfiles:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

len(name)

# splitting movie id and rating
movieId=[]
rating=[]
for feat in name:
    movieId.append(feat.rsplit( "_", 1 )[ 0 ])
    rating.append(feat.rsplit( "_", 1 )[ 1 ])

review = []
for path in onlyfiles:
    txt = open("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/train/pos/"+path, encoding="utf8")
    review.append(txt.read())

movie_dict = {"review": review, "target": 1}
movie_df = pd.DataFrame(data=movie_dict)

onlyfilesNeg = [f for f in listdir("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/train/neg") if isfile(join("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/train/neg", f))]

name =[]
for path in onlyfilesNeg:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

movieId=[]
rating=[]
for feat in name:
    movieId.append(feat.rsplit( "_", 1 )[ 0 ])
    rating.append(feat.rsplit( "_", 1 )[ 1 ])

review =[]
for path in onlyfilesNeg:
    txt = open("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/train/neg/"+path, encoding="utf8")
    review.append(txt.read())


movie_dict_neg = {"review": review, "target": 0}
movie_df1 = pd.DataFrame(data=movie_dict_neg)

movieDF = movie_df1.append(movie_df,ignore_index = True)
train = movieDF.review
target=movieDF.target

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(movieDF.review)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
X_train_tf = tf_transformer.transform(X_train)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,max_iter=5, tol=None)),])

text_clf.fit(train,target)

# ##########################3getting test data###################################3333
testonlyfiles = [f for f in listdir("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/test/pos")
             if isfile(join("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/test/pos", f))]

len(testonlyfiles)

# removing .txt from names
name = []
for path in testonlyfiles:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

len(name)

# splitting movie id and rating
movieId=[]
rating=[]
for feat in name:
    movieId.append(feat.rsplit( "_", 1 )[ 0 ])
    rating.append(feat.rsplit( "_", 1 )[ 1 ])


review = []
for path in testonlyfiles:
    txt = open("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/test/pos/"+path, encoding="utf8")
    review.append(txt.read())

test_movie_dict = {"review": review, "target": 1}
test_movie_df = pd.DataFrame(data=movie_dict)
testonlyfilesNeg = [f for f in listdir("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/test/neg")
                    if isfile(join("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/test/neg", f))]

name =[]
for path in testonlyfilesNeg:
    os.path.splitext(path)
    name.append(os.path.splitext(path)[0])

movieId=[]
rating=[]
for feat in name:
    movieId.append(feat.rsplit( "_", 1 )[ 0 ])
    rating.append(feat.rsplit( "_", 1 )[ 1 ])

review =[]
for path in testonlyfilesNeg:
    txt = open("C:/Users/bluer/Documents/IR Notes/aclImdb_v1.tar/aclImdb_v1/aclImdb/test/neg/"+path, encoding="utf8")
    review.append(txt.read())


test_movie_dict_neg = {"review": review, "target": 0}
test_movie_df1 = pd.DataFrame(data=movie_dict_neg)

test_movieDF = test_movie_df1.append(test_movie_df,ignore_index = True)
test = test_movieDF.review
test_target = test_movieDF.target

k = text_clf.predict(test)
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



