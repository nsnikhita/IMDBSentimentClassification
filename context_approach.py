import gc
import glob
import string
import statistics as stat
import numpy as np
import collections as col
import nltk
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

nltk.download('punkt')

train_pos_path = "C:/Users/nikhi/aclImdb/train/pos/"
train_neg_path = "C:/Users/nikhi/aclImdb/train/neg/"
test_pos_path = "C:/Users/nikhi/aclImdb/test/pos/"
test_neg_path = "C:/Users/nikhi/aclImdb/test/neg/"

vectorizer_pos = CountVectorizer()
vectorizer_neg = CountVectorizer()

ignore_stops = set(["cannot", "mightn't", "shan't", "don't", 'isn', 'against', 'more', "wasn't", 'no', 'wasn', "weren't", "won't", 'mustn', 'shouldn', 'hadn', 'didn', 'doesn', "should've", 'very', "doesn't", 'needn', "didn't", 'wouldn', "needn't", 'below', "hasn't", "haven't", 'not', "wouldn't", 'over', "mustn't", 'mightn', 'hasn', "hadn't", "aren't", 'ain', "couldn't", 'haven', "isn't", 'don', 'few', 'weren', 'nor', 'does', 'couldn', 'but', 'down', "shouldn't", 'aren', 'won', "mightn't", "shan't", "don't", 'isn', 'against', 'more', "wasn't", 'no', 'wasn', "weren't", "won't", 'too', 'mustn', 'shouldn', 'hadn', 'didn', 'doesn', "should've", "doesn't", 'needn', 'shan', "didn't", 'wouldn', "needn't", 'below', "hasn't", "haven't", 'not', "wouldn't", 'over', 'most', "mustn't", 'mightn', 'above', 'hasn', "hadn't", "aren't", 'ain', "couldn't", 'haven', "isn't", 'don', 'off', 'couldn', "shouldn't", 'aren', 'won'])
remove_words = set(["went", "iii"])
stp_words = set(stopwords.words('english')) - ignore_stops
stp_words |= remove_words
punct = string.punctuation.replace("-", "")

""" training data * feature values
 feature_list: no_of_keys, max_key_score, min_key_score, avg_key_scores, std_dev_key_scores, inactive_keys, inactive_key_%
               max_dist_btw_keys, min_distance_btw_keys, no_of_keys_every_10_words, no_of_keys_every_20_words, no_of_keys_every_30_words,
               count_of_keys/total_no_of_keys, std_dev_key_scores, avg_key_language_model, 
"""

# Constants used:
key_term_threshold = 3
context_span = 4
context_term_val_threshold = .05
dirichlet_const = 2000

#globals
pos_active_keys = []
pos_key_scores = {}
pos_word_freq = {}
pos_key_context_map = {}
pos_context_key_map = {}
pos_sum = 0

neg_active_keys = []
neg_key_scores = {}
neg_word_freq = {}
neg_key_context_map = {}
neg_context_key_map = {}
neg_sum = 0

# returns extracted and cleaned reviews from respective path
def get_data(path):
    data = []
    temp = 0
    for files in glob.glob(path + "*.txt"):
        if temp == 10000:
            break
        infile = open(files, encoding="utf8")
        #fix case and remove punctuations, nunbers
        dat = infile.readline().lower()
        infile.close()
        dat = dat.replace('<br />', '')
        table = str.maketrans(punct, ' '*len(punct))
        stripped = dat.translate(table)
        # filter out stop words
        stripped = word_tokenize(stripped)
        #stripped = [word for word in stripped if word.isalpha()]  reduced accuracy
        words = [w for w in stripped if not w in stp_words]
        a = ' '.join(words)
        data.append(a)
        temp += 1
    return data


# returns the key terms and their occurrences in both positive and negative reviews(helps extracting contexts)
def get_key_terms(X_train,  prim_freq, sec_freq, prim_tot, sec_tot):
    # TODO: include code to do dirichlet smoothing
    prim_train = np.array(X_train.toarray())
    temp = 0
    prim_key_terms = col.OrderedDict()
    prim_term_occurance = {}

    for key in prim_freq.keys():
        temp_sec_freq = 1
        if key in sec_freq:
            temp_sec_freq =sec_freq[key]
        #score = (prim_freq[key] / temp_sec_freq) * float(sec_tot / prim_tot)
        score  = dirichlet(prim_freq[key], prim_tot, len(prim_freq.keys())) / dirichlet(temp_sec_freq, sec_tot, len(sec_freq.keys()))
        if score > key_term_threshold:
            prim_key_terms[key] = score
            prim_term_occurance[key] = np.flatnonzero(prim_train[:, temp]).tolist()
        temp += 1
    return prim_key_terms, prim_term_occurance

# fetches the substrings 3 words left and right of each key
def fetch_substring(review, key, position = -1):
    if position == -1:
        position = review.index(key)
    start = max(0, position - context_span)
    end = min(len(review), (position + context_span + 1))
    return review[start:position] + review[(position + 1):end]


#returns dict containing the associated key term and number of occurrences of each context term and total count
def get_context_dict(key_terms, key_term_occurance, train):
    print("key terms")
    print(key_terms)
    print("key occurance")
    print(key_term_occurance)
    context_dict = {}  # dict mapping each context term to associated key word and its frequency
    for key in key_terms:
        context_per_key = {}
        for review_index in key_term_occurance[key]:
            review = train[review_index].split()
            for idx, rv in enumerate(review):
                if rv == review: #TODO: can  have multiple occurrences
                    substring = fetch_substring(review, key, idx)
                    for word in substring:
                        if word in context_per_key:
                            context_per_key[word] += 1
                        else:
                            context_per_key[word] = 1
        context_dict[key] = context_per_key
    return context_dict


# extracts context terms based on score
def get_context_terms(key_context_prim, key_context_sec, prim_key_freq, sec_key_freq, prim_sum, sec_sum):
    key_context_score_map = {}
    context_key_map = {}
    for prim_kt, prim_kt_val in key_context_prim.items():
        prim_key_val = prim_key_freq[prim_kt]
        sec_kt_val = {}
        context_score_map = {}
        neg_key_freq = 1
        if prim_kt in sec_key_freq:
            neg_key_freq = sec_key_freq[prim_kt]
        if prim_kt in key_context_sec:
            sec_kt_val = key_context_sec[prim_kt]
        for prim_context in prim_kt_val.keys():
            sec_context = 0
            if prim_context in sec_kt_val:
                 sec_context = sec_kt_val[prim_context]
            #score = float(prim_kt_val[prim_context]/ prim_key_val) - float(sec_context / neg_key_freq)
            score = dirichlet(prim_kt_val[prim_context], prim_key_val, prim_sum) - \
                    dirichlet(sec_context, neg_key_freq, sec_sum)
            if score >= context_term_val_threshold:
                context_score_map[prim_context] = score
                if prim_context in context_key_map:
                    context_key_map[prim_context].add(prim_kt)
                else:
                    context_key_map[prim_context] = set([prim_kt])
        key_context_score_map[prim_kt] = context_score_map
    return key_context_score_map, context_key_map


# returns max, avg and std dev of the distance between successive key terms in the review
def max_avg_stddev(values, min_flag = False):
    if len(values) == 1:
        return [values[0], values[0], 0]
    else:
        dif = []
        for index, val in enumerate(values):
            if min_flag:
                if len(values) == 2:
                    dif.append(values[1] - values[0])
                    break
                elif index == (len(values) - 2):
                    break
                dif.append(min((values[index + 1] - values[index]), (values[index + 2] - values[index + 1])))
            else:
                if index == (len(values) - 1):
                    break
                dif.append(values[index + 1] - values[index])

        std_dev = 0
        if len(dif) > 1:
            std_dev = stat.stdev(dif)
        return [max(dif), float(sum(dif) / len(dif)), std_dev]


# returns the sliding window max for window sizes 10, 20, 30
def sliding_window(interval_10):
    max_10 = 0
    max_20 = 0; temp_20 = 0
    max_30 = 0; temp_30 = 0
    for idx, val in enumerate(interval_10):
        temp_20 += val; temp_30 += val
        max_10 = max(max_10, val)
        if (idx + 1) % 2 == 0:
           max_20 = max(temp_20, max_20)
           temp_20 = 0
        if (idx + 1) % 3 == 0:
           max_30 = max(temp_30, max_30)
           temp_30 = 0
        if idx == (len(interval_10) - 1):
           max_20 = max(temp_20, max_20)
           max_30 = max(temp_30, max_30)
    return [max_10, max_20, max_30]


def get_max_avg_stdev(data):
    dev = 0
    maximum = 0
    avg = 0
    if len(data) > 0:
        maximum = max(data)
        avg = float(sum(data) / max(1, len(data)))
    if len(data) > 1:
        dev = stat.stdev(data)
    return [maximum, avg, dev]

# extracts features related to context terms
def context_related_features(review, key_terms, active_keys, context_stored_scores, prim_context_key_map):
    freqs = []
    percents = []
    common_context_scores = []
    context_score_ratio = []
    present_contexts_freq = {}
    common_key_freq_all = [] # no of key terms sharing same context across all reviews
    common_key_freq_review = [] # no of key terms sharing the same context per review
    all_contexts = set(list(prim_context_key_map.keys())) # set of all context terms ever encountered
    for key in key_terms.keys():
        contexts = context_stored_scores[key]
        if key in active_keys:
            for occurrence in key_terms[key]:
                substring = fetch_substring(review, key, occurrence)
                all_contexts_avg = float(sum(list(contexts.values())) / max(1, len(contexts.keys())))
                commons = set(substring).intersection(set(list(contexts.keys())))
                percents.append(float((len(commons) * 100) / max(1, len(contexts.keys()))))
                freqs.append(len(commons))
                avg = []
                for context in commons:
                    avg.append(contexts[context])
                    common_key_freq_all.append(len(prim_context_key_map[context]))
                    common_key_freq_review.append(len(set(list(key_terms.keys())).intersection(prim_context_key_map[context])))
                temp = float(sum(avg) / max(1, len(avg)))
                common_context_scores.append(temp)
                context_score_ratio.append(temp / max(1, all_contexts_avg))
                for term in substring:
                    if term in all_contexts:
                        if term in present_contexts_freq:
                            present_contexts_freq[term] += 1
                        else:
                            present_contexts_freq[term] = 1
        else:
            freqs.append(0)
            percents.append(0)
            common_context_scores.append(0)
            context_score_ratio.append(0)
    if len(freqs) == 0:
        return [0]*19
    context_features = get_max_avg_stdev(freqs)
    context_features += get_max_avg_stdev(percents)
    context_features += get_max_avg_stdev(common_context_scores)
    context_features += get_max_avg_stdev(context_score_ratio)
    context_features += get_max_avg_stdev(present_contexts_freq.values())
    context_features += get_max_avg_stdev(common_key_freq_all)[1:]
    context_features += get_max_avg_stdev(common_key_freq_review)[1:]
    return context_features


# extracts the first 23 feature values
def get_features_1_to_43(review, key_term_scores, word_frequencies, total_words, key_context_scores, prim_context_key_map, active_keys):
    features = [0]*6
    keys_inter = {}
    key_inter_pos = []
    key_scores = []
    key_language_model = []
    interval_10 = []; temp = 0
    total_count = 0
    for index, word in enumerate(review):
        if word in key_term_scores:
            total_count += 1
            temp += 1
            if word in keys_inter:
                keys_inter[word] += [index]
            else:
                keys_inter[word] = [index]
            key_inter_pos.append(index)
            key_scores.append(key_term_scores[word])
            if word in word_frequencies:
                key_language_model.append(word_frequencies[word] / total_words)
            else:
                key_language_model.append(0)
        if (index + 1) % 10 == 0:
            interval_10.append(temp)
            temp = 0
        elif index == (len(review) - 1):
            interval_10.append(temp)
    inactive_keys = set(keys_inter.keys()).difference(set(active_keys))
    if total_count != 0:
        features[0] = total_count
        features[1] = max(key_scores)
        features[2] = float(sum(key_scores) / features[0])
        if features[0] == 1:
           features[3] = 0
        else:
           features[3] = stat.stdev(key_scores)
        features[4] = len(inactive_keys)
        features[5] = float((features[4] * 100)/ total_count)
        features += get_max_avg_stdev([len(val) for val in keys_inter.values()])
        dummy  = max_avg_stddev(key_inter_pos)
        features += dummy
        features += [float(dummy[0] / len(review)), float(dummy[1] / len(review))]
        features += max_avg_stddev(key_inter_pos, True)
        features += sliding_window(interval_10)
        features.append(float(total_count / len(key_term_scores.keys())))
        if len(key_term_scores.keys()) == 1:
            features.append(0)
        else:
            features.append(stat.stdev(list(key_term_scores.values())))
        features.append(float(sum(key_language_model) / len(key_language_model)))
        if len(key_language_model) == 1:
            features.append(0)
        else:
            features.append(stat.stdev(key_language_model))
        features += context_related_features(review, keys_inter, active_keys, key_context_scores, prim_context_key_map)
    else:
        features = [0]*43 # update to total number of features
    return features


def dirichlet(numer, denom, corpus):
    return (numer + dirichlet_const * float(1 / corpus)) / (denom + dirichlet_const)

def fetch_feature_matrix(positive_reviews, negative_reviews):
    global pos_active_keys, neg_active_keys, pos_key_scores, neg_key_scores, pos_word_freq, neg_word_freq, \
         pos_sum, neg_sum, pos_key_context_map, neg_key_context_map, pos_context_key_map, neg_context_key_map

    X_train = vectorizer_pos.fit_transform(positive_reviews)
    Y_train = vectorizer_neg.fit_transform(negative_reviews)# should we fit it with same or different classifier objects ?
    train_pos = X_train.toarray().sum(axis=0)
    train_neg = Y_train.toarray().sum(axis=0)
    print ("\n  Extracting Language model.")
    print ("Total positive words = ", train_pos.sum())
    print ("Total negative words = ", train_neg.sum())  # will this be a problem since i train with same classifier

    pos_word_freq = col.OrderedDict(zip(vectorizer_pos.get_feature_names(), train_pos))
    print("pos word frequency")
    print(pos_word_freq)
    neg_word_freq = col.OrderedDict(zip(vectorizer_neg.get_feature_names(), train_neg))
    print("neg word frequency")
    print(neg_word_freq)
    print ("\n  Extracting key words.")
    gc.collect()
    neg_sum = train_neg.sum()
    pos_sum = train_pos.sum()
    neg_uniques = len(neg_word_freq.keys())
    pos_uniques = len(pos_word_freq.keys())
    pos_key_scores, key_occurrance_pos = get_key_terms(X_train, pos_word_freq, neg_word_freq, pos_uniques, neg_uniques)
    neg_key_scores, key_occurrance_neg = get_key_terms(Y_train, neg_word_freq, pos_word_freq, neg_uniques, pos_uniques)
    print (str(len(pos_key_scores.keys())), " positive key words extracted.")
    print (str(len(neg_key_scores.keys())), " negative key words extracted.")
    print ("\n  Extracting context terms.")
    # extracting context term occurrences and count
    pos_key_context_map_temp = get_context_dict(pos_key_scores.keys(), key_occurrance_pos, positive_reviews)
    neg_key_context_map_temp = get_context_dict(neg_key_scores.keys(), key_occurrance_neg, negative_reviews)
    key_occurrance_pos.clear();key_occurrance_neg.clear()
    gc.collect()

    # extracting context terms, active_pos_keys represents those positive keys with context terms associated with it
    pos_key_context_map, pos_context_key_map = get_context_terms(pos_key_context_map_temp, neg_key_context_map_temp,
                                                                 pos_word_freq, neg_word_freq, pos_uniques, neg_uniques)
    print("positive key context map")
    print (pos_key_context_map)
    # active_neg_keys represents those negative keys with context terms associated with it
    neg_key_context_map, neg_context_key_map = get_context_terms(neg_key_context_map_temp, pos_key_context_map_temp,
                                                                 neg_word_freq, pos_word_freq, neg_uniques, pos_uniques)
    print("neg key context map")
    print (neg_key_context_map)
    pos_key_context_map_temp.clear();#neg_key_context_map_temp.clear()
    gc.collect()


    feature_vals = []
    print ("Extracting Positive review Features:")
    pos_active_keys = [kt for kt in pos_key_context_map.keys() if pos_key_context_map[kt]]
    all_reviews = positive_reviews + negative_reviews
    for idx, review in enumerate(all_reviews):
        if (idx+1) % 1000 == 0:
            print (((idx + 1) * 100) / len(all_reviews), "% reviews done.")
        feature_vals.append(
            get_features_1_to_43(review.split(), pos_key_scores, pos_word_freq, pos_sum, pos_key_context_map,
                                 pos_context_key_map, pos_active_keys))
    gc.collect()

    return feature_vals


if __name__ == '__main__':

    p = Pool(2)
    [train_positive, train_negative, test_positive, test_negative] = \
        p.map(get_data, [train_pos_path, train_neg_path, test_pos_path, test_neg_path])
    gc.collect()
    print("\n Data extracting and preprocessing done. Processed review count = ", str(4 * len(train_positive)))
    print ("\n Training :")
    print(len(train_positive))
    feature_matrix_training_label = [1]*len(train_positive) + [0]*len(train_negative)
    feature_matrix_training = fetch_feature_matrix(train_positive, train_negative)

    print ("\n \n Training completed. Begin Testing!")
    all_tests = test_positive + test_negative
    print ("all test size = ", len(all_tests))
    test_features = []
    for idx, review in enumerate(all_tests):
        if (idx+1) % 1000 == 0:
            print (((idx + 1) * 100) / len(all_tests), "% reviews done.")
        test_features.append(
            get_features_1_to_43(review.split(), pos_key_scores, pos_word_freq, pos_sum, pos_key_context_map,
                                 pos_context_key_map, pos_active_keys))


    print ("\n Done")
    for arr in feature_matrix_training:
        if len(arr) != 43:
            print ("\n", len(arr))
    X = np.array(feature_matrix_training)
    y = np.array(feature_matrix_training_label)
    model = svm.SVC()
    model.fit(X, y)
    test = np.array(test_features)
    pred = model.predict(test)
    print("predicted values")
    print(pred)
    print ("Matrix size training", len(feature_matrix_training))
    print ("Matrix size testing", len(test_features))
    print ("\n Accuracy = ", (accuracy_score(y, pred) * 100), " %")