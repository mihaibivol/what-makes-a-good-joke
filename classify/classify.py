import numpy as np
import lda
import nltk
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction import DictVectorizer

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree


from math import log
from random import shuffle

from processdata import get_data


TRAIN_COUNT = 20000
FUNNY_THRESHOLD = 5

####################################
# Prepare data.

jokes = get_data()

for joke in jokes:
    if joke.rating > 0:
        log_rating = log(joke.rating)
        joke.rating = 0 if log_rating < FUNNY_THRESHOLD else 1
    else:
        joke.rating = 0

shuffle(jokes)
train_jokes = jokes[:TRAIN_COUNT]
test_jokes = jokes[TRAIN_COUNT:]

target = [j.rating for j in train_jokes]

#####################################


def process_joke(joke):
    data = {}

    # Lowercase text.
    joke.text = joke.text.lower()

    # Replace text with dict.
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer()
    tokenizer = vectorizer.build_tokenizer()

    def tokenize_text(text, prefix=''):
        d = {}
        for term in tokenizer(text):
            if term in stop_words:
                continue
            d[prefix + term] = d.get(prefix + term, 0) + 1
        return d

    data.update(tokenize_text(joke.text, 't_'))
    data.update({('cat_' + cat): 1 for cat in joke.categories})
    data.update({('subcat_' + cat): 1 for cat in joke.subcategories})

    return data

def evaluate(predictions, test_data):
    good = 0.0
    bad = 0.0
    total = 0
    for idx, joke in enumerate(test_data):
        pred = predictions[idx]
        if pred == joke.rating:
            good += 1
        else:
            bad += 1
        total += 1

    return ((good / total) * 100)


#####################################
# Train data.

dict_data = [process_joke(j) for j in train_jokes]
dictv = DictVectorizer()
X_train = dictv.fit_transform(dict_data)


###################################
# Test data.

test_dict_data = [process_joke(j) for j in test_jokes]
X_test = dictv.transform(test_dict_data)

###################################


# LDA with logistic regression
#
#vocabulary = dictv.get_feature_names()
#model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
#model.fit(X_train)
#n_top_words=10
#for i, topic_dist in enumerate(model.topic_word_):
#    topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-n_top_words:-1]
#    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#
# Random forest classifier - gives about 70% accuracy
cl1 = RandomForestClassifier(n_estimators=50, verbose=1,
                                    n_jobs=4)
cl1.fit(X_train, target)
pr1 = cl1.predict(X_test)
allpred = np.array(pr1)
print"Random forest: " + "%.2f" % (evaluate(pr1, test_jokes)) + "%"


# SVM classifier - gives about 63% accuracy
cl2 = svm.SVC()
cl2.fit(X_train, target)
pr2 = cl2.predict(X_test)
allpred += pr2
print"SVM with RBF Kernel: " + "%.2f" % (evaluate(pr2, test_jokes)) + "%"


# Logistic regression classifier - gives about 81% accuracy
cl3 = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, 
                        fit_intercept=True, intercept_scaling=1, 
                        class_weight=None, random_state=None, 
                        solver='liblinear', max_iter=100, 
                        multi_class='ovr', verbose=0)
cl3.fit(X_train, target)
pr3 = cl3.predict(X_test)
allpred += pr3
print"Logistic regression: " + "%.2f" % (evaluate(pr3, test_jokes)) + "%"


# SGD classifier - gives about 73% accuracy
cl4 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                     fit_intercept=True, n_iter=5, shuffle=True, verbose=0, 
                     epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', 
                     eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                      average=False)
cl4.fit(X_train, target)
pr4 = cl4.predict(X_test)
allpred += pr4
print"SGD: " + "%.2f" % (evaluate(pr4, test_jokes)) + "%"


# KNN Classifier - gives about 59% accuracy
cl5 = NearestCentroid()
cl5.fit(X_train, target)
pr5 = cl5.predict(X_test)
print"KNN: " + "%.2f" % (evaluate(pr5, test_jokes)) + "%"


# Decision tree classifier - gives about 75% accuracy
cl6 = tree.DecisionTreeClassifier()
cl6.fit(X_train, target)
pr6 = cl6.predict(X_test)
allpred += pr6
print"Decision tree: " + "%.2f" % (evaluate(pr6, test_jokes)) + "%"


maxpred = max(allpred)
pr7 = [1 if x > maxpred / 2 else 0 for x in allpred]
print "Bagging: " + "%.2f" % (evaluate(pr7, test_jokes)) + "%"
