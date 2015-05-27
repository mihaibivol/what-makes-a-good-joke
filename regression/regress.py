from nltk.corpus import stopwords
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

from math import log
from random import shuffle

from processdata import get_data



TRAIN_COUNT = 20000
FUNNY_THRESHOLD = 5

####################################
# Prepare data.

jokes = get_data()

for joke in jokes:
    joke.rating = log(joke.rating) if joke.rating > 0 else 0

ratingmax = max(j.rating for j in jokes)
ratingmin = min(j.rating for j in jokes)
for joke in jokes:
    joke.rating = (joke.rating - ratingmin) / (ratingmax - ratingmin)

shuffle(jokes)
train_jokes = jokes[:TRAIN_COUNT]
test_jokes = jokes[TRAIN_COUNT:]

target = [j.rating for j in train_jokes]
test_target = [j.rating for j in test_jokes]

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

def evaluate(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())



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

#vocabulary = dictv.get_feature_names()
#model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
#model.fit(X_train)
#n_top_words=10
#for i, topic_dist in enumerate(model.topic_word_):
#    topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-n_top_words:-1]
#    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#
#

pred = [.5 for _ in test_target]
print ".5 Benchmark %f" % evaluate(pred, test_target)

from regression.xgb import xgbt
pred = xgbt(X_train, target, X_test, test_target)
print "Extreme Gradient Boosting: %f" % evaluate(pred, test_target)
