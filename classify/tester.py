from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


from math import log
from random import shuffle

from processdata import get_data


TRAIN_COUNT = 20000
FUNNY_THRESHOLD = 5

def test(method):
    """Perform a test using method
    method: target = method(train_X, target, test_X)
    """
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
        vectorizer = TfidfVectorizer(stop_words='english')
        tokenizer = vectorizer.build_tokenizer()

        def tokenize_text(text, prefix=''):
            d = {}
            for term in tokenizer(text):
                d[prefix + term] = d.get(prefix + term, 0) + 1
            return d

        data.update(tokenize_text(joke.text, 't_'))
        data.update({('cat_' + cat): 1 for cat in joke.categories})
        data.update({('subcat_' + cat): 1 for cat in joke.subcategories})

        return data


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
    # Train test cycle.
    predictions = method(X_train, target, X_test)


    ###################################################
    good = 0.0
    bad = 0.0
    total = 0
    for idx, joke in enumerate(test_jokes):
        pred = predictions[idx]
        if pred == joke.rating:
            good += 1
        else:
            bad += 1
        total += 1

    print "%d good out of %d. Bad: %d." % (good, total, bad)
    print "%.2f" % ((good / total) * 100)
