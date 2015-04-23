from sklearn.naive_bayes import MultinomialNB

from classify import tester

def random_forest(X_train, target, X_test):
    classifier = MultinomialNB()
    classifier.fit(X_train, target)

    return classifier.predict(X_test)

tester.test(random_forest)


