from sklearn.ensemble import RandomForestClassifier

from classify import tester

def random_forest(X_train, target, X_test):
    classifier = RandomForestClassifier(n_estimators=50, verbose=1,
                                        n_jobs=4)
    classifier.fit(X_train, target)

    return classifier.predict(X_test)

tester.test(random_forest)


