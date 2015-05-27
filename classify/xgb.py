import tester
import xgboost as xgb

def xgbt(X_train, target, X_test):
    dtrain = xgb.DMatrix(X_train, label=target)
    dtest = xgb.DMatrix(X_test)
    plst = [('objective', 'multi:softmax'), ('num_class', 2)]
    bst = xgb.train(plst, dtrain, 1000, [(dtrain, 'train')])

    return bst.predict(dtest)

tester.test(xgbt)


