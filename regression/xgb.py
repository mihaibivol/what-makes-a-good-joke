import xgboost as xgb

def xgbt(X_train, target, X_test, ytest):
    dtrain = xgb.DMatrix(X_train, label=target)
    dtest = xgb.DMatrix(X_test, label=ytest)
    plst = [('objective', 'reg:logistic'), ('eval_metric', 'rmse')]
    bst = xgb.train(plst, dtrain, 200,
                    [(dtrain, 'train'), (dtest, 'eval')])

    return bst.predict(dtest)



