from sklearn.linear_model import Lasso
import numpy as np
import pickle as pkl


def lasso_pred(x_train, y_train, x_test, y_test, train, percent):
    # LASSO
    if train:
        lasso = Lasso(alpha=0.1)
        lasso.fit(x_train, y_train)
        pkl.dump(lasso, open('logs/lasso_{}.pkl'.format(percent), 'wb'))
        print('**********************************')
        print("Lasso alpha=0.1")
        print("training set score:{:.3f}".format(lasso.score(x_train, y_train)))
    else:
        lasso = pkl.load(open('logs/lasso_{}.pkl'.format(percent), 'rb'))
        print('**********************************')
        print("Lasso alpha=0.1")

    y_test_pred = lasso.predict(x_test)
    print("Number of features used:{}".format(np.sum(lasso.coef_ != 0)))
    return y_test_pred, lasso.score(x_test, y_test)
