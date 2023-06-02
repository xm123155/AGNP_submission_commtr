from sklearn.ensemble import RandomForestRegressor
import pickle as pkl


def rf_pred(x_train, y_train, x_test, y_test, train, percent):
    # random forest
    if train:
        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)
        pkl.dump(rf, open('logs/rf_{}.pkl'.format(percent), 'wb'))
        print('**********************************')
        print("random forest")
        print("training set score:{:.3f}".format(rf.score(x_train, y_train)))
    else:
        rf = pkl.load(open('logs/rf_{}.pkl'.format(percent), 'rb'))
        print('**********************************')
        print("random forest")

    y_test_pred = rf.predict(x_test)
    return y_test_pred, rf.score(x_test, y_test)
