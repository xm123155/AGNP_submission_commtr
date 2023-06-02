from Model.LASSO import lasso_pred
from Model.RF import rf_pred
from Model.mlp import mlp_pred
from sklearn import metrics
import numpy as np
import pickle as pkl


train = False
percent = 70


def load_dataset_agnp(data_path, percent):
    datasets = []
    for i in range(4500):
        if percent == 10:
            dataset = pkl.load(open(data_path + '/sub_data_of_time_{}.pkl'.format(i), 'rb'))
        else:
            dataset = pkl.load(open(data_path + '/sub_data_of_time_{}_{}%.pkl'.format(i, percent), 'rb'))
        datasets.append(dataset)
    datasets = np.concatenate(datasets, axis=0)
    x_train, y_train = datasets[:, :12], datasets[:, 12]
    datasets = []
    for i in range(4500, 4680):
        dataset = pkl.load(open(data_path + '/data_of_time_{}.pkl'.format(i), 'rb'))
        datasets.append(dataset)
    datasets = np.concatenate(datasets, axis=0)
    x_test, y_test = datasets[:, :12], datasets[:, 12]
    pkl.dump([x_test, y_test], open('data/traditional_data.pkl', 'wb'))
    return x_train, y_train, x_test, y_test


def main():
    l_rmse, r_rmse, m_rmse = [], [], []
    l_mae, r_mae, m_mae = [], [], []
    if train:
        x_train, y_train, x_test, y_test = load_dataset_agnp('', percent)
    else:
        x_train, y_train = [], []
        [x_test, y_test] = pkl.load(open('data/traditional_data.pkl', 'rb'))
    lasso_y_test_pred, lasso_score = lasso_pred(x_train, y_train, x_test, y_test, train, percent)
    rf_y_test_pred, rf_score = rf_pred(x_train, y_train, x_test, y_test, train, percent)
    mlp_y_test_pred, mlp_score = mlp_pred(x_train, y_train, x_test, y_test, train, percent)

    y_test = y_test[:, np.newaxis]
    lasso_y_test_pred = lasso_y_test_pred[:, np.newaxis]
    rf_y_test_pred = rf_y_test_pred[:, np.newaxis]
    mlp_y_test_pred = mlp_y_test_pred[:, np.newaxis]

    l_mae.append(metrics.mean_absolute_error(lasso_y_test_pred, y_test))
    r_mae.append(metrics.mean_absolute_error(rf_y_test_pred, y_test))
    m_mae.append(metrics.mean_absolute_error(mlp_y_test_pred, y_test))
    l_rmse.append(np.sqrt(metrics.mean_squared_error(lasso_y_test_pred, y_test)))
    r_rmse.append(np.sqrt(metrics.mean_squared_error(rf_y_test_pred, y_test)))
    m_rmse.append(np.sqrt(metrics.mean_squared_error(mlp_y_test_pred, y_test)))

    print("end")
    print("label = Idle")
    print("RMSE mean of LASSO: {:.3f}".format(np.mean(l_rmse)))
    print("MAE mean of LASSO: {:.3f}".format(np.mean(l_mae)))
    print("RMSE mean of rf: {:.3f}".format(np.mean(r_rmse)))
    print("MAE mean of rf: {:.3f}".format(np.mean(r_mae)))
    print("RMSE mean of mlp: {:.3f}".format(np.mean(m_rmse)))
    print("MAE mean of mlp: {:.3f}".format(np.mean(m_mae)))


if __name__ == '__main__':
    main()
