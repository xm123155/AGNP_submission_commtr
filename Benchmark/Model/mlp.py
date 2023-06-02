import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class mlp(nn.Module):
    def __init__(self, hidden_neuron_num):
        super(mlp, self).__init__()
        self.first_neuron = hidden_neuron_num
        self.second_neuron = hidden_neuron_num
        self.first_layer = nn.Linear(in_features=12, out_features=self.first_neuron)
        self.second_layer = nn.Linear(in_features=self.first_neuron, out_features=self.second_neuron)

        self.output_layer = nn.Linear(in_features=self.second_neuron, out_features=1)

    def forward(self, dataset):
        dataset = F.relu(self.first_layer(dataset))
        dataset = F.relu(self.second_layer(dataset))
        dataset = self.output_layer(dataset)

        return dataset.flatten()


def mlp_pred(x_train, y_train, x_test, y_test, train_flag, percent):
    if train_flag:
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        training_set = TensorDataset(x_train, y_train)
        training_dataloader = DataLoader(training_set, batch_size=11960, num_workers=0, drop_last=True)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    test_set = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set), num_workers=0, drop_last=True)
    y_test_pred = np.zeros(len(y_test)), 0
    if not train_flag:
        model = mlp(256).to(device)
        ckp = torch.load('logs/mlp_model_{}.pt'.format(percent))
        model.load_state_dict(ckp)
        y_test_pred = test(test_dataloader, model)
        print('**********************************')
        print("multi-layer perceptron")
    else:
        model = mlp(256).to(device)
        train(training_dataloader, model, test_dataloader, percent)
    return y_test_pred


def train(training_set, model, test_set, percent):
    loss_f = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
    num_epoch = 1000
    for epoch in range(num_epoch):
        train_loss = 0
        for image, label in training_set:
            optimizer.zero_grad()
            output = model(image.to(device))
            loss = loss_f(output, label.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.size(0)
        train_loss = train_loss / len(training_set.dataset)
        print("epoch {}, Training loss: {:.4f}".format(epoch+1, train_loss))
        if epoch % 1 == 0:
            _ = test(test_set, model)
        torch.save(model.state_dict(), 'logs/mlp_model_{}_{}.pt'.format(percent, epoch))


def test(test_set, model):
    loss = 0
    output = 0
    with torch.no_grad():
        for data, label in test_set:
            output = model(data.to(device))
            output = output.cpu().data.numpy()
            loss += mean_absolute_error(output, label)
            rmse = np.sqrt(mean_squared_error(label, output))
    # print('average score of mlp after validation: {:.3f}'.format(a))
    loss = loss / len(test_set)
    print("RMSE mean of mlp: {:.3f}".format(rmse))
    print("MAE mean of mlp: {:.3f}".format(loss))
    return output
