import numpy
import torch
import numpy as np
import torch.utils.data as Data
import pandas as pd

train_dataframe = pd.read_csv("stanford-stats-202-prediction-2024/training.csv")
test_dataframe = pd.read_csv("stanford-stats-202-prediction-2024/test.csv")
ids = test_dataframe["id"].to_numpy()
query_id = train_dataframe['query_id']
query_id_test = test_dataframe['query_id']

train_dataframe = train_dataframe.drop(columns=["query_id", "url_id", "id"])
test_dataframe = test_dataframe.drop(columns=["query_id", "url_id", "id"])

train_dataset = train_dataframe.to_numpy()

N, D_in, D_out = train_dataframe.shape[0], 39, 1
learning_rate = 1e-4
H1, H2 = 600, 200
epoch_num = 12
Batch_size = 500
module_num = 1
eps = 0.001
threshold = 0.5

columns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
inter_ = [[0, 2], [3, 4], [5, 6], [7, 8],
          [0, 3], [0, 4], [2, 5], [3, 7],
          [5, 8], [6, 9], [7, 9]]

torch.set_default_device('mps')


def inter(matx):
    for i in inter_:
        num1 = matx[:, i[0]]
        num2 = matx[:, i[1]]
        matx = torch.concatenate((matx.T, (num1 * num2)[None, :])).T
    return matx


def add_std_mean(matx, set_index, label, test):
    current_index = 0
    if test:
        query_id_ = torch.tensor(query_id_test[set_index].to_numpy(), dtype=torch.int64)
    else:
        query_id_ = torch.tensor(query_id[set_index].to_numpy(), dtype=torch.int64)
    new_matx = torch.zeros((set_index.size, 2 * columns.size))
    # new_matx = torch.zeros((set_index.size, len(cov_[0])))
    max_ = set_index.size
    label_ = label
    while current_index < max_:
        current_id = query_id_[current_index]
        group_index = query_id_ == current_id
        group_num = torch.sum(group_index)
        query_group = matx[group_index][:, columns]
        if group_num == 1:
            current_index += group_num
            continue
        else:
            if torch.sum(label_[group_index].nonzero()) == 0:
                current_index += group_num
                continue
            else:
                query_group_labeled = query_group[label_[group_index].nonzero().view(-1)]
                std_mean_ = torch.std_mean(query_group_labeled, dim=0, unbiased=False)
                std_mean_ = torch.concatenate((std_mean_[0], std_mean_[1]))
            # std_mean_ = [None, :].expand(int(group_num), -1)
        new_matx[group_index] = std_mean_
        current_index += group_num
    matx = torch.concatenate((matx.T, new_matx.T)).T
    return matx


def tran(matx, index=np.arange(N), label=torch.ones(N), test=False):
    matx = inter(matx)
    matx = add_std_mean(matx, index, label, test)
    return matx


def norm(set, mean, std):
    if std:
        set = set / torch.std(set, dim=0) + eps
    if mean:
        set = set - torch.mean(set, dim=0)
    return set


def get_model():
    model_ = torch.nn.Sequential(torch.nn.Linear(D_in, H2))
    for i in range(module_num):
        model_.append(torch.nn.Linear(H2, H1))
        model_.append(torch.nn.ReLU())
        model_.append(torch.nn.BatchNorm1d(H1))
        model_.append(torch.nn.Linear(H1, H2))
        model_.append(torch.nn.ReLU())
    model_.append(torch.nn.Linear(H2, D_out))
    return model_


def get_dataset(train_input, train_output):
    torch_dataset_ = Data.TensorDataset(train_input, train_output.to(torch.float32))
    training_set_ = Data.DataLoader(dataset=torch_dataset_, batch_size=Batch_size)
    return training_set_


def one_train(model_, training, optimizer_):
    for t, (data, label) in enumerate(training):
        optimizer_.zero_grad()
        outputs = model_(data)
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(outputs.view(-1)), label)
        loss.backward()
        optimizer_.step()


model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
x_train = train_dataset[:, :-1]
x_train = tran(x_train)
x_train = norm(x_train, True, True)
y_train = train_dataset[:, -1]
print("done training matrix")
test_dataset = torch.tensor(test_dataframe.to_numpy(), dtype=torch.float32)
test_dataset = tran(test_dataset, np.arange(30001), torch.ones(30001), True)
test_dataset = norm(test_dataset, True, True)
train_set_ = get_dataset(x_train, y_train)
print("done testing matrix")
print("start training")
for i in range(epoch_num):
    one_train(model, train_set_, optimizer)
    with torch.no_grad():
        model.eval()
        output_train = model(x_train).view(-1)
        new_labels = (torch.sigmoid(output_train) >= threshold)
        ac_train = (new_labels == y_train).to(torch.float32)
        print("epoch: {}, accuracy_train: {}".format(i, ac_train.mean()))
        model.train()
model.eval()

test_output = (torch.sigmoid(model(test_dataset)) >= threshold).view(-1)
test_output = pd.DataFrame({"id": ids, "relevance": test_output.cpu().numpy().astype(np.int64)})
test_output.to_csv("nn_prediction.csv", index=False)
