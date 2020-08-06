import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import pandas as pd


class FFNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.activation = torch.sigmoid
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def one_hot(df, key):
    return pd.concat([df, pd.get_dummies(df[key], prefix=key)], axis=1).drop(key, axis=1)


def normalize_inplace(df, key):
    df[key] = df[key] / np.linalg.norm(df[key])


def separate_targets(df, target_key):
    filter_col = [col for col in df if col.startswith(target_key)]
    targets = df[filter_col]
    df = df.drop(filter_col, axis=1)

    return df, targets


def get_train_and_test(df, targets, train_perc):
    X = torch.tensor(df.to_numpy())
    T = torch.tensor(targets.to_numpy())
    T = torch.argmax(T, dim=1)
    split = int(train_perc * X.shape[0])
    train_X, test_X = X[:split], X[split:]
    train_T, test_T = T[:split], T[split:]

    return (train_X, train_T), (test_X, test_T)


def get_dataloaders(data_tensors, batch_size):
    (train_X, train_T), (test_X, test_T) = data_tensors

    train_dl = DataLoader(TensorDataset(train_X, train_T), batch_size=batch_size)
    test_dl = DataLoader(TensorDataset(test_X, test_T), batch_size=batch_size)

    return train_dl, test_dl


def compare(model_dicts, train_dl, test_dl, epochs, criterion):
    try:
        for model_dict in model_dicts:
            model_dict['train_history'] = []
            model_dict['test_history'] = []

        for epoch in range(epochs):
            print('epoch', epoch)

            for model_dict in model_dicts:
                model = model_dict['model']
                model_name = model_dict['name']
                optimizer = model_dict['optimizer']

                with torch.enable_grad():
                    model.train()
                    total_loss = 0

                    # do the training epoch
                    for x, t in train_dl:
                        optimizer.zero_grad()

                        if 'num_steps' in model_dict:
                            y = model(x, num_steps=model_dict['num_steps']).unsqueeze(0)
                        else:
                            y = model(x)

                        loss = criterion(y, t)
                        total_loss += loss.item()

                        loss.backward()
                        optimizer.step()

                    avg_loss = total_loss / len(train_dl)
                    print('[%s] training loss: %f' % (model_name, avg_loss))
                    model_dict['train_history'].append(avg_loss)


                model.eval()
                with torch.no_grad():
                    total_correct = 0.0

                    for x, t in test_dl:
                        if 'num_steps' in model_dict:
                            y = model(x, num_steps=model_dict['num_steps']).unsqueeze(0)
                        else:
                            y = model(x)

                        pred = torch.argmax(y, axis=1)
                        total_correct += torch.sum(pred == t).item()

                    accuracy = total_correct / len(test_dl.dataset)
                    print('[%s] testing accuracy:', accuracy)
                    model_dict['test_history'].append(accuracy)

    except KeyboardInterrupt:
        print('early exit keyboard interrupt')
