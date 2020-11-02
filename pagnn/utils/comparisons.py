import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import pandas as pd

from tqdm import tqdm


def count_params(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())


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


def get_train_and_test(df, targets, train_perc, dtype=torch.float):
    X = torch.tensor(df.to_numpy(), dtype=dtype)
    T = torch.tensor(targets.to_numpy(), dtype=dtype)
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


def compare(model_dicts, train_dl, test_dl, epochs, criterion, use_tqdm=True, test_accuracy=False, device=torch.device('cpu'), flat_dim=None):
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
                    iterator = train_dl
                    if use_tqdm:
                        iterator = tqdm(iterator, desc='[train] %s' % (model_name), total=len(iterator))

                    for x, t in iterator:
                        if flat_dim is not None:
                            x = x.flatten(flat_dim)

                        x = x.to(device) #.unsqueeze(0)
                        t = t.to(device) #.unsqueeze(0)

                        optimizer.zero_grad()

                        # if 'num_steps' in model_dict:
                            # y = model(x, num_steps=model_dict['num_steps'])
                        # else:
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
                    total_correct = 0
                    total_loss = 0

                    iterator = test_dl
                    if use_tqdm:
                        iterator = tqdm(iterator, desc='[test] %s' % (model_name), total=len(iterator))

                    for x, t in iterator:
                        if flat_dim is not None:
                            x = x.flatten(flat_dim)

                        x = x.to(device)
                        t = t.to(device)

                        if 'num_steps' in model_dict:
                            y = model(x, num_steps=model_dict['num_steps'])
                        else:
                            y = model(x)

                        if test_accuracy:
                            pred = torch.argmax(y, axis=1)
                            total_correct += torch.sum(pred == t).item()
                        else:
                            total_loss += criterion(y, t).item()

                    if test_accuracy:
                        accuracy = total_correct / len(test_dl.dataset)
                        print('[%s] testing accuracy:' % model_name, accuracy)
                        model_dict['test_history'].append(accuracy)
                    else:
                        avg_loss = total_loss / len(test_dl)
                        print('[%s] testing loss: %f' % (model_name, avg_loss))
                        model_dict['test_history'].append(avg_loss)

    except KeyboardInterrupt:
        print('early exit keyboard interrupt')
