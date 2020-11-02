import torch
import torch.nn.functional as F

import numpy as np
from pagnn.pagnn import PAGNNLayer, _pagnn_op

import matplotlib.pyplot as plt


def test_pagnn_op():
    # simple case
    topo = np.array([[0, 1, 1],
                     [0, 0, 0],
                     [0, 0, 0]])
    state = np.array([1, 0, 0])
    expected_next_state = state.dot(topo)
    actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
    assert np.array_equal(actual_next_state, expected_next_state)

    # add a recurrent connection
    topo = np.array([[0, 1, 1],
                     [0, 1, 0],
                     [0, 0, 0]])
    state = np.array([1, 1, 0])
    expected_next_state = state.dot(topo)
    actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
    assert np.array_equal(actual_next_state, expected_next_state)

    # add a recurrent connection & some more nodes
    topo = np.array([[0, 1, 1, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, 0]])
    state = np.array([1, 1, 0, 1])
    expected_next_state = state.dot(topo)
    actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
    assert np.array_equal(actual_next_state, expected_next_state)

    # do some random tests
    for _ in range(10):
        N = np.random.randint(5, 20)
        topo = np.random.uniform(size=(N, N))
        state = np.random.uniform(size=N)
        expected_next_state = state.dot(topo)
        actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
        assert np.allclose(actual_next_state.numpy(), expected_next_state)


def test_pagnn_op_with_batches():
    # simple case
    topo = np.array([[0, 1, 1],
                     [0, 0, 0],
                     [0, 0, 0]])
    state = np.array([[1, 0, 0],
                      [0, 1, 1]])
    expected_next_state = state.dot(topo)
    actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
    assert np.array_equal(actual_next_state, expected_next_state)

    # add a recurrent connection
    topo = np.array([[0, 1, 1],
                     [0, 1, 0],
                     [0, 0, 0]])
    state = np.array([[1, 1, 0],
                      [1, 1, 1]])
    expected_next_state = state.dot(topo)
    actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
    assert np.array_equal(actual_next_state, expected_next_state)

    # add a recurrent connection & some more nodes
    topo = np.array([[0, 1, 1, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, 0]])
    state = np.array([[1, 1, 0, 1],
                      [0, 0, 1, 0]])
    expected_next_state = state.dot(topo)
    actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
    assert np.array_equal(actual_next_state, expected_next_state)

    # do some random tests
    for _ in range(20):
        N = np.random.randint(1, 5)
        D = np.random.randint(5, 20)
        topo = np.random.uniform(size=(D, D))
        state = np.random.uniform(size=(N, D))
        expected_next_state = state.dot(topo)
        actual_next_state = _pagnn_op(torch.tensor(state), torch.tensor(topo))
        assert np.allclose(actual_next_state.numpy(), expected_next_state)


def fit(X, T, model, epochs=10, lr=0.01, return_inferences=False, batch_size=1):
    optimizer = torch.optim.SGD(lr=lr, params=model.parameters())
    criterion = torch.nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, T)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        for x, t in dataloader:
            optimizer.zero_grad()

            if batch_size > 1:
                x = x.unsqueeze(1)
                t = t.unsqueeze(1)

            y = model(x)
            loss = criterion(y, t)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(loss_sum / len(dataloader))

    if return_inferences:
        inferences = []
        for x, _ in dataloader:
            if batch_size > 1:
                x = x.unsqueeze(1)
                inferences.extend(model(x))
            else:
                inferences.append(model(x))
        return losses, inferences
    return losses


def test_linear_regression():
    for batch_size in range(1, 10):
        print(batch_size) 

        X = torch.arange(100).float()
        T = torch.tensor(np.random.uniform(low=-100, high=100) * X.numpy() + np.random.uniform(low=-100, high=100))

        # normalize the data
        X = (X - X.mean()) / X.std()
        T = (T - T.mean()) / T.std()

        linear = torch.nn.Linear(1, 1)
        pagnn = PAGNNLayer(1, 1, 0, retain_state=False)

        linear_losses = fit(X, T, linear, epochs=20, batch_size=batch_size)
        pagnn_losses = fit(X, T, pagnn, epochs=20, batch_size=batch_size)

        print('linear loss history', linear_losses)
        print('pagnn loss history', pagnn_losses)

        is_allclose = np.allclose(pagnn_losses[-1], linear_losses[-1], rtol=0.001, atol=0.001) 
        assert is_allclose


def test_bias():
    for batch_size in range(1, 5):
        X = torch.arange(100).float()
        X = (X - X.mean()) / X.std()

        T = torch.tensor(X.numpy() + np.random.uniform(low=-10, high=10))
        # T = (T - T.mean()) / T.std()

        linear = torch.nn.Linear(1, 1)
        pagnn = PAGNNLayer(1, 1, 0, retain_state=False)

        linear_losses = fit(X, T, linear, batch_size=batch_size)
        pagnn_losses = fit(X, T, pagnn, batch_size=batch_size)

        print('linear loss history', linear_losses)
        print('pagnn loss history', pagnn_losses)

        is_allclose = np.allclose(pagnn_losses[-1], linear_losses[-1], rtol=0.1, atol=0.1) 
        assert is_allclose


class NonLinear(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


def test_nonlinear_regression():
    for batch_size in range(1, 4):
        X = torch.arange(0, 2, step=0.05).float()
        T = torch.tensor(np.random.uniform(low=-100, high=100) * np.sin(X.numpy()) + np.random.uniform(low=-100, high=100))

        # normalize the data
        X = (X - X.mean()) / X.std()
        T = (T - T.mean()) / T.std()

        nonlinear = NonLinear(1, 10, 1)
        pagnn = PAGNNLayer(1, 1, 5, steps=2, retain_state=False, activation=F.relu)

        nonlinear_losses, nonlinear_Y = fit(X, T, nonlinear, lr=0.05, epochs=200, return_inferences=True, batch_size=batch_size)
        pagnn_losses, pagnn_Y = fit(X, T, pagnn, lr=0.05, epochs=200, return_inferences=True, batch_size=batch_size)

        print('nonlinear loss history', nonlinear_losses)
        print('pagnn loss history', pagnn_losses)

        is_allclose = np.allclose(pagnn_losses[-1], nonlinear_losses[-1], rtol=0.01, atol=0.01) 
        
        if not is_allclose:
            plt.plot(X, T, label='data')
            plt.plot(X, nonlinear_Y, label='baseline predictions')
            plt.plot(X, pagnn_Y, label='pagnn predictions')
            plt.legend()
            plt.show()

        assert is_allclose
