import torch
import numpy as np
from pagnn.pagnn import PAGNNLayer, _pagnn_op


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


def fit(X, T, model, epochs=10):
    optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        for x, t in zip(X, T):
            optimizer.zero_grad()
            y = model(x.unsqueeze(0))
            loss = criterion(y, t.unsqueeze(0))
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(loss_sum / len(T))
    return losses


def test_linear_regression():
    for _ in range(10):
        X = torch.arange(100).float()
        T = torch.tensor(np.random.uniform(low=-100, high=100) * X.numpy() + np.random.uniform(low=-100, high=100))

        # normalize the data
        X = (X - X.mean()) / X.std()
        T = (T - T.mean()) / T.std()

        linear = torch.nn.Linear(1, 1)
        pagnn = PAGNNLayer(1, 1, 0, retain_state=False)

        linear_losses = fit(X, T, linear)
        pagnn_losses = fit(X, T, pagnn)

        print('linear loss history', linear_losses)
        print('pagnn loss history', pagnn_losses)

        assert np.allclose(pagnn_losses[-1], linear_losses[-1])
