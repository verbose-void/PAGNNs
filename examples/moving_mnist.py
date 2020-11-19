import torch
from moving_mnist_utils import MovingMNIST

if __name__ == '__main__':
    batch_size = 128

    train_set = MovingMNIST(root='datasets/moving_mnist', train=True, download=True)
    test_set = MovingMNIST(root='datasets/moving_mnist', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=True)

    for X, T in train_loader:
        print(X.shape, T.shape)
        break
