import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pagnn.pagnn import PAGNNLayer
from pagnn.utils.comparisons import compare, FFNN, count_params
from pagnn.utils.visualize import draw_networkx_graph


if __name__ == '__main__':
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])

    train_dataset = torchvision.datasets.MNIST('datasets', download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('datasets', download=True, train=False, transform=transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=6)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=6)

    D = 28*28
    C = 10
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dicts = []

    linear_lr = 0.001
    linear_model = torch.nn.Linear(D, C)
    linear_model.to(device)
    linear = {
        'name': 'Linear(%i, %i, #p=%i)' % (D, C, count_params(linear_model)),
        'model': linear_model,
        'optimizer': torch.optim.Adam(linear_model.parameters(), lr=linear_lr),
    }

    model_dicts.append(linear)

    configs = [
        {'num_steps': 1, 'activation': None},
        {'num_steps': 3, 'activation': None},
        {'num_steps': 5, 'activation': None},
        {'num_steps': 3, 'activation': 'relu'},
        {'num_steps': 5, 'activation': 'relu'},
    ]

    for config in configs:
        pagnn_lr = 0.001
        extra_neurons = 0
        n = D + C + extra_neurons
        activation_func = None
        if config['activation'] == 'relu':
            activation_func = F.relu
        elif config['activation'] is not None:
            raise Exception()
        pagnn_model = PAGNNLayer(D, C, extra_neurons, steps=config['num_steps'], retain_state=False, activation=activation_func) # graph_generator=nx.generators.classic.complete_graph)
        pagnn_model.to(device)
        if config['activation'] is None:
            model_name = 'PAGNNLayer(n=%i, #p=%i, steps=%i)' % (n, count_params(pagnn_model), config['num_steps'])
        else:
            model_name = 'PAGNNLayer(n=%i, #p=%i, steps=%i) + %s' % (n, count_params(pagnn_model), config['num_steps'], config['activation'])
        pagnn = {
            'name': model_name,
            'model': pagnn_model,
            # 'num_steps': config['num_steps'],
            'optimizer': torch.optim.Adam(pagnn_model.parameters(), lr=pagnn_lr),
        }

        model_dicts.append(pagnn)

    ffnn_lr = 0.001
    ffnn_model = FFNN(D, D, C)
    ffnn_model.to(device)
    ffnn = {
        'name': 'FFNN(%i, %i, %i, #p=%i)' % (D, D, C, count_params(ffnn_model)),
        'model': ffnn_model,
        'optimizer': torch.optim.Adam(ffnn_model.parameters(), lr=ffnn_lr),
    }

    model_dicts.append(ffnn)

    # print('pagnn num params:', sum(p.numel() for p in pagnn_model.parameters()))
    # print('ffnn num params:', sum(p.numel() for p in ffnn_model.parameters()))

    criterion = F.cross_entropy
    epochs = 10
    compare(model_dicts, train_dl, test_dl, epochs, criterion, test_accuracy=True, device=device, flat_dim=1)
    
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('MNIST - (PAGNN vs FFNN vs Linear)', fontsize=24)

    plt.subplot(211)
    for model_dict in model_dicts:
        plt.plot(model_dict['train_history'], label=model_dict['name'])
    # plt.plot(ffnn['train_history'], label='FFNN (lr: %f)' % ffnn_lr)
    plt.legend()
    plt.title('train loss')

    plt.subplot(212)
    for model_dict in model_dicts:
        plt.plot(model_dict['test_history'], label=model_dict['name'])
    # plt.plot(ffnn['test_history'], label='FFNN')
    plt.legend()
    plt.title('test accuracy')

    """ ARCH TOO BIG
    plt.subplot(212)
    print('creating graph...')
    pagnn = model_dicts[-2] # get the last pagnn network defined
    draw_networkx_graph(pagnn['model'], mode='scaled_weights')
    plt.title('%s architecture' % pagnn['name'])
    """

    plt.savefig('examples/figures/mnist.png', transparent=True)
    plt.show()
