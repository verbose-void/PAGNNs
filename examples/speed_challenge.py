import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import wget
import errno
# import cv2
import tqdm
import imageio
import argparse

from pagnn import p_resnet, PAGNNLayer

from rigl_torch.RigL import RigLScheduler

from examples.speed_challenge_utils import SpeedChallenge, PConv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comma AI Speed Challenge')
    # parser.add_argument('--arch', default='p_resnet50', type=str, help='architecture (ex. p_resnet50)')
    parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--sequence-length', default=10, type=int, help='number of reference frames for a given sample')
    parser.add_argument('--sequence-spacing', default=1, type=int, help='number to space the start of sequences by')
    parser.add_argument('--image-resize', default=128, type=int, help='number passed into the Resize transform')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate for SGD')
    parser.add_argument('--workers', default=1, type=int, help='dataloader workers')
    parser.add_argument('--batch-size', default=1, type=int, help='training batch size')
    parser.add_argument('--pagnn-extra-neurons', default=300, type=int, help='number of extra neurons to allocate for PAGNNs')
    parser.add_argument('--pagnn-steps', default=3, type=int, help='number of steps for PAGNN to take every sequence input')
    parser.add_argument('--tqdm', action='store_true', help='use tqdm for loops')
    parser.add_argument('--print-freq', default=5, type=int, help='frequency to print')

    """ pruning args """
    parser.add_argument('--dense-allocation', default=None, type=float, help='dense allocation for RigL')
    parser.add_argument('--delta', default=20, type=int, help='rigl delta param')

    args = parser.parse_args()
    print(args)
    print()

    tfs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.image_resize),
        transforms.ToTensor(),
    ])

    train_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, set_type='train', sequence_length=args.sequence_length, sequence_start_spacing=args.sequence_spacing)
    eval_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, set_type='eval', sequence_length=args.sequence_length, sequence_start_spacing=args.sequence_spacing)
    test_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, set_type='test', sequence_length=args.sequence_length, sequence_start_spacing=args.sequence_spacing)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    criterion = torch.nn.functional.mse_loss

    max_speed = torch.max(train_set.labels)
    min_speed = torch.min(train_set.labels)
    print('max speed', max_speed, 'min speed', min_speed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = p_resnet.__dict__[args.arch](num_classes=1, retain_state=True, sequence_inputs=True, pagnn_activation=torch.nn.functional.relu, pagnn_extra_neurons=args.pagnn_extra_neurons, pagnn_steps=args.pagnn_steps).to(device)
    model = PConv().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def random_guess():
        return torch.tensor(np.random.uniform(size=T.shape, low=min_speed, high=max_speed))

    pruner = lambda: True
    if args.dense_allocation is not None:
        T_end = int(0.75 * len(train_loader))
        pruner = RigLScheduler(model, optimizer, dense_allocation=args.dense_allocation, T_end=T_end, delta=args.delta)

    def train():
        total_loss = 0
        model.train()
        t = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=not args.tqdm)
        for i, (X, T) in t:
            X, T = X.to(device), T.to(device)

            optimizer.zero_grad()

            Y = model(X)
            assert Y.shape == T.shape

            loss = criterion(Y, T)
            loss.backward()
        
            if pruner():
                optimizer.step()

            t.set_description('[TRAIN] loss: %f' % (total_loss/(i + 1)))

            if not args.tqdm and (i+1) % args.print_freq  == 0:
                print('[TRAIN %i/%i] loss: %f' % (i, len(train_loader), total_loss / (i + 1)))

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # print('[TRAIN Epoch %i] average loss' % epoch, avg_loss)
        return avg_loss
        

    @torch.no_grad()
    def evaluate(dataloader, save_results_file=None):
        total_loss = 0
        model.eval()
        output_predictions = []
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), disable=not args.tqdm):
                if type(data) == tuple:
                    X, T = data[0].to(device), data[1].to(device)
                    # Y = random_guess()
                    Y = model(X)
                    assert Y.shape == T.shape
                    loss = criterion(Y, T)
                    total_loss += loss
                    if not args.tqdm and (i+1) % args.print_freq  == 0:
                        print('[EVAL %i/%i] loss: %f' % (i, len(dataloader), total_loss.item() / (i + 1)))

                else:
                    X = data[0].to(device)
                    Y = model(X)
                    output_predictions.extend(Y)

        if len(output_predictions) > 0 and save_results_file is not None:
            raise NotImplemented()
            return

        avg_loss = (total_loss / len(dataloader)).item()
        # print('[EVAL Epoch %i] average loss' % epoch, avg_loss)
        return avg_loss

    """
    for epoch in range(args.epochs):
        train_loss = train()
        eval_loss = evaluate(eval_loader)
        print('epoch\ttrain loss\teval loss')
        print('%i\t%f\t%f' % (epoch, train_loss, eval_loss))
    """

    # run on test data and save output
    print('Finished training.... Running inference on output data')
    evaluate(test_loader, save_results_file='speed-challenge-outputs.txt')
    
    


