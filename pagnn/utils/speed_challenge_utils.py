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


class PConv(torch.nn.Module):
    def __init__(self, num_classes=1, extra_p_neurons=5, retain_state=True, sequence_inputs=True, p_activation=F.relu, p_steps=1):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3)
        self.max_pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3)
        self.max_pool2 = torch.nn.MaxPool2d(3)
        # self.pagnn = PAGNNLayer(18, num_classes, extra_p_neurons, retain_state=retain_state, sequence_inputs=sequence_inputs, activation=p_activation, steps=p_steps)
        self.pagnn = PAGNNLayer(18, num_classes, extra_p_neurons, retain_state=retain_state, activation=p_activation, steps=p_steps)


    def forward(self, x, force_no_reset=False):
        update_view = False
        if len(x.shape) > 4:
            update_view = True
            pagnn_batch_size = x.shape[0]
            pagnn_seq_length = x.shape[1]
            x = x.view((-1, *x.shape[2:]))

        x = self.conv1(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)

        if update_view:
            x = x.view(pagnn_batch_size, pagnn_seq_length, -1)

        # x = self.pagnn(x, force_no_reset=force_no_reset)
        x = self.pagnn(x, sequence=True)
        return x


class SpeedChallenge(torch.utils.data.Dataset):
    urls = [
        'https://github.com/commaai/speedchallenge/raw/master/data/train.mp4',
        'https://github.com/commaai/speedchallenge/raw/master/data/test.mp4',
        'https://raw.githubusercontent.com/commaai/speedchallenge/master/data/train.txt'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'speed_challenge_train.pt'
    test_file = 'speed_challenge_test.pt'

    def __init__(self, root, set_type='train', split=0.8, download=True, transform=None, sequence_length=10, sequence_start_spacing=1):
        self.root = root
        self.set_type = set_type
        self.split = split
        self.reader = None
        self.sequence_length = sequence_length
        self.sequence_start_spacing = sequence_start_spacing
        self.transform = transform
        
        if download:
            self.download()

        file_path = os.path.join(self.root, self.raw_folder, 'test.mp4' if set_type == 'test' else 'train.mp4')
        self.reader = imageio.get_reader(file_path,  'ffmpeg')
        with open(os.path.join(self.root, self.raw_folder, 'train.txt'), 'r') as speeds:
            speeds = speeds.readlines()
        self.labels = torch.zeros(len(speeds), dtype=torch.float32)
        for i, speed in enumerate(speeds):
            self.labels[i] = float(speed.replace('\n', ''))

        N_training = int(self.split*len(self.labels))
        if self.set_type == 'train':
            self.labels = self.labels[:N_training]
        elif self.set_type == 'eval':
            self.labels = self.labels[N_training:]
        elif self.set_type == 'test':
            self.labels = None
            """
            print('counting frames for test set...')
            for num, _ in enumerate(self.reader):
                pass

            self.length = num + 1
            print('test set length:', self.length)
            """
            self.length = 10798
        else:
            raise Exception()

        """
        print('counting frames...')
        for num, _ in enumerate(self.reader):
            pass

        self.length = num + 1
        """


    def _check_exists(self):
        if self.set_type == 'test':
            return os.path.exists(os.path.join(self.root, self.raw_folder, 'test.mp4'))
        return os.path.exists(os.path.join(self.root, self.raw_folder, 'train.mp4'))
    

    def download(self):
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            # data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            if not os.path.exists(file_path):
            # with open(file_path, 'wb') as f:
            #     f.write(data.read())
            # with open(file_path.replace('.gz', ''), 'wb') as out_f, \
            #         gzip.GzipFile(file_path) as zip_f:
            #     out_f.write(zip_f.read())
            # os.unlink(file_path)
                wget.download(url, file_path)

    def __len__(self):
        l = len(self.labels) if self.labels is not None else self.length
        return l // self.sequence_start_spacing - self.sequence_length + 1


    def __getitem__(self, index):
        index *= self.sequence_start_spacing

        sample1 = self.reader.get_data(index)
        if self.labels is not None:
            out_labels = torch.zeros(self.sequence_length)
            out_labels[0] = self.labels[index]

        if self.transform is not None:
            sample1 = self.transform(sample1)
        out = torch.zeros((self.sequence_length, *sample1.shape))
        out[0] = sample1

        for sample_n in range(1, self.sequence_length):
            sample = self.reader.get_data(index + sample_n)
            if self.transform is not None:
                sample = self.transform(sample)
            out[sample_n] = sample

            if self.labels is not None:
                out_labels[sample_n] = self.labels[index+sample_n]

        if self.labels is not None:
            return out, out_labels
        return out


def get_args():
    parser = argparse.ArgumentParser(description='Comma AI Speed Challenge')
    # parser.add_argument('--arch', default='p_resnet50', type=str, help='architecture (ex. p_resnet50)')
    parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--sequence-length', default=10, type=int, help='number of reference frames for a given sample')
    parser.add_argument('--sequence-spacing', default=5, type=int, help='number to space the start of sequences by')
    parser.add_argument('--image-resize', default=None, type=int, help='number passed into the Resize transform')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate for SGD')
    parser.add_argument('--workers', default=0, type=int, help='dataloader workers')
    parser.add_argument('--batch-size', default=1, type=int, help='training batch size')
    parser.add_argument('--pagnn-extra-neurons', default=15, type=int, help='number of extra neurons to allocate for PAGNNs')
    parser.add_argument('--pagnn-steps', default=1, type=int, help='number of steps for PAGNN to take every sequence input')
    parser.add_argument('--tqdm', action='store_true', help='use tqdm for loops')
    parser.add_argument('--print-freq', default=5, type=int, help='frequency to print')

    """ pruning args """
    parser.add_argument('--dense-allocation', default=None, type=float, help='dense allocation for RigL')
    parser.add_argument('--delta', default=20, type=int, help='rigl delta param')

    args = parser.parse_args()
    print(args)
    print()

    return args

def get_loaders(args):
    raw = [transforms.ToPILImage()]
    if args.image_resize is not None:
        raw.append(transforms.Resize(args.image_resize))
    raw.append(transforms.ToTensor())
    tfs = transforms.Compose(raw)

    train_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, set_type='train', sequence_length=args.sequence_length, \
                               sequence_start_spacing=args.sequence_spacing)
    eval_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, set_type='eval', sequence_length=args.sequence_length, \
                              sequence_start_spacing=args.sequence_length) # eval we don't want to create new data
    test_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, set_type='test', sequence_length=args.sequence_length, \
                              sequence_start_spacing=args.sequence_length) # test we don't want to create new data

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    return train_loader, eval_loader, test_loader
