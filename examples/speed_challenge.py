import torch
from torchvision import transforms
import numpy as np
import os
import wget
import errno
# import cv2
import tqdm
import imageio
import argparse
from pagnn import p_resnet


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

    def __init__(self, root, train=True, split=0.8, download=True, transform=None, sequence_length=10, sequence_start_spacing=1):
        self.root = root
        self.is_train = train
        self.split = split
        self.reader = None
        self.sequence_length = sequence_length
        self.sequence_start_spacing = sequence_start_spacing
        self.transform = transform
        
        if download:
            self.download()

        train_path = os.path.join(self.root, self.raw_folder, 'train.mp4')
        self.reader = imageio.get_reader(train_path,  'ffmpeg')
        with open(os.path.join(self.root, self.raw_folder, 'train.txt'), 'r') as speeds:
            speeds = speeds.readlines()
        self.labels = torch.zeros(len(speeds), dtype=torch.float32)
        for i, speed in enumerate(speeds):
            self.labels[i] = float(speed.replace('\n', ''))

        N_training = int(self.split*len(self.labels))
        if self.is_train:
            self.labels = self.labels[:N_training]
        else:
            self.labels = self.labels[N_training:]

        """
        print('counting frames...')
        for num, _ in enumerate(self.reader):
            pass

        self.length = num + 1
        """


    def _check_exists(self):
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
        return len(self.labels) // self.sequence_start_spacing - self.sequence_length


    def __getitem__(self, index):
        index *= self.sequence_start_spacing

        sample1 = self.reader.get_data(index)
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
            out_labels[sample_n] = self.labels[index+sample_n]

        return out, out_labels
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comma AI Speed Challenge')
    parser.add_argument('--arch', default='p_resnet50', type=str, help='architecture (ex. p_resnet50)')
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

    args = parser.parse_args()
    print(args)
    print()

    tfs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.image_resize),
        transforms.ToTensor(),
    ])

    train_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, train=True, sequence_length=args.sequence_length, sequence_start_spacing=args.sequence_spacing)
    test_set = SpeedChallenge('datasets/speed_challenge', transform=tfs, train=False, sequence_length=args.sequence_length, sequence_start_spacing=args.sequence_spacing)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    criterion = torch.nn.functional.mse_loss

    max_speed = torch.max(train_set.labels)
    min_speed = torch.min(train_set.labels)
    print('max speed', max_speed, 'min speed', min_speed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = p_resnet.__dict__[args.arch](num_classes=1, retain_state=True, sequence_inputs=True, pagnn_activation=torch.nn.functional.relu, pagnn_extra_neurons=args.pagnn_extra_neurons, pagnn_steps=args.pagnn_steps).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def random_guess():
        return torch.tensor(np.random.uniform(size=T.shape, low=min_speed, high=max_speed))

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
            optimizer.step()

            t.set_description('[TRAIN] loss: %f' % (total_loss/(i + 1)))

            if not args.tqdm and (i+1) % args.print_freq  == 0:
                print('[TRAIN %i/%i] loss: %f' % (i, len(train_loader), total_loss / (i + 1)))

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # print('[TRAIN Epoch %i] average loss' % epoch, avg_loss)
        return avg_loss
        

    @torch.no_grad()
    def evaluate():
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (X, T) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), disable=not args.tqdm):
                X, T = X.to(device), T.to(device)
                # Y = random_guess()
                Y = model(X)
                assert Y.shape == T.shape
                loss = criterion(Y, T)
                total_loss += loss
                if not args.tqdm and (i+1) % args.print_freq  == 0:
                    print('[EVAL %i/%i] loss: %f' % (i, len(test_loader), total_loss.item() / (i + 1)))
        avg_loss = (total_loss / len(test_loader)).item()
        # print('[EVAL Epoch %i] average loss' % epoch, avg_loss)
        return avg_loss

    for epoch in range(args.epochs):
        train_loss = train()
        eval_loss = evaluate()
        print('epoch\ttrain loss\teval loss')
        print('%i\t%f\t%f' % (epoch, train_loss, eval_loss))

