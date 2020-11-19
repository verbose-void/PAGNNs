import torch
import os
import wget
import errno
# import cv2
import imageio


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

    def __init__(self, root, train=True, download=True):
        self.root = root
        self.is_train = train
        self.reader = None
        
        if download:
            self.download()

        train_path = os.path.join(self.root, self.raw_folder, 'train.mp4')
        self.reader = imageio.get_reader(train_path,  'ffmpeg')
        with open(os.path.join(self.root, self.raw_folder, 'train.txt'), 'r') as speeds:
            speeds = speeds.readlines()
        self.labels = torch.zeros(len(speeds), dtype=torch.float32)
        for i, speed in enumerate(speeds):
            self.labels[i] = float(speed.replace('\n', ''))

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
        return len(self.labels)


    def __getitem__(self, index):
        if not self.is_train:
            raise NotImplemented()

        return torch.tensor(self.reader.get_data(index)), self.labels[index]
            

if __name__ == '__main__':

    train_set = SpeedChallenge('datasets/speed_challenge')
    train_loader = torch.utils.data.DataLoader(train_set)

    for X, label in train_loader:
        print(X.shape, label)
        break
