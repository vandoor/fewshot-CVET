import pickle
import numpy as np
import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
IMAGE_PATH1 = osp.join(ROOT_PATH2, 'cifar-10-batches-py')

print(IMAGE_PATH1)


def identity(x):
    return x


class Cifar_10(Dataset):
    def __init__(self, setname, args, augment=False):
        im_size = args.orig_imsize
        self.raw_data = []
        self.labels = []
        self.data = []
        self.process_data(setname)

        self.num_class = len(set(self.labels))

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ]

        if args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list +
                [
                    transforms.Normalize(np.array([x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                         np.array([x/255.0 for x in [70.68188272,   68.27635443,  72.54505529]])),

                ]
            )
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def process_data(self, setname):
        if setname == 'train':
            file_names = [f'data_batch_{i}' for i in range(1, 6)]
            datas = []
            labels = []
            for i,file_name in enumerate(file_names):
                Data_dir = osp.join(IMAGE_PATH1, file_name)
                with open(Data_dir, 'rb') as f:
                    raw_data = pickle.load(f, encoding='latin1')
                    labels.append(np.array(raw_data['labels']))
                    datas.append(raw_data['data'].reshape(-1, 3, 32, 32).astype('float'))
            self.labels = np.concatenate(labels).tolist()
            self.data = np.concatenate(datas)
        elif setname == 'test' or setname == 'val':
            file_names = ['test_batch']
            datas = []
            labels = []
            for i, file_name in enumerate(file_names):
                Data_dir = osp.join(IMAGE_PATH1, file_name)
                with open(Data_dir, 'rb') as f:
                    raw_data = pickle.load(f, encoding='latin1')
                    labels.append(np.array(raw_data['labels']))
                    datas.append(raw_data['data'].reshape(-1, 3, 32, 32).astype('float'))
            self.labels = np.concatenate(labels).tolist()
            self.data = np.concatenate(datas)

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def __getitem__(self, i):
        dat, label = self.data[i], self.labels[i]
        dat = dat.transpose(1, 2, 0)
        dat = np.array(dat, dtype=np.uint8)
        dat = Image.fromarray(dat)

        img1 = self.transform(dat)
        img2 = self.transform(dat)

        return img1, img2, label


