from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFAR100(Dataset):
    """support FC100 and CIFAR-FS"""
    def __init__(self, root='data',
                 partition='train',
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = root
        self.partition = partition
        self.transform = transform

        self.file_pattern = '%s.pickle'
        self.data = {}

        with open(os.path.join(self.data_root, '%s.pickle' % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            labels = data['labels']
            # adjust sparse labels to labels from 0 to n.
            cur_class = 0
            label2label = {}
            for idx, label in enumerate(labels):
                if label not in label2label:
                    label2label[label] = cur_class
                    cur_class += 1
            new_labels = []
            for idx, label in enumerate(labels):
                new_labels.append(label2label[label])
            self.labels = new_labels

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        if self.transform:
            img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        return img, target, item

    def __len__(self):
        return len(self.labels)


class MetaCIFAR100(CIFAR100):
    def __init__(self,
                 root='data',
                 partition='train',
                 train_transform=None,
                 test_transform=None,
                 n_ways=5,
                 n_shots=1,
                 n_queries=15,
                 n_aug_support_samples=5,
                 n_test_runs=1000,
                 fix_seed=True):
        """
        Dataset for meta learning.
        :param root: data directory
        :param partition: data partition
        :param train_transform: transform for training data.
        :param test_transform: transform for testing data.
        :param n_ways: number of classes in episodes of evaluation.
        :param n_shots: number of samples for each class in episodes of evaluation.
        :param n_queries: number of queries in each episode.
        :param n_aug_support_samples:  number of support samples augmented.
        :param n_test_runs: number of episodes for the total evaluation.
        :param fix_seed: whether fix the seed for reproduction.
        """
        super().__init__(root, partition)
        self.fix_seed = fix_seed
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = n_test_runs
        self.n_aug_support_samples = n_aug_support_samples

        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            samples = imgs[support_xs_ids_sampled]
            # multiply n_aug_support_samples
            support_xs.append(np.vstack([samples] * self.n_aug_support_samples))
            support_ys.append([idx] * (self.n_shots * self.n_aug_support_samples))

            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])

        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)

        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        support_ys = support_ys.reshape((self.n_ways * self.n_shots * self.n_aug_support_samples,))

        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 2
    args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = '../data/CIFAR-FS'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = CIFAR100(args.data_root, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaCIFAR100(args.data_root, 'train', n_shots=2)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[1])
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
    print(metaimagenet.__getitem__(500)[3])