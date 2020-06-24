from torchvision import transforms

import learn2learn as l2l
from PIL import Image

from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

from collections import namedtuple
BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'valid', 'test'))


def get_normal_tasksets(
        root='data',
        dataset='cifar10-fc100'
    ):
    if dataset == 'mini-imagenet':
        train_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            # transforms.Resize(160, interpolation=Image.BILINEAR),
            # transforms.RandomCrop(128),
            transforms.RandomResizedCrop(128),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            # transforms.Resize(160, interpolation=Image.BILINEAR),
            # transforms.CenterCrop(128),
            transforms.Resize(128),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(160, interpolation=Image.BILINEAR),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(160, interpolation=Image.BILINEAR),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

    if dataset == 'cifar-fs':
        train_dataset = l2l.vision.datasets.CIFARFS(
            root=root,
            transform=train_transform,
            mode='train'
        )

        valid_dataset = l2l.vision.datasets.CIFARFS(
            root=root,
            transform=train_transform,
            mode='validation'
        )

        test_dataset = l2l.vision.datasets.CIFARFS(
            root=root,
            transform=test_transform,
            mode='test'
        )
    elif dataset == 'cifar-fc100':
        train_dataset = l2l.vision.datasets.FC100(
            root=root,
            transform=train_transform,
            mode='train'
        )

        valid_dataset = l2l.vision.datasets.FC100(
            root=root,
            transform=train_transform,
            mode='validation'
        )

        test_dataset = l2l.vision.datasets.FC100(
            root=root,
            transform=test_transform,
            mode='test'
        )
    elif dataset == 'mini-imagenet':
        train_dataset = l2l.vision.datasets.MiniImagenet(
            root=root,
            transform=train_transform,
            mode='train'
        )

        valid_dataset = l2l.vision.datasets.MiniImagenet(
            root=root,
            transform=train_transform,
            mode='validation'
        )

        test_dataset = l2l.vision.datasets.MiniImagenet(
            root=root,
            transform=test_transform,
            mode='test'
        )
    elif dataset == 'tiered-imagenet':
        train_dataset = l2l.vision.datasets.TieredImagenet(
            root=root,
            transform=train_transform,
            mode='train',
            download=True
        )

        valid_dataset = l2l.vision.datasets.TieredImagenet(
            root=root,
            transform=train_transform,
            mode='validation',
            download=True
        )

        test_dataset = l2l.vision.datasets.TieredImagenet(
            root=root,
            transform=test_transform,
            mode='test',
            download=True
        )
    else:
        raise Exception("dataset {} not available.".format(dataset))

    return train_dataset, valid_dataset, test_dataset


def get_few_shot_tasksets(
        root='data',
        dataset='cifar10-fc100',
        train_ways=5,
        train_samples=10,
        test_ways=5,
        test_samples=10,
        n_train_tasks=2000,
        n_test_tasks=1000,
    ):
    """
    Fetch the train, valid, test meta tasks of given dataset.
    :param root: data directory.
    :param dataset: name of dataset.
    :param train_ways: number of ways of few-shot training.
    :param train_samples: number of each-way samples for a training task.
    :param test_ways: number of ways of few-shot evaluation and testing.
    :param test_samples: number of each-way samples for a valid or test task.
    :param n_train_tasks: total number of train tasks.
    :param n_test_tasks: total number of valid and test tasks.
    :return:
    """

    train_dataset, valid_dataset, test_dataset = get_normal_tasksets(root=root, dataset=dataset)
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]

    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=n_train_tasks,
    )

    valid_tasks = l2l.data.TaskDataset(
        dataset=valid_dataset,
        task_transforms=test_transforms,
        num_tasks=n_test_tasks,
    )

    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=n_test_tasks,
    )

    return BenchmarkTasksets(train_tasks, valid_tasks, test_tasks)


if __name__ == '__main__':
    tasks = get_few_shot_tasksets(dataset='cifar-fs')
    tasks = get_normal_tasksets(dataset='cifar-fs')
    # tasks = get_normal_tasksets(dataset='cifar-fc100')
    # batch = tasks.train.sample()
    # x, y = batch
    # print(x.size())
    # print(y.size())
    # print(y)
    # x, y = tasks.train[0]
    # print(x.size())
    # print(y.size())
    from torch.utils.data import ConcatDataset

    import torch.utils.data
    dataset = ConcatDataset([tasks[1], tasks[0]])
    # print(len(tasks[0]))
    # print(len(tasks[1]))
    # print(len(dataset))
    # loader = torch.utils.data.DataLoader(dataset, batch_size=100)
    # for x, y in loader:
    #     print(x.size(), y.size())
    dset = l2l.data.MetaDataset(dataset)
    tsk = l2l.data.TaskDataset(dset)
    batch = tsk.sample()
    print(batch)
