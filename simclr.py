import hydra
from omegaconf import DictConfig
import logging

import random
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34
from torchvision import transforms

import learn2learn as l2l

from models import SimCLR
from tqdm import tqdm


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class CIFAR10Pair(CIFAR10):
#     """Generate mini-batche pairs on CIFAR10 training set."""
#     def __getitem__(self, idx):
#         img, target = self.data[idx], self.targets[idx]
#         img = Image.fromarray(img)  # .convert('RGB')
#         imgs = [self.transform(img), self.transform(img)]
#         return torch.stack(imgs), target  # stack a positive pair


class OmniglotPair(Dataset):
    # 1623 handwritten characters from 50 different alphabets, 20 samples for each character, in total 32460 samples.
    def __init__(self, root='data', transform=None, download=True):
        self.dataset = l2l.vision.datasets.FullOmniglot(
            root=root,
            transform=None,
            download=download
        )
        self.transform = transform

    def __getitem__(self, i):
        img, label = self.dataset[i]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), label  # stack a positive pair

    def __len__(self):
        return len(self.dataset)


def nt_xent(x, t=0.5):
    # input x already normalized
    x_scores = (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

#
# # color distortion composed by color jittering and color dropping.
# # See Section A of SimCLR: https://arxiv.org/abs/2002.05709
# def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
#     # s is the strength of color distortion
#     color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
#     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
#     rnd_gray = transforms.RandomGrayscale(p=0.2)
#     color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
#     return color_distort


@hydra.main(config_path='simclr_config.yml')
def train(args: DictConfig) -> None:
    #assert torch.cuda.is_available()
    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        lambda img: (1.0 - img)
    ])
    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir

    omniglot_pair = OmniglotPair(
        root=data_dir,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        omniglot_pair,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    omniglot = l2l.vision.datasets.FullOmniglot(
        root=data_dir,
        transform=transform,
        download=True
    )

    dataset = l2l.data.MetaDataset(omniglot)
    classes = list(range(1623))
    random.shuffle(classes)

    n_way = 5
    n_shot = 1
    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=n_way,
                                             k=n_shot + 1,
                                             filter_labels=classes[1200:]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_tasks = l2l.data.TaskDataset(
        dataset,
        task_transforms=test_transforms,
        num_tasks=1000
    )

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim, in_channel=1).cuda()
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    # SimCLR training
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)

            optimizer.zero_grad()
            rep, projection = model(x)
            loss = nt_xent(projection, args.temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

        # save checkpoint very log_interval epochs
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
            torch.save(model.state_dict(), 'simclr_{}_epoch{}.pt'.format(args.backbone, epoch))

            model.eval()
            acc_meter = AverageMeter('acc')
            test_bar = tqdm(test_tasks)
            for task in test_bar:
                x, y = task
                x, y = x.cuda(), y.cuda()
                x_reps, _ = model(x)   # representations of input x

                # idx
                query_idx = torch.zeros_like(y)
                query_idx[::n_shot] = 1
                query_idx = query_idx.bool()

                # split
                x_query, y_query = x_reps[query_idx], y[query_idx]  # (n_way, proj_dim), (n_way*n_shot, proj_dim)
                x_support, y_query = x_reps[~query_idx], y[~query_idx]

                cosine_dist = (x_query @ x_support.t()).view(n_way, n_shot, n_way).sum(dim=1)
                pred = cosine_dist.argmax(dim=1)
                acc = (pred == y_query).float().mean()
                acc_meter.update(acc.item())
                test_bar.set_description("Few-shot test acc: {:.4f}".format(acc_meter.avg))


if __name__ == '__main__':
    train()


