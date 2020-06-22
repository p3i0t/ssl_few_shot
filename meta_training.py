import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from resnet_wider import resnet50x1, resnet50x2, resnet50x4
from get_tasks import get_few_shot_tasksets
from tqdm import tqdm


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


class FewShotLearner(nn.Module):
    def __init__(self, backbone='resnet50x1', dataset='mini-imagenet', n_ways=5, n_shots=1, n_queries=5):
        super().__init__()
        assert dataset in ['cifar-fc100', 'cifar-fs', 'mini-imagenet']

        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.backbone = eval(args.backbone)()
        self.proj_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove final fully connected layer

        checkpoint_path = '{}-{}.pt'.format(backbone, dataset)

        state = torch.load(checkpoint_path, map_location='cpu')
        self.backbone.load_state_dict(state, strict=False)  # last layer removed, strict=False

    def x_forward(self, x):
        b, prod, c, h, w = x.size()  # prod = n_way * n_aug
        return F.normalize(self.backbone(x.view(-1, c, h, w)), dim=-1)  # normalized representation

    def forward(self, x, train=True):
        b, way_shot_query, c, h, w = x.size()

        n_queries = self.n_queries if train else 1
        x_ = x.view(b, self.n_ways, (self.n_shots + n_queries), c, h, w).contiguous()

        # construct y
        lbls = torch.arange(self.n_ways).to(x.device).view(1, self.n_ways, 1).contiguous()
        y_ = lbls.repeat(b, 1, (self.n_shots + n_queries)).contiguous()

        x_support, x_queries = torch.split_with_sizes(x_, split_sizes=[self.n_shots, n_queries], dim=2)
        y_support, y_queries = torch.split_with_sizes(y_, split_sizes=[self.n_shots, n_queries], dim=2)

        rep_s = self.x_forward(x_support.contiguous().view(b, self.n_ways * self.n_shots, c, h, w))
        rep_q = self.x_forward(x_queries.contiguous().view(b, self.n_ways * n_queries, c, h, w))

        q = rep_q.view(b, self.n_ways * n_queries, self.proj_dim)
        # centroid of same way/class
        s = rep_s.view(b, self.n_ways, self.n_shots, self.proj_dim).mean(dim=2)
        s = s.clone().permute(0, 2, 1).contiguous()

        cosine_scores = q @ s  # batch matrix multiplication
        logits = cosine_scores.view(-1, self.n_ways) / 0.1
        labels = y_queries.contiguous().view(-1)

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc


def run_worker(rank, args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)

    model = FewShotLearner(
        backbone=args.backbone,
        dataset=args.dataset,
        n_ways=args.n_ways,
        n_shots=args.n_shots,
        n_queries=args.n_queries
    )
    model = model.cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    tasksets = get_few_shot_tasksets(
        root=args.root,
        dataset=args.dataset,
        train_ways=args.n_ways,
        test_ways=args.n_ways,
        train_samples=args.n_shots + args.n_queries,
        test_samples=args.n_shots + 1,
        n_train_tasks=args.n_train_tasks,
        n_test_tasks=args.n_test_tasks
    )

    train_sampler = DistributedSampler(tasksets.train)
    train_loader = torch.utils.data.DataLoader(
        dataset=tasksets.train,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler
    )
    test_sampler = DistributedSampler(tasksets.test)
    test_loader = torch.utils.data.DataLoader(
        dataset=tasksets.test,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        sampler=test_sampler
    )

    cudnn.benchmark = True
    print("Meta-Training")
    run_epoch(model, train_loader, optimizer=optimizer, rank=rank)
    print("Test")
    run_epoch(model, test_loader, optimizer=None, rank=rank)


def run_epoch(model, data_loader, optimizer=None, rank=0):
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')

    if optimizer:
        model.train()
    else:
        model.eval()

    data_bar = tqdm(data_loader)
    for x in data_bar:
        x = x.cuda(rank, non_blocking=True)
        loss, acc = model(x)

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        data_bar.set_description("loss: {:.4f}, acc: {:.4f}".format(loss_meter.avg, acc_meter.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Meta-Baseline")
    parser.add_argument('--root', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='mini-imagenet',
                        help='one of [cifar-fs, cifar-fc100, mini-imagenet, tiered-imagenet]')
    parser.add_argument('--backbone', type=str, default='resnet50x1', help='name of backbone')
    parser.add_argument('--gpus', type=int, default=2, help='gpu device id')
    parser.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parser.add_argument('--n_ways', type=int, default=5, help='n_ways')
    parser.add_argument('--n_shots', type=int, default=1, help='n_shots')
    parser.add_argument('--n_queries', type=int, default=5, help='n_queries')
    parser.add_argument('--n_train_tasks', type=int, default=1000, help='n_train_tasks')
    parser.add_argument('--n_test_tasks', type=int, default=1000, help='n_train_tasks')
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    args.world_size = n_gpus
    mp.spawn(run_worker, nprocs=n_gpus, args=(args, ))

