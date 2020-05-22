import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main():
    model = resnet50x1()
    sd = 'resnet50-1x.pth'
    sd = torch.load(sd, map_location='cpu')
    model.load_state_dict(sd['state_dict'])

    model = model.cuda()
    cudnn.benchmark = True

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Create Datasets
    test_dataset = l2l.vision.datasets.MiniImagenet(root='data', mode='test', transform=trans)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    n_way = 5
    n_shot = 1
    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(test_dataset,
                                             n=n_way,
                                             k=1 + n_shot),
        l2l.data.transforms.LoadData(test_dataset),
        l2l.data.transforms.RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=100)

    model.eval()
    acc_meter = AverageMeter('acc')
    test_bar = tqdm(test_tasks)
    for task in test_bar:
        x_ = task[0].cuda()
        logits = model(x_)  # representations of input x

        x_reps = F.normalize(logits, dim=1)
        # idx
        query_idx = torch.zeros(x_reps.size(0)).cuda()
        query_idx[::n_shot + 1] = 1
        query_idx = query_idx.bool()

        # split
        x_query = x_reps[query_idx]  # (n_way, proj_dim), (n_way*n_shot, proj_dim)
        x_support = x_reps[~query_idx]

        y_ = torch.arange(x_query.size(0)).to(x_query.device)

        cosine_dist = (x_query @ x_support.t()).view(n_way, n_way, n_shot).sum(dim=2)
        pred = cosine_dist.argmax(dim=1)
        acc = (pred == y_).float().mean()
        acc_meter.update(acc.item())
        test_bar.set_description("Few-shot test acc: {:.4f}".format(acc_meter.avg))


if __name__ == '__main__':
    main()