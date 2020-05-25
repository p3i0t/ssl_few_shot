import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

from datasets.cifar import MetaCIFAR100
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
    n_proj = 2048
    # # 2-layer MLP projector
    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 2048),
    #     nn.ReLU(),
    #     nn.Linear(2048, n_proj)
    # )
    model.fc = nn.Identity()

    model = model.cuda()
    cudnn.benchmark = True

    trans = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    n_ways = 5
    n_shots = 1
    n_queries = 3
    n_aug_support = 2
    train_set = MetaCIFAR100(
        root='data/CIFAR-FS',
        partition='train',
        train_transform=trans,
        test_transform=trans,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=n_queries,
        n_aug_support_samples=n_aug_support
    )

    test_set = MetaCIFAR100(
        root='data/CIFAR-FS',
        partition='test',
        train_transform=trans,
        test_transform=trans,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=1,
        n_aug_support_samples=1
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=3,
        shuffle=False,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=3,
        shuffle=False,
        drop_last=False,
    )
    # # print(dataset[0][0].size())
    # # print(dataset[0][1].shape)
    # # print(dataset[0][1])
    # # print(dataset[0][2].size())
    # # print(dataset[0][3].shape)
    # # print(dataset[0][3])
    #
    # b = next(iter(meta_loader))
    # for e in b:
    #     print(type(e), e.size())
    # exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        model.train()
        loss_meter = AverageMeter("ce_loss")
        acc_meter = AverageMeter("acc_loss")
        train_bar = tqdm(train_loader)
        for x_support, y_support, x_queries, y_queries in train_bar:
            x_s = x_support.cuda()
            y_s = y_support.cuda()
            x_q = x_queries.cuda()
            y_q = y_queries.cuda()

            b, prod, c, h, w = x_s.size()  # prod = n_way * n_aug
            rep_s = F.normalize(model(x_s.view(-1, c, h, w)), dim=-1)
            rep_q = F.normalize(model(x_q.view(-1, c, h, w)), dim=-1)

            q = rep_q.view(b, n_ways * n_queries, n_proj)
            s = rep_s.view(b, n_ways, n_shots * n_aug_support, n_proj).mean(dim=2)  # centroid of same way/class
            s = s.permute(0, 2, 1).contiguous()

            cosine_scores = q@s  # batch matrix multiplication
            logits = cosine_scores.view(-1, n_ways)
            labels = y_q.view(-1)

            loss = F.cross_entropy(logits, labels)
            acc = (logits.argmax(dim=1) == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), logits.size(0))
            acc_meter.update(acc.item(), logits.size(0))

            train_bar.set_description("Epoch{}, Meta-training loss: {:.4f}, acc: {}".format(epoch, loss_meter.avg, acc_meter.avg))

        model.eval()
        loss_meter = AverageMeter("ce_loss")
        acc_meter = AverageMeter("acc_loss")
        test_bar = tqdm(test_loader)

        for x_support, y_support, x_queries, y_queries in test_bar:
            x_s = x_support.cuda()
            y_s = y_support.cuda()
            x_q = x_queries.cuda()
            y_q = y_queries.cuda()

            b, prod, c, h, w = x_s.size()  # prod = n_way * n_aug
            rep_s = F.normalize(model(x_s.view(-1, c, h, w)), dim=-1)
            rep_q = F.normalize(model(x_q.view(-1, c, h, w)), dim=-1)

            q = rep_q.view(b, n_ways, n_proj)
            s = rep_s.view(b, n_ways, n_shots, n_proj).mean(dim=2)  # centroid of same way/class
            s = s.permute(0, 2, 1).contiguous()

            cosine_scores = q @ s  # batch matrix multiplication
            logits = cosine_scores.view(-1, n_ways)
            labels = y_q.view(-1)

            loss = F.cross_entropy(logits, labels)
            acc = (logits.argmax(dim=1) == labels).float().mean()

            loss_meter.update(loss.item(), logits.size(0))
            acc_meter.update(acc.item(), logits.size(0))

            test_bar.set_description("Meta-test loss: {:.4f}, acc: {}".format(loss_meter.avg, acc_meter.avg))


if __name__ == '__main__':
    main()