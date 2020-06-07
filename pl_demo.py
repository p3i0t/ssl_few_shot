import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from PIL import Image
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

import torch.backends.cudnn as cudnn

from datasets.cifar import MetaCIFAR100

import pytorch_lightning as pl


train_transform = transforms.Compose([
    lambda x: Image.fromarray(x),
    transforms.Resize(160, interpolation=Image.BILINEAR),
    transforms.RandomCrop(128),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    lambda x: Image.fromarray(x),
    transforms.Resize(160, interpolation=Image.BILINEAR),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])


class FewShotLearner(pl.LightningModule):
    def __init__(self, backbone='resnet50x1'):
        super().__init__()

        self.backbone = eval(backbone)()
        if backbone == 'resnet50x1':
            checkpoint_path = 'resnet50-1x.pth'
        elif backbone == 'resnet50x2':
            checkpoint_path = 'resnet50-2x.pth'
        elif backbone == 'resnet50x4':
            checkpoint_path = 'resnet50-4x.pth'

        state = torch.load(checkpoint_path, map_location='cpu')
        self.backbone.load_state_dict(state['state_dict'])

        self.proj_dim = self.backbone.fc.out_features

        self.n_ways = 5
        self.n_shots = 1
        self.n_queries = 6
        self.n_aug_support = 1

    def forward(self, x):
        b, prod, c, h, w = x.size()  # prod = n_way * n_aug
        return F.normalize(self.backbone(x.view(-1, c, h, w)), dim=-1)

    def training_step(self, batch, batch_idx):
        x_support, y_support, x_queries, y_queries = batch

        b, prod, c, h, w = x_support.size()  # prod = n_way * n_aug
        rep_s = self(x_support)
        rep_q = self(x_queries)

        q = rep_q.view(b, self.n_ways * self.n_queries, self.proj_dim)
        # centroid of same way/class
        s = rep_s.view(b, self.n_ways, self.n_shots * self.n_aug_support, self.proj_dim).mean(dim=2)
        s = s.clone().permute(0, 2, 1).contiguous()

        cosine_scores = q @ s  # batch matrix multiplication
        logits = cosine_scores.view(-1, self.n_ways) / 0.2
        labels = y_queries.view(-1)

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        train_set = MetaCIFAR100(
            root='data/CIFAR-FS',
            partition='train',
            train_transform=train_transform,
            test_transform=test_transform,
            n_ways=self.n_ways,
            n_shots=self.n_shots,
            n_queries=self.n_queries,
            n_aug_support_samples=self.n_aug_support
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=16,
            shuffle=True,
            drop_last=False,
        )
        return train_loader

    def test_dataloader(self):
        test_set = MetaCIFAR100(
            root='data/CIFAR-FS',
            partition='test',
            train_transform=train_transform,
            test_transform=test_transform,
            n_ways=self.n_ways,
            n_shots=self.n_shots,
            n_queries=1,
            n_aug_support_samples=1
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=16,
            shuffle=False,
            drop_last=False
        )
        return test_loader

    def test_step(self,  batch, batch_idx):
        x_support, y_support, x_queries, y_queries = batch

        b, prod, c, h, w = x_support.size()  # prod = n_way * n_aug
        rep_s = self(x_support)
        rep_q = self(x_queries)

        q = rep_q.view(b, self.n_ways * 1, self.proj_dim)
        # centroid of same way/class
        s = rep_s.view(b, self.n_ways, self.n_shots * 1, self.proj_dim).mean(dim=2)
        s = s.clone().permute(0, 2, 1).contiguous()

        cosine_scores = q @ s  # batch matrix multiplication
        logits = cosine_scores.view(-1, self.n_ways) / 0.2
        labels = y_queries.view(-1)

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {'test_loss': loss, 'test_acc': acc}


if __name__ == '__main__':
    fewshot_learner = FewShotLearner()
    trainer = pl.Trainer(
        gpus=2,
        max_epochs=1,
        distributed_backend='ddp',
        precision=32,
    )
    trainer.fit(fewshot_learner)
    trainer.test()
