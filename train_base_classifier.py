import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl

from resnet_wider import resnet50x1, resnet50x2, resnet50x4
from _get_tasks import get_normal_tasksets

# number of train, valid, test sets in meta-dataset.
n_classes_dict = {
    'cifar-fc100': [64, 16, 20],
    'cifar-fs': [60, 20, 20],
    'mini-imagenet': [64, 16, 20],
    'tiered-imagenet': [351, 97, 160],
}


class BaseClassifierLearner(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        assert self.hparams.dataset in ['cifar-fc100', 'cifar-fs', 'mini-imagenet', 'tiered-imagenet']
        self.train_set, self.val_set, self.test_set = get_normal_tasksets(self.hparams.root, self.hparams.dataset)

        self.n_classes = n_classes_dict[self.hparams.dataset][0]  # n_classes of train
        # if self.hparams.train_mode == 'train_val':
        #     self.n_classes = sum(n_classes_dict[self.hparams.dataset][:2])  # n_classes of train and valid
        # elif self.hparams.train_mode == 'train':
        #     self.n_classes = n_classes_dict[self.hparams.dataset][0]  # n_classes of train
        # else:
        #     raise Exception('train mode not available.')

        self.backbone = eval(self.hparams.backbone)()
        if self.hparams.backbone == 'resnet50x1':
            checkpoint_path = 'resnet50-1x.pth'
        elif self.hparams.backbone == 'resnet50x2':
            checkpoint_path = 'resnet50-2x.pth'
        elif self.hparams.backbone == 'resnet50x4':
            checkpoint_path = 'resnet50-4x.pth'

        state = torch.load(checkpoint_path, map_location='cpu')
        self.backbone.load_state_dict(state['state_dict'])

        # replace last linear layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.n_classes)

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=64,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        return {'loss': avg_loss, 'acc': avg_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Meta-Baseline")
    parser.add_argument('--root', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar-fs',
                        help='one of [cifar-fs, cifar-fc100, mini-imagenet, tiered-imagenet]')
    parser.add_argument('--backbone', type=str, default='resnet50x1', help='name of backbone')
    parser.add_argument('--train_mode', type=str, default='train', help='whether use valid in training')
    parser.add_argument('--gpus', type=int, default=2, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
    args = parser.parse_args()

    base_classifier = BaseClassifierLearner(args)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        distributed_backend='ddp',
        precision=16,
        weights_summary=None,  # close model summary
    )
    trainer.fit(base_classifier)
    torch.save(
        base_classifier.backbone.state_dict(),
        '{}-{}.pt'.format(args.backbone, args.dataset)
    )
