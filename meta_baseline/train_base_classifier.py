import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl
from resnet_wider import resnet50x1, resnet50x2, resnet50x4
from get_tasks import get_normal_tasksets


class BaseClassifierLearner(pl.LightningModule):
    def __init__(self, backbone='resnet50x1', root='data', dataset='cifar-fc100', train_mode='train_val'):
        super().__init__()
        assert dataset in ['cifar-fc100', 'cifar-fs', 'mini-imagenet', 'tiered-imagenet']
        self.train_set, self.valid_set, self.test_set = get_normal_tasksets(root, dataset)

        if train_mode == 'train_val':
            self.n_classes = len(set(self.train_set.targets)) + len(set(self.valid_set.targets))
            self.train_set = ConcatDataset([self.train_set, self.valid_set])
        elif train_mode == 'train':
            self.n_classes = len(set(self.train_set.targets))
        else:
            raise Exception('train mode not available.')

        self.backbone = eval(backbone)()
        if backbone == 'resnet50x1':
            checkpoint_path = 'resnet50-1x.pth'
        elif backbone == 'resnet50x2':
            checkpoint_path = 'resnet50-2x.pth'
        elif backbone == 'resnet50x4':
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
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        logs = {'avg_loss': avg_loss, 'avg_acc': avg_acc}

        return {'avg_loss': avg_loss, 'avg_acc': avg_acc, 'log': logs, 'progress_bar': logs}

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_set,
            batch_size=100,
            drop_last=False,
        )
        return test_loader

    def test_step(self,  batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {'test_loss': loss, 'test_acc': acc}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_acc': avg_acc}

        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'log': logs, 'progress_bar': logs}


@hydra.main(config_path='base_config.yaml')
def train(args: DictConfig) -> None:
    base_classifier = BaseClassifierLearner(backbone=args.backbone, dataset=args.dataset)
    trainer = pl.Trainer(
        gpus=2,
        max_epochs=args.epochs,
        distributed_backend='ddp',
        precision=16,
        weights_summary=None,  # close model summary

    )
    trainer.fit(base_classifier)
    trainer.save_checkpoint('{}-{}-{}.pt'.format(args.backbone, args.dataset, args.train_mode))
    trainer.test()


if __name__ == '__main__':
    train()
