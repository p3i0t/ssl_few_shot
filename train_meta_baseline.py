import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from resnet_wider import resnet50x1, resnet50x2, resnet50x4
from get_tasks import get_few_shot_tasksets


class FewShotLearner(pl.LightningModule):
    def __init__(self, backbone='resnet50x1', root='data', dataset='cifar-fc100', train_mode='train_val',
                 n_ways=5, n_shots=1, n_queries=5, n_train_tasks=3000, n_test_tasks=1000):
        super().__init__()

        assert dataset in ['cifar-fc100', 'cifar-fs', 'mini-imagenet', 'tiered-imagenet']

        self.backbone = eval(backbone)()
        self.proj_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove final fully connected layer

        checkpoint_path = '{}-{}-{}.pt'.format(backbone, dataset, train_mode)

        state = torch.load(checkpoint_path, map_location='cpu')
        self.backbone.load_state_dict(state, strict=False)  # last layer removed, strict=False

        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries

        self.n_train_tasks = n_train_tasks
        self.n_test_tasks = n_test_tasks

        self.tasksets = get_few_shot_tasksets(
            root=root,
            dataset=dataset,
            train_ways=self.n_ways,
            test_ways=self.n_ways,
            train_samples=self.n_shots + self.n_queries,
            test_samples=self.n_shots + 1,
            n_train_tasks=self.n_train_tasks,
            n_test_tasks=self.n_test_tasks
        )

    def forward(self, x):
        b, prod, c, h, w = x.size()  # prod = n_way * n_aug
        return F.normalize(self.backbone(x.view(-1, c, h, w)), dim=-1)

    def _batch_forward(self, batch, train=True):
        x, _ = batch  # there is problem with y, so discard and construct ourselves.
        b, way_shot_query, c, h, w = x.size()

        n_queries = self.n_queries if train else 1
        x_ = x.view(b, self.n_ways, (self.n_shots + n_queries), c, h, w).contiguous()

        # construct y
        lbls = torch.arange(self.n_ways).to(x.device).view(1, self.n_ways, 1).contiguous()
        y_ = lbls.repeat(b, 1, (self.n_shots + n_queries)).contiguous()

        x_support, x_queries = torch.split_with_sizes(x_, split_sizes=[self.n_shots, n_queries], dim=2)
        y_support, y_queries = torch.split_with_sizes(y_, split_sizes=[self.n_shots, n_queries], dim=2)

        rep_s = self(x_support.contiguous().view(b, self.n_ways * self.n_shots, c, h, w))
        rep_q = self(x_queries.contiguous().view(b, self.n_ways * n_queries, c, h, w))

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.tasksets.train,
            batch_size=20,
            drop_last=False,
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        loss, acc = self._batch_forward(batch)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        logs = {'avg_loss': avg_loss, 'avg_acc': avg_acc}

        return {'avg_loss': avg_loss, 'avg_acc': avg_acc, 'log': logs, 'progress_bar': logs}

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.tasksets.test,
            batch_size=20,
            drop_last=False,
        )
        return test_loader

    def test_step(self,  batch, batch_idx):
        loss, acc = self._batch_forward(batch, train=False)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_acc': avg_acc}

        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'log': logs, 'progress_bar': logs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Meta-Baseline")
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar-fs',
                        help='one of [cifar-fs, cifar-fc100, mini-imagenet, tiered-imagenet]')
    parser.add_argument('--backbone', type=str, default='resnet50x1', help='name of backbone')
    parser.add_argument('--train_mode', type=str, default='train_val', help='whether use valid in training')
    parser.add_argument('--gpus', type=int, default=2, help='gpu device id')
    parser.add_argument('--n_ways', type=int, default=5, help='n_ways')
    parser.add_argument('--n_shots', type=int, default=1, help='n_shots')
    parser.add_argument('--n_queries', type=int, default=5, help='n_queries')
    parser.add_argument('--n_train_tasks', type=int, default=3000, help='n_train_tasks')
    parser.add_argument('--n_test_tasks', type=int, default=1000, help='n_train_tasks')
    parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
    args = parser.parse_args()

    fewshot_learner = FewShotLearner(
        backbone=args.backbone,
        dataset=args.dataset,
        root=args.data,
        train_mode=args.train_mode,
        n_ways=args.n_ways,
        n_shots=args.n_shots,
        n_queries=args.n_queries,
        n_train_tasks=args.n_train_tasks,
        n_test_tasks=args.n_test_tasks
    )
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        distributed_backend='ddp',
        precision=16,
        weights_summary=None,
    )
    trainer.fit(fewshot_learner)
    trainer.test()
