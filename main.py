import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from resnet_wider import resnet50x1, resnet50x2, resnet50x4
from get_tasks import get_few_shot_tasksets


class FewShotLearner(pl.LightningModule):
    def __init__(self, backbone='resnet50x1', root='data', dataset='cifar10-fc100'):
        super().__init__()

        assert dataset in ['cifar10-fc100', 'cifar10-fs', 'mini-imagenet', 'tiered-imagenet']

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

        self.n_train_tasks = 2000
        self.n_test_tasks = 1000

        self.tasksets = get_few_shot_tasksets(
            root=root,
            dataset=dataset,
            train_ways=self.n_ways,
            test_ways=self.n_ways,
            train_samples=self.n_shots + self.n_queries,
            test_samples=self.n_shots + self.n_queries,
            n_train_tasks=self.n_train_tasks,
            n_test_tasks=self.n_test_tasks
        )

    def forward(self, x):
        b, prod, c, h, w = x.size()  # prod = n_way * n_aug
        return F.normalize(self.backbone(x.view(-1, c, h, w)), dim=-1)

    def _batch_forward(self, batch):
        x, y = batch
        b, way_shot_query, c, h, w = x.size()

        x_ = x.view(b, self.n_ways, (self.n_shots + self.n_queries), c, h, w).contiguous()
        y_ = y.view(b, self.n_ways, (self.n_shots + self.n_queries)).contiguous()

        x_support, x_queries = torch.split_with_sizes(x_, split_sizes=[self.n_shots, self.n_queries], dim=2)
        y_support, y_queries = torch.split_with_sizes(y_, split_sizes=[self.n_shots, self.n_queries], dim=2)

        rep_s = self(x_support.view(b, self.n_ways * self.n_shots, c, h, w).contiguous())
        rep_q = self(x_queries.view(b, self.n_ways * self.n_queries, c, h, w).contiguous())

        q = rep_q.view(b, self.n_ways * self.n_queries, self.proj_dim)
        # centroid of same way/class
        s = rep_s.view(b, self.n_ways, self.n_shots * self.n_aug_support, self.proj_dim).mean(dim=2)
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

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.tasksets.validation,
            batch_size=20,
            drop_last=False,
        )
        return val_loader

    def validation_step(self,  batch, batch_idx):
        loss, acc = self._batch_forward(batch)
        return {'val_loss': loss, 'val_acc': acc}

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.tasksets.test,
            batch_size=20,
            drop_last=False,
        )
        return test_loader

    def test_step(self,  batch, batch_idx):
        loss, acc = self._batch_forward(batch)
        return {'test_loss': loss, 'test_acc': acc}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_acc': avg_acc}

        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'log': logs, 'progress_bar': logs}


if __name__ == '__main__':
    fewshot_learner = FewShotLearner(backbone='resnet50x1')
    trainer = pl.Trainer(
        gpus=2,
        max_epochs=1,
        distributed_backend='ddp',
        precision=16,
    )
    trainer.fit(fewshot_learner)
    trainer.test()
