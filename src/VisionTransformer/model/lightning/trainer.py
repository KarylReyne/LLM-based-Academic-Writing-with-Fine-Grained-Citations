import pytorch_lightning as pl
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from utils.optimizers import Ranger
from utils.configuration import Configuration
from utils.io import UEMA, Timer
import torch.distributed as dist

class TrainerModule(pl.LightningModule):
    def __init__(self, model, cfg: Configuration, state_dict={}):
        super().__init__()
        self.cfg = cfg

        print(f"RANDOM SEED: {cfg.seed}")
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        self.net = model

        print(f"Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

        self.lr = self.cfg.learning_rate
        self.own_loggers = {}
        self.timer = Timer()

        self.num_updates = -1

    def forward(self, input):
        return self.net(input)

    def log(self, name, value, on_step=True, on_epoch=True, prog_bar=False, logger=True):
        super().log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger, sync_dist=True)

        if name not in self.own_loggers:
            self.own_loggers[name] = UEMA(1000)

        self.own_loggers[name].update(value.item() if isinstance(value, th.Tensor) else value)

    def training_step(self, batch, batch_idx):
        results = self(batch[0])

        loss = results["loss"] 

        self.log("loss", results['loss'])
        self.log("l1", results['l1'])
        self.log("ssim", results['ssim'])

        if self.num_updates < self.trainer.global_step:
            self.num_updates = self.trainer.global_step
            print("Epoch[{}|{}|{}|{:.2f}%]: {}, Loss: {:.2e}, L1: {:.2e}, SSIM: {:.2e}".format(
                self.trainer.local_rank,
                self.trainer.global_step,
                self.trainer.current_epoch,
                (batch_idx + 1) / len(self.trainer.train_dataloader) * 100,
                str(self.timer),
                float(self.own_loggers['loss']),
                float(self.own_loggers['l1']),
                float(self.own_loggers['ssim'])
            ), flush=True)

        return loss

    def configure_optimizers(self):
        return Ranger([
            {'params': self.net.parameters(), 'lr': self.cfg.learning_rate, 'weight_decay': self.cfg.weight_decay},
        ])

