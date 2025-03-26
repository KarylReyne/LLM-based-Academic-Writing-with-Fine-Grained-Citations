import os
import torch as th

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.lightning_hdf5 import HDF5DataModule
from model.vit import VisionTransformer
from model.lightning.trainer import TrainerModule
from utils.configuration import Configuration
from utils.io import PeriodicCheckpoint

def train(cfg: Configuration, checkpoint_path):
    
    os.makedirs(f"out/{cfg.model_path}", exist_ok=True)

    model = None
    data_module = None
    
    if cfg.model.data == 'hdf5':
        model = VisionTransformer(cfg = cfg.model)
        data_module = HDF5DataModule(cfg)
    else:
        raise NotImplementedError(f"Data {cfg.model.data} not implemented")

    trainer_module = TrainerModule(model, cfg)

    # Load the model from the checkpoint if provided, otherwise create a new model
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model = TrainerModule.load_from_checkpoint(checkpoint_path, cfg=cfg, model=model)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=f"out/{cfg.model_path}",
        filename="Model-{epoch:02d}-{loss:.2f}",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    periodic_checkpoint_callback = PeriodicCheckpoint(
        save_path=f"out/{cfg.model_path}",
        save_every_n_steps=1000,  # Save checkpoint every 1000 global steps
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=cfg.model.gradient_accumulation_steps,
        accelerator = 'cuda' if th.cuda.is_available() else 'cpu',
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        precision=16 if cfg.model.mixed_precision else 32,
        enable_progress_bar=False,
        logger=False
    )

    if cfg.validate:
        trainer.validate(trainer_module, data_module)
    else:
        trainer.fit(trainer_module, data_module)

