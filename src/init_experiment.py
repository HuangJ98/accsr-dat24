import os
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from src.model import Model
from src.utils import exp_folder, load_subconfigs, create_datafiles


def init_exp(args):
    """Initialize the config, trainer and model."""

    config = load_subconfigs(OmegaConf.load(args.config))
    config.trainer.update(
        {
            "devices": int(args.devices),
            "nodes": int(os.environ["SLURM_NNODES"])
            if "SLURM_NNODES" in os.environ
            else 1,
        }
    )
    config.data.config.batch_size = int(
        config.data.config.batch_size * config.trainer.devices
    )
    config.data.config.asr_repo = config.asr.repo
    config.commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        num_nodes=config.trainer.nodes,
        max_epochs=config.trainer.max_epochs,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        logger=TensorBoardLogger("logs", name=exp_folder(config)),
        num_sanity_val_steps=0,
        callbacks=ModelCheckpoint(save_last=True),
    )

    config.data = create_datafiles(
        config.data, config.ensemble.action, trainer.logger.log_dir,
    )
    model = Model(config)

    OmegaConf.save(config, os.path.join(trainer.logger.log_dir, "config.yaml"))

    return config, trainer, model
