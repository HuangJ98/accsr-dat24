import os
from argparse import ArgumentParser
from shutil import rmtree

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from omegaconf import OmegaConf

def main(args):
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator=args.accelerator,
        num_nodes=args.devices,
        max_epochs=2,
    )

    config = OmegaConf.load(args.config)
    model = EncDecRNNTBPEModel.from_pretrained("stt_en_conformer_transducer_large")
    train_dl = model._setup_dataloader_from_config(config.train)
    test_dl = model._setup_dataloader_from_config(config.test)
    model.setup_optimization(config.optim)
    trainer.fit(model, train_dl, test_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    args = parser.parse_args()
    main(args)
