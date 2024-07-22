import os
from argparse import ArgumentParser
import logging

from src.evaluate import evaluate
from src.init_experiment import init_exp


def main(args):

    config, trainer, model = init_exp(args)

    # create logger in experiment folder to log progress: dump to file and stdout
    logger = logging.getLogger("progress")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(trainer.logger.log_dir, "progress.log")
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    # run the experiment
    if "asr" not in model.action:
        if config.ac.binary is True:
            assert config.ac.n_accents == 2
        else:
            assert len(config.data.seen_accents) == config.ac.n_accents

    if model.action == "train_ac":
        model.asr.freeze()
        if config.ac.binary is True:
            trainer.fit(model)
        else:
            ac_dataloaders = model.val_dataloader()[: config.ac.n_accents]
            trainer.validate(model, dataloaders=ac_dataloaders)
            trainer.fit(model, val_dataloaders=ac_dataloaders)
        model.asr.unfreeze()

    elif "train" in model.action:
        trainer.validate(model)
        trainer.fit(model)

    elif model.action == "evaluate_asr":
        evaluate(model.asr, config, trainer.logger)

    elif model.action == "evaluate_ac":
        model.pretrain_ac = True
        if config.ac.binary is True:
            trainer.validate(model)
        else:
            ac_dataloaders = model.val_dataloader()[: config.ac.n_accents]
            trainer.validate(model, dataloaders=ac_dataloaders)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
