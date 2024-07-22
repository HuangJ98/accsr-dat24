import optuna
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


from src.model import Model
from src.utils import exp_folder, load_subconfigs, create_datafiles


def objective(trial):
    ac_lr = trial.suggest_float("lr", 0.001, 1)
    config = load_subconfigs(OmegaConf.load("config/experiment.yaml"))
    config.ensemble.mode = "train"
    config.asr.ckpt = None
    config.optim.ac_args.lr = ac_lr
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        limit_val_batches=0.3,
        logger=TensorBoardLogger("logs", name=exp_folder(config)),
        num_sanity_val_steps=0,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_wer")],
    )
    config.data = create_datafiles(
        config.data,
        config.ensemble.action,
        trainer.logger.log_dir,
    )
    model = Model(config)
    hyperparameters = dict(ac_lr=ac_lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)
    return trainer.callback_metrics["val_wer"].item()


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize", pruner=pruner, study_name="ensemble"
    )
    study.optimize(objective, n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
