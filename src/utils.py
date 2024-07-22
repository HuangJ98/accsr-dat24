import json
import os
import logging

import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator, CUDAAccelerator

from src.dataloader import setup_dataloader


LOGGER = logging.getLogger("progress")


def to_lang_dict(data, langs):
    """Convert the given data (array or tensor) to a dictionary
    with the given languages as keys."""
    if torch.is_tensor(data):
        data = data.tolist()
    if len(data) != len(langs):
        raise ValueError("data and langs must be the same size")
    return {lang: data[i] for i, lang in enumerate(langs)}


def setup_dataloaders(config, mode, concat=None):
    """
    Return a list one or more dataloaders, as defined in the config.

    - "mode" can be "train" or "test". "train" manifest files are those named
        "train_{lang}", where "lang" is any of the "seen" languages. Similarly,
        "test" manifest files are those named "test_{lang}, where "lang" is any of
        both seen and unseen languages".
    - If concat is an int, two dataloaders are created: one for the first "concat"
        files and one for the rest. If concat is None, a separate dataloader is
        created for each file.
    """

    dl_config = config.config
    dl_config.update(config[f"config_{mode}"])

    dataloaders = []
    if concat is not None:
        # create a single dataloader for all files
        if concat == 0:
            dl_config.manifest_filepath = config[f"{mode}_files"]
            return setup_dataloader(dl_config)
        # create 2 dataloaders, 1 for the first "concat" files and 1 for the rest
        else:
            dl_config.manifest_filepath = config[f"{mode}_files"][:concat]
            dataloaders.append(setup_dataloader(dl_config))
            dl_config.manifest_filepath = config[f"{mode}_files"][concat:]
            dataloaders.append(setup_dataloader(dl_config))

    # create a separate dataloader for each file
    else:
        for filepath in config[f"{mode}_files"]:
            dl_config.manifest_filepath = filepath
            dataloaders.append(setup_dataloader(dl_config))

    return dataloaders


def exp_folder(config):
    """Return the experiment folder given the config."""
    folder = config.language
    if config.job == "analysis":
        return os.path.join(folder, "analysis")
    elif config.job == "test":
        return os.path.join(folder, "tests", "exp_folder")
    action = config.ensemble.action
    if "_" in action:
        mode, model = action.split("_")
    else:
        model = "ensemble"
        mode = action
    folder = os.path.join(folder, model, mode)
    if model != "asr":
        ac_type = "binary" if config.ac.binary else "multi-class"
        folder += f"/{ac_type}/b{config.ensemble.branch}"
    if model == "ensemble":
        folder += f"/{config.ensemble.mode}"
    return folder


def create_datafiles(config, mode, log_dir):
    """
    Add the root folder to the paths of the audiofiles and store the resulting
    manifest files within the experiment folder.
    """
    
    test = [f"test_{acc}.txt" for acc in config.seen_accents + config.unseen_accents]
    if "train" in mode:
        train = [f"train_{acc}.txt" for acc in config.seen_accents]
    else:
        train = []

    for files in [train, test]:
        new_paths = list()
        for f in files:
            new_path = add_root(
                os.path.join(config.folder, f),
                config.root,
                log_dir,
                min_dur=config.min_dur,
                max_dur=config.max_dur,
            )
            new_paths.append(new_path)

        if len(new_paths) > 0:
            config[f"{f.split('_')[0]}_files"] = new_paths

    return config


def add_root(manifest_filepath, root_folder, log_dir, min_dur=None, max_dur=None):
    """
    Add the root folder to the paths of the audiofiles and store the resulting
    manifest files within the experiment folder.
    """

    LOGGER.info(f"Adding root `{root_folder}` to datafile `{manifest_filepath}`")

    # create new manifest file; if it already exists, skip
    new_path = os.path.join(log_dir, manifest_filepath)
    LOGGER.info(f"New datafile: `{new_path}`")
    if os.path.exists(new_path):  # manifest already modified
        LOGGER.warn(f"New filepath `{new_path}` already exists")
        return new_path

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    too_short, too_long, included = list(), list(), list()
    with open(new_path, "w") as writer:
        with open(manifest_filepath) as reader:
            for line in reader:
                obj = json.loads(line)
                dur = obj["duration"]
                if min_dur is not None and dur < min_dur:
                    too_short.append(dur)
                    continue
                if max_dur is not None and dur > max_dur:
                    too_long.append(dur)
                    continue
                included.append(dur)
                obj["audio_filepath"] = obj["audio_filepath"].replace(
                    "{root}", root_folder
                )
                writer.write(json.dumps(obj) + "\n")

        LOGGER.info(
            f"{len(included)} samples included ({round(sum(included) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_short)} samples too short ({round(sum(too_short) / 3600, 3 )} h)"
        )
        LOGGER.info(
            f"{len(too_long)} samples too long ({round(sum(too_long) / 3600, 3 )} h)"
        )

    return new_path


def load_subconfigs(config):
    """Given a config, load all the subconfigs that are specified in the config into
    the same level as the parameter. Configs are specified by parameters ending with
    '_file'. If a value of the config is a dict, call this function again recursively."""
    for key, value in config.items():
        if isinstance(value, DictConfig):
            config[key] = load_subconfigs(value)
        elif key.endswith("_file"):
            new_config = OmegaConf.load(value)
            for key, value in new_config.items():
                config[key] = value
    return config


def get_device(trainer):
    """Return the device used by the trainer."""
    if isinstance(trainer.accelerator, CPUAccelerator):
        return torch.device("cpu")
    elif isinstance(trainer.accelerator, GPUAccelerator) or isinstance(trainer.accelerator, CUDAAccelerator):
        return torch.device("cuda")
    else:
        raise RuntimeError("Unknown accelerator type")
