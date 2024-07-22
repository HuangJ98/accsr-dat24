import importlib
import torch


def init_asr(config):
    """
    - Instantiate an ASR model with the config.
    - Load the checkpoint if specified.
    """

    module_str, cls_str = config.cls.rsplit(".", 1)
    module = importlib.import_module(module_str)
    cls = getattr(module, cls_str)
    model = cls.from_pretrained(config.model)

    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location=torch.device("cpu"))
        ckpt_dict = {
            k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("asr.")
        }
        model_dict = model.state_dict()
        model_dict.update(ckpt_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        model.cuda()

    return model
