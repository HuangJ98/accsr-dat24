import logging

import torch

from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

from src.asr.whisper_dataset import WhisperDataset


LOGGER = logging.getLogger("progress")


def setup_dataloader(config):
    """
    Create a NeMo dataloader with speech labels.
    """

    LOGGER.info(f"Creating dataloader for {config.manifest_filepath}")
    LOGGER.info(f"\tDataloader config: {config}")
    LOGGER.info(f"\tBatch size: {config.batch_size}")

    if "augmentor" in config:
        augmentor = process_augmentations(config.augmentor)
    else:
        augmentor = None

    if config.asr_repo == "nemo":
        dataset = AudioToCharDataset(
            manifest_filepath=config.manifest_filepath,
            labels=config.get("labels", None),
            sample_rate=config.sample_rate,
            int_values=config.get("int_values", False),
            augmentor=augmentor,
            max_utts=config.get("max_utts", 0),
            blank_index=config.get("blank_index", -1),
            unk_index=config.get("unk_index", -1),
            normalize=config.get("normalize_transcripts", False),
            trim=config.get("trim_silence", False),
            parser=config.get("parser", "en"),
            return_sample_id=True,
        )
    elif config.asr_repo == "whisper":
        dataset = WhisperDataset(
            datafiles=config.manifest_filepath, language=config.get("language", "en")
        )
    else:
        raise ValueError(f"Unknown ASR repo: {config.asr_repo}")

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=config.get("drop_last", False),
        shuffle=config.get("shuffle", False),
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )


def collate_fn(batch):
    """
    - Pads the shorter audio tensors with zeros.
    - Lengths, labels and labels_lengths are not modified.
    - If one of the audios has multiple channels, only the first one is kept.
    """
    _, audio_lengths, labels, labels_lengths = zip(*batch)
    max_audio_len = max(audio_lengths).item()

    # pad the waveforms
    audio_padded = []
    for sig, sig_len, _, _ in batch:
        if sig.ndim > 1:
            sig = sig[:, 0]
        sig_len = sig_len.item()
        if sig_len < max_audio_len:
            pad = (0, max_audio_len - sig_len)
            sig = torch.nn.functional.pad(sig, pad)
        audio_padded.append(sig)

    # pad the labels
    labels_padded = []
    max_label_len = max(labels_lengths).item()
    for label in labels:
        label_len = label.shape[0]
        if label_len < max_label_len:
            pad = (0, max_label_len - label_len)
            label = torch.nn.functional.pad(label, pad, value=-100)
        labels_padded.append(label)

    audio_padded = torch.stack(audio_padded)
    audio_lengths = torch.stack(audio_lengths)
    labels_padded = torch.stack(labels_padded)
    labels_lengths = torch.stack(labels_lengths)
    return audio_padded, audio_lengths, labels_padded, labels_lengths
