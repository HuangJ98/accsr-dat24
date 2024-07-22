"""PTL wrapper for whisper ASR"""


import json
import tempfile
import os

import torch
from tqdm import tqdm
import whisper
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf

from src.dataloader import setup_dataloader


SAMPLE_RATE = 16000


class WhisperWrapper(LightningModule):
    def __init__(self, config, trainer=None):

        super().__init__()
        self.config = config
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.options = whisper.DecodingOptions(
            language=config.lang, without_timestamps=True, fp16=False,
        )
        self.model = whisper.load_model(config.model_name)
        self.encoder = self.model.encoder
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            language=config.lang, task=self.options.task, multilingual=False
        )
        self.no_decay_params = ["bias", "LayerNorm.weight"]
        self._optimizer_param_groups = None  # added on runtime

    @classmethod
    def from_pretrained(cls, model_name):
        """
        Wrapper of the construction, so that we can load NeMo and whisper models
        with the same function.
        """
        config = OmegaConf.create({"model_name": model_name, "lang": "en"})
        return cls(config)

    def forward(self, input_signal, input_signal_length):
        """
        Call the forward pass of the encoder. The input has to be a tensor
        containing the indices of the input sequence tokens in the vocabulary.
        The input_ids_length is required by the NeMo models and is ignored here.
        """
        mels = compute_mels(input_signal, self.device)
        return self.model.encoder(mels)

    def training_step(self, batch, batch_id):
        """
        - Run both the encoder and decoder forward pass
        - Compute, log and return the loss
        """

        torch.set_grad_enabled(True)
        loss = self._compute_loss(batch)[1]
        return {"loss": loss}

    def validation_step(self, batch, batch_id):
        """
        - Compute the model output and loss with  `_compute_loss`
        - In the model output and labels, replace -100 with the tokenizer's EOT token
        - Decode the model output and labels using the tokenizer
        - The WER is not computed; a dummy value is returned, as it is required by the
            ensemble model.
        """

        loss = self._compute_loss(batch)[1]
        return {"val_loss": loss, "val_wer": 0.0}

    def _compute_loss(self, batch):
        """
        - Get the decoded tokens from the labels, by adding the SOT token and removing
            the EOT token
        - Run the batch through the model and compute the loss
        - Return the model output and the loss
        """

        audio, labels = batch[0], batch[2]

        # get the decoded tokens from the labels
        sot = (
            torch.ones(labels.shape[0], 1, dtype=labels.dtype)
            * self.tokenizer.sot_sequence_including_notimestamps[0]
        ).to(self.device)
        dec_input_ids = torch.hstack((sot, labels[:, :-1]))
        dec_input_ids[dec_input_ids == -100] = self.tokenizer.eot

        melspec = compute_mels(audio, self.device)
        audio_features = self.model.encoder(melspec)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        return out, loss

    @torch.no_grad()
    def transcribe(self, audiofiles, batch_size, num_workers):
        """
        Transcribes the given audiofiles and returns their transcriptions in the same
        order. The transcriptions are returned as a list of strings.

        - This method is called by `src.` and conforms to the API of the
            NeMo models.
        """

        if audiofiles is None or len(audiofiles) == 0:
            return [[]]

        # Work in tmp directory - will store manifest file there
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as fp:
                for audio_file in audiofiles:
                    entry = {
                        "audio_filepath": audio_file,
                        "duration": 10,
                        "text": "",
                    }
                    fp.write(json.dumps(entry) + "\n")

            data_config = OmegaConf.create(
                {
                    "manifest_filepath": manifest,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "sample_rate": SAMPLE_RATE,
                    "asr_repo": "whisper",
                }
            )
            dataloader = setup_dataloader(data_config)

            texts = list()
            for test_batch in tqdm(dataloader, desc="Transcribing"):
                texts += self.transcribe_batch(test_batch)

        return [texts]

    def transcribe_batch(self, batch):
        """
        - Pad each audio to span 30 seconds
        - Compute log-mel spectrograms
        - Return the predicted texts
        """

        mels = compute_mels(batch[0].to(self.device), self.device)
        out = self.model.decode(mels, options=self.options)

        return [decoding.text for decoding in out]


def compute_mels(audio, device):
    """
    Fit the audio batch to 30 seconds and compute the mel-spectrogram
    of each audio tensor with Whisper's method.
    """

    if audio.shape[1] < 30 * SAMPLE_RATE:
        audio = torch.cat(
            [
                audio,
                torch.zeros(audio.shape[0], 30 * SAMPLE_RATE - audio.shape[1]).to(
                    device
                ),
            ],
            dim=1,
        )
    elif audio.shape[1] > 30 * SAMPLE_RATE:
        audio = audio[:, : 30 * SAMPLE_RATE]

    mels = list()
    for i in range(audio.shape[0]):  # iterate over waveforms in batch
        mels.append(whisper.log_mel_spectrogram(audio[i]).unsqueeze(0))

    return torch.cat(mels, dim=0)
