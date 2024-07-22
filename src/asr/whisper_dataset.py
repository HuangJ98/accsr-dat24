import os
import json

import torch
import torchaudio
import whisper


SAMPLE_RATE = 16000


class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, datafiles, language):
        """
        Create a dataset from a list of datafiles. The sample rate is the
        sampling rate of the audio files; if it is different from Whisper's sampling
        rate (16kHz), the audio files will be resampled.

        """

        super().__init__()

        # ensure that the datafiles are a list
        if isinstance(datafiles, str):
            datafiles = [datafiles]

        # store the number of samples of each datafile
        self.datafiles = datafiles
        self.n_samples = list()
        for datafile in self.datafiles:
            self.n_samples.append(0)
            with open(datafile) as f:
                for _ in f:
                    self.n_samples[-1] += 1

        # initialize the Whisper tokenizer
        self.options = whisper.DecodingOptions(
            language=language, without_timestamps=True
        )
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=language, task=self.options.task
        )

    def __len__(self):
        """Return the number of samples in all datafiles."""
        return sum(self.n_samples)

    def __getitem__(self, sample_idx):
        """
        Return the `sample_idx`-th sample of the dataset. The id is the index of the
        sample when considering all the datafiles as a single dataset. The item is
        returned as a tuple of (audio, audio_length, tokens, tokens_length). This is
        how NeMo does it, and it allows us to use the same collation function.
        """

        # find the datafile containing the id-th sample
        for datafile_idx, n_samples in enumerate(self.n_samples):
            if sample_idx < n_samples:
                break
            sample_idx -= n_samples

        # load the audio and the tokens
        with open(self.datafiles[datafile_idx]) as f:
            for line_idx, line in enumerate(f):
                if line_idx == sample_idx:
                    break

        obj = json.loads(line)
        audio = load_wave(obj["audio_filepath"])
        tokens = self.tokenizer.encode(obj["text"])

        # add the SOT and EOT tokens, as done by openai_whisper_finetuning
        tokens = torch.tensor(
            (
                *self.tokenizer.sot_sequence_including_notimestamps[1:],
                *tokens,
                self.tokenizer.eot,
            )
        )

        return (
            audio,
            torch.tensor(audio.shape[0]),
            tokens,
            torch.tensor(tokens.shape[0]),
        )


def load_wave(wave_path):
    """
    Load the waveform from the given path. If the sampling rate is different from
    Whisper's sampling rate (16kHz), resample the waveform. Return the waveform as a
    1D tensor.
    """

    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    return waveform.squeeze(0)
