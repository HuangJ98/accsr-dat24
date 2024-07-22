"""
Assert that the Whisper dataset returns the correct batches.
"""


import unittest
import json

from omegaconf import OmegaConf
import torch
import whisper

from src.dataloader import setup_dataloader
from src.asr.whisper_ptl import SAMPLE_RATE


class TestWhisperDataset(unittest.TestCase):
    def setUp(self):
        """
        Create a dataloader with the Whisper dataset.
        """
        self.batch_size = 2
        self.config = OmegaConf.create(
            {
                "manifest_filepath": "tests/whisper_dataset/train_us.txt",
                "batch_size": self.batch_size,
                "num_workers": 0,
                "sample_rate": SAMPLE_RATE,
                "asr_repo": "whisper",
            }
        )
        self.dataloader = setup_dataloader(self.config)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, language="en", task="transcribe"
        )

    def test_out_shapes(self):
        """
        Compare the batches returned by the Whisper dataset with the expected
        batches, stored in tests/whisper_dataset.
        """
        for i, batch in enumerate(self.dataloader):
            audio, audio_len, tokens, tokens_len = batch
            for values, lengths in [(audio, audio_len), (tokens, tokens_len)]:
                self.assertEqual(values.shape[0], self.batch_size)
                self.assertEqual(values.shape[1], lengths.max().item())

    def test_tokens(self):
        """
        Ensure that the tokens are correct by comparing the decoded text
        with the orginal transcripts, present in the manifest.
        """

        transcripts = list()
        for line in open(self.config.manifest_filepath):
            obj = json.loads(line)
            transcripts.append(obj["text"])
        self.assertTrue(len(transcripts) > 0)

        start_tokens = len(self.tokenizer.sot_sequence_including_notimestamps[1:])
        for batch in self.dataloader:
            labels, labels_len = batch[2], batch[3]
            for i in range(len(labels)):
                transcript = transcripts.pop(0)
                label = labels[i][start_tokens : labels_len[i] - 1]
                decoded = self.tokenizer.decode(label)
                self.assertEqual(
                    decoded, transcript, f"Expected {transcript}, got {decoded}"
                )

        self.assertEqual(len(transcripts), 0)
