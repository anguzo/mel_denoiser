import glob
import os

import lightning as L
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class GuitarMelDataset(Dataset):
    def __init__(
        self, root_dir, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.folders = sorted(glob.glob(os.path.join(root_dir, "*")))

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,  # power (float, optional): Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc. (Default: ``2``)
            mel_scale="slaney",
            norm="slaney",
            center=True,
            pad_mode="reflect",
        )

        self.log_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80.0,
        )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = self.folders[idx]
        wet_path = os.path.join(folder_path, "distortion.wav")
        dry_path = os.path.join(folder_path, "clean.wav")

        wet_waveform, sr_wet = torchaudio.load(wet_path)
        dry_waveform, sr_dry = torchaudio.load(dry_path)

        if wet_waveform.size(0) > 1:
            wet_waveform = torch.mean(wet_waveform, dim=0, keepdim=True)
        if dry_waveform.size(0) > 1:
            dry_waveform = torch.mean(dry_waveform, dim=0, keepdim=True)

        if sr_wet != self.sample_rate or sr_dry != self.sample_rate:
            wet_waveform = torchaudio.functional.resample(
                wet_waveform, sr_wet, self.sample_rate
            )
            dry_waveform = torchaudio.functional.resample(
                dry_waveform, sr_dry, self.sample_rate
            )

        wet_mel = self.mel_transform(wet_waveform)
        dry_mel = self.mel_transform(dry_waveform)

        wet_mel = self.log_transform(wet_mel)
        dry_mel = self.log_transform(dry_mel)

        wet_mel = wet_mel.squeeze(0)
        dry_mel = dry_mel.squeeze(0)

        return wet_mel, dry_mel


class MelDenoiserDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # pass along any mel-spectrogram or sample rate settings
        self.kwargs = kwargs

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GuitarMelDataset(
                os.path.join(self.data_dir, "train"), **self.kwargs
            )
            self.val_dataset = GuitarMelDataset(
                os.path.join(self.data_dir, "val"), **self.kwargs
            )
        if stage == "test" or stage is None:
            self.test_dataset = GuitarMelDataset(
                os.path.join(self.data_dir, "test"), **self.kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
