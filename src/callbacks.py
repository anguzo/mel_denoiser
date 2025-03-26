import auraloss
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchmetrics
from lightning import Callback


class LoggingCallback(Callback):
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        # define transformations inverse to dataset.py
        self.db_to_amplitude = lambda x: torchaudio.functional.DB_to_amplitude(
            x, ref=1.0, power=1.0
        )
        self.inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=128,
            sample_rate=sample_rate,
            mel_scale="slaney",
            norm="slaney",
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, power=2.0
        )
        # define auraloss metrics
        self.si_sdr_loss = auraloss.time.SISDRLoss()
        self.mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 4096],
            hop_sizes=[256, 512, 1024],
            win_lengths=[1024, 2048, 4096],
            scale="mel",
            n_bins=128,
            sample_rate=sample_rate,
            perceptual_weighting=True,
        )
        # define aggregation metrics
        self.si_sdr_metric = torchmetrics.aggregation.MeanMetric()
        self.mrstft_metric = torchmetrics.aggregation.MeanMetric()

        self.device = None
        self._reset_metrics()

    def _reset_metrics(self):
        self.si_sdr_metric.reset()
        self.mrstft_metric.reset()

        self.example_dry_mel = None
        self.example_wet_mel = None
        self.example_recovered_mel = None
        self.example_dry_audio = None
        self.example_wet_audio = None
        self.example_recovered_audio = None

    def _mel_to_audio(self, mel_tensor):
        mel_spec = self.db_to_amplitude(mel_tensor)
        mel_to_audio_spec = self.inverse_mel_scale(mel_spec)
        audio = self.griffin_lim(mel_to_audio_spec)
        return audio

    def _ensure_audio_channels(self, audio):
        # if audio is 1D (samples,) -> (1, 1, samples)
        if audio.dim() == 1:
            return audio.unsqueeze(0).unsqueeze(0)
        # if audio is 2D (batch, samples) -> (batch, 1, samples)
        if audio.dim() == 2:
            return audio.unsqueeze(1)
        return audio

    def _log_metrics(self, logger, si_sdr, mrstft, phase, global_step):
        logger.experiment.add_scalar(f"{phase}/si_sdr", si_sdr, global_step)
        logger.experiment.add_scalar(f"{phase}/aura_mrstft", mrstft, global_step)

    def _log_mel_spectrogram(self, logger, mel_dict, phase, example_idx, global_step):
        for tag, mel in mel_dict.items():
            mel_np = mel.cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 2))
            im = ax.imshow(mel_np, aspect="auto", origin="lower", interpolation="none")
            fig.colorbar(im, ax=ax)
            ax.set_title(tag)
            ax.set_xlabel("Frames")
            ax.set_ylabel("Channels")
            plt.tight_layout()
            logger.experiment.add_figure(
                f"{phase}_example_{example_idx}/{tag}", fig, global_step
            )
            plt.close(fig)

    def _log_audio(self, logger, audio_dict, phase, example_idx, global_step):
        for tag, audio in audio_dict.items():
            logger.experiment.add_audio(
                f"{phase}_example_{example_idx}/{tag}",
                audio.cpu(),
                global_step,
                sample_rate=self.sample_rate,
            )

    def _process_batch_end(self, pl_module, batch, batch_idx):
        wet_mel, dry_mel = batch

        with torch.no_grad():
            recovered_mel = pl_module(wet_mel)

        dry_audio = self._mel_to_audio(dry_mel)
        wet_audio = self._mel_to_audio(wet_mel)
        recovered_audio = self._mel_to_audio(recovered_mel)

        dry_audio = self._ensure_audio_channels(dry_audio)
        wet_audio = self._ensure_audio_channels(wet_audio)
        recovered_audio = self._ensure_audio_channels(recovered_audio)

        batch_size = wet_mel.size(0)

        batch_si_sdr = self.si_sdr_loss(recovered_audio, dry_audio)
        batch_mrstft = self.mrstft_loss(recovered_audio, dry_audio)

        self.si_sdr_metric.update(batch_si_sdr)
        self.mrstft_metric.update(batch_mrstft)

        if batch_idx == 0:
            num_examples = min(10, batch_size)
            self.example_dry_mel = dry_mel[:num_examples]
            self.example_wet_mel = wet_mel[:num_examples]
            self.example_recovered_mel = recovered_mel[:num_examples]
            self.example_dry_audio = dry_audio[:num_examples]
            self.example_wet_audio = wet_audio[:num_examples]
            self.example_recovered_audio = recovered_audio[:num_examples]

    def _log_epoch_end(self, trainer, phase):
        # negate si_sdr
        avg_si_sdr = -self.si_sdr_metric.compute()
        avg_mrstft = self.mrstft_metric.compute()
        self._log_metrics(
            trainer.logger, avg_si_sdr, avg_mrstft, phase, trainer.global_step
        )
        if self.example_dry_mel is not None:
            for i in range(self.example_dry_mel.size(0)):
                mel_dict = {
                    "y_(dry)_mel": self.example_dry_mel[i],
                    "x_(wet)_mel": self.example_wet_mel[i],
                    "y_pred_(recovered)_mel": self.example_recovered_mel[i],
                }
                self._log_mel_spectrogram(
                    trainer.logger, mel_dict, phase, i, trainer.global_step
                )
                audio_dict = {
                    "y_(dry)_audio": self.example_dry_audio[i],
                    "x_(wet)_audio": self.example_wet_audio[i],
                    "y_pred_(recovered)_audio": self.example_recovered_audio[i],
                }
                self._log_audio(
                    trainer.logger, audio_dict, phase, i, trainer.global_step
                )

    def setup(self, trainer, pl_module, stage):
        if self.device is None:
            self.device = pl_module.device
            self.inverse_mel_scale.to(self.device)
            self.griffin_lim.to(self.device)
            self.si_sdr_loss.to(self.device)
            self.mrstft_loss.to(self.device)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset_metrics()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._process_batch_end(pl_module, batch, batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_epoch_end(trainer, phase="valid")

    def on_test_epoch_start(self, trainer, pl_module):
        self._reset_metrics()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._process_batch_end(pl_module, batch, batch_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_epoch_end(trainer, phase="test")
