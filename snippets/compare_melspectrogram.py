import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

sample_rate = 44100
n_fft = 2048
hop_length = 512
n_mels = 128


def compute_torchaudio_melspec(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    mel_spec = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,  # power (float, optional): Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc. (Default: ``2``)
        mel_scale="slaney",
        norm="slaney",
        center=True,
        pad_mode="reflect",
    )(waveform)

    log_mel_spec = T.AmplitudeToDB(
        stype="power",
        top_db=80.0,
    )(mel_spec)

    # log_mel_spec = F.amplitude_to_DB(
    #     mel_spec,
    #     multiplier=10.0,  # multiplier (float): Use 10. for power and 20. for amplitude
    #     db_multiplier=torch.log10(
    #         torch.tensor(1.0).clamp(min=1e-10)
    #     ),  # to make identical to librosa `log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))`
    #     amin=1e-10,
    #     top_db=80.0,
    # )

    return mel_spec.squeeze().numpy(), log_mel_spec.squeeze().numpy()


def compute_librosa_melspec(audio_path):
    y, sr = librosa.load(audio_path, sr=sample_rate)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,  # power : float > 0 [scalar] Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.
        htk=False,
        norm="slaney",
        center=True,
        pad_mode="reflect",
    )

    log_mel_spec = librosa.power_to_db(
        mel_spec,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
    )

    return mel_spec, log_mel_spec


def compare_spectrograms(audio_path):
    print(f"Comparing mel spectrograms for: {audio_path}")

    torch_mel, torch_log_mel = compute_torchaudio_melspec(audio_path)
    librosa_mel, librosa_log_mel = compute_librosa_melspec(audio_path)

    min_length = min(torch_mel.shape[1], librosa_mel.shape[1])
    torch_mel = torch_mel[:, :min_length]
    librosa_mel = librosa_mel[:, :min_length]
    torch_log_mel = torch_log_mel[:, :min_length]
    librosa_log_mel = librosa_log_mel[:, :min_length]

    mel_mse = np.mean((torch_mel - librosa_mel) ** 2)
    log_mel_mse = np.mean((torch_log_mel - librosa_log_mel) ** 2)

    print(f"Mel Spectrogram MSE:\t{mel_mse}")
    print(f"Log Mel Spectrogram MSE:\t{log_mel_mse}")

    # Create directory for plots if it doesn't exist
    os.makedirs("./data/comparison", exist_ok=True)

    # Plot differences
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("Torchaudio Mel Spectrogram")
    plt.imshow(torch_mel, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Librosa Mel Spectrogram")
    plt.imshow(librosa_mel, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Torchaudio Log Mel Spectrogram")
    plt.imshow(torch_log_mel, aspect="auto", origin="lower")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Librosa Log Mel Spectrogram")
    plt.imshow(librosa_log_mel, aspect="auto", origin="lower")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("./data/comparison/mel_spectrogram_comparison.png")

    # Plot absolute differences
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Mel Spectrogram Difference (MSE: {mel_mse:.6f})")
    diff_img = plt.imshow(
        np.abs(torch_mel - librosa_mel), aspect="auto", origin="lower"
    )
    plt.colorbar(diff_img)

    plt.subplot(1, 2, 2)
    plt.title(f"Log Mel Spectrogram Difference (MSE: {log_mel_mse:.6f})")
    log_diff_img = plt.imshow(
        np.abs(torch_log_mel - librosa_log_mel), aspect="auto", origin="lower"
    )
    plt.colorbar(log_diff_img)

    plt.tight_layout()
    plt.savefig("./data/comparison/mel_spectrogram_differences.png")

    return mel_mse, log_mel_mse


if __name__ == "__main__":
    audio_path = "/gpfs/mariana/smbgroup/guitareffectsaire/guitaset_egdb_idmt-smt-guitar_distortion_as_input/test/0/input.wav"
    mel_mse, log_mel_mse = compare_spectrograms(audio_path)
