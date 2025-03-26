import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

sample_rate = 44100
n_fft = 2048
hop_length = 512
n_mels = 128

audio_path = "/gpfs/mariana/smbgroup/guitareffectsaire/guitaset_egdb_idmt-smt-guitar_distortion_as_input/test/0/input.wav"

waveform, sr = torchaudio.load(audio_path)
if waveform.size(0) > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    sr = sample_rate

original_path = "./data/original_torchaudio.wav"
torchaudio.save(original_path, waveform, sr)

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

mel_spec_from_log = F.DB_to_amplitude(
    log_mel_spec,
    ref=1.0,
    power=1.0,  # power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.
)


mel_to_audio_spec = T.InverseMelScale(
    n_stft=n_fft // 2 + 1,
    n_mels=mel_spec_from_log.shape[-2],
    sample_rate=sr,
    mel_scale="slaney",
    norm="slaney",
)(mel_spec_from_log)

griffin_lim = T.GriffinLim(
    n_fft=n_fft,
    hop_length=hop_length,
    power=2.0,  # power (float, optional): Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc. (Default: ``2``)
    # pad_mode "reflect" hardcoded
)

mel_to_audio = griffin_lim(mel_to_audio_spec)

reconstructed_path = "./data/reconstructed_torchaudio.wav"
torchaudio.save(reconstructed_path, mel_to_audio, sr)
