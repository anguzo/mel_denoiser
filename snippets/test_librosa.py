import librosa
import soundfile as sf

sample_rate = 44100
n_fft = 2048
hop_length = 512
n_mels = 128

audio_path = "/gpfs/mariana/smbgroup/guitareffectsaire/guitaset_egdb_idmt-smt-guitar_distortion_as_input/test/0/input.wav"

y, sr = librosa.load(audio_path, sr=sample_rate)

original_path = "./data/original_librosa.wav"
sf.write(original_path, y, sr)


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

mel_spec_from_log = librosa.db_to_power(
    log_mel_spec,
    ref=1.0,
)

mel_to_audio = librosa.feature.inverse.mel_to_audio(
    mel_spec_from_log,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    htk=False,
    norm="slaney",
    pad_mode="reflect",
)

reconstructed_path = "./data/reconstructed_librosa.wav"
sf.write(reconstructed_path, mel_to_audio, sr)
