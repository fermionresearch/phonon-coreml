"""Log-mel front end = the pip CPU engine's arithmetic (HF ParakeetFeatureExtractor): pre-emphasis 0.97, 512-pt STFT hop 160 win 400
Hann (centre, zero pad), power, slaney mel 128, log(x + 2^-24), per-feature normalisation over the true length. numpy only."""
import numpy as np
N_FFT, HOP, WIN, N_MELS, PREEMPH, SR = 512, 160, 400, 128, 0.97, 16000
LOG_GUARD = 2.0 ** -24

def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64); f_sp = 200.0 / 3; min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp; logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-30) / min_log_hz) / logstep, f / f_sp)
def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64); f_sp = 200.0 / 3; min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp; logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)
def mel_filters(sr=SR, n_fft=N_FFT, n_mels=N_MELS):
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2); mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f); ramps = np.subtract.outer(mel_f, fftfreqs); w = np.zeros((n_mels, 1 + n_fft // 2))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]; upper = ramps[i + 2] / fdiff[i + 1]; w[i] = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)
_MELF = mel_filters()
_WINDOW = np.hanning(WIN).astype(np.float32)                  # torch.hann_window(400, periodic=False) = HF ParakeetFeatureExtractor
_L = (N_FFT - WIN) // 2
_WIN512 = np.pad(_WINDOW, (_L, N_FFT - WIN - _L))             # torch.stft centres a short window inside n_fft

def log_mel(wave, rule="hf"):
    """wave float32 [n] 16 kHz -> feats float32 [128, T] normalised, T = n//160 (HF: features_lengths = n // hop; the STFT's
    extra last frame is dropped, exactly as the HF attention mask drops it)."""
    x = np.asarray(wave, dtype=np.float32); n = len(x)
    x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]])
    x = np.pad(x, (N_FFT // 2, N_FFT // 2))                      # centre, constant (zero) pad as torch.stft pad_mode="constant"
    T = (1 + n // HOP) if rule == "pip" else n // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(T)[:, None]
    frames = x[idx] * _WIN512
    spec = np.fft.rfft(frames.astype(np.float64), axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)   # [T, 257]
    mel = np.log(power @ _MELF.T + LOG_GUARD)                    # [T, 128]
    mean = mel.mean(0, keepdims=True); var = ((mel - mean) ** 2).sum(0) / (T - 1)
    mel = (mel - mean) / (np.sqrt(var) + 1e-5)
    return mel.T.astype(np.float32)
