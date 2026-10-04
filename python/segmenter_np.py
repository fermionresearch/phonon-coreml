"""Line-for-line port of fermion._speech.segment.plan / is_silent (numpy) for the reference runner; no silence padding anywhere."""
import numpy as np
SR = 16000; SINGLE_SHOT_MAX_S = 35.0; BAND_MIN_S = 25.0; BAND_MAX_S = 35.0; TARGET_S = 30.0; BLOCK_S = 0.05; NOISE_FLOOR_RMS = 0.004; GATE_RATIO = 0.18
def is_silent(audio): return audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4
def plan(audio, sample_rate=SR):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1); n = int(audio.size)
    if n <= int(SINGLE_SHOT_MAX_S * sample_rate): return [(0, n)]
    block = int(BLOCK_S * sample_rate); n_blocks = -(-n // block); padded = np.zeros(n_blocks * block, dtype=np.float32); padded[:n] = audio
    rms = np.sqrt(np.mean(padded.reshape(n_blocks, block) ** 2, axis=1)); single = int(SINGLE_SHOT_MAX_S * sample_rate); windows = []; start = 0
    while n - start > single:
        lo = start + int(BAND_MIN_S * sample_rate); hi = start + int(BAND_MAX_S * sample_rate); b0 = -(-lo // block); b1 = hi // block
        peak = float(rms[start // block:b1].max()) if b1 > start // block else 0.0; gate = max(NOISE_FLOOR_RMS, GATE_RATIO * peak); quiet = rms[b0:b1] <= gate
        cut = start + int(TARGET_S * sample_rate)
        if quiet.any():
            best = None; i = 0
            while i < quiet.size:
                if not quiet[i]: i += 1; continue
                j = i
                while j < quiet.size and quiet[j]: j += 1
                mid = int(((b0 + i) + (b0 + j)) * block // 2); cand = (j - i, -abs(mid - (start + int(TARGET_S * sample_rate))), mid)
                if best is None or cand > best: best = cand
                i = j
            cut = best[2]
        cut = max(lo, min(hi, cut, n)); windows.append((start, cut)); start = cut
    windows.append((start, n)); return windows
