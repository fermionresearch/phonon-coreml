#!/usr/bin/env python3
"""Phonon-2 Core ML reference runner (Python example): wav in, text + words out. Needs coremltools, numpy, soundfile (no torch).

    python phonon_coreml.py <bundle-dir> file.wav [more.wav ...] [--json out.json] [--reference] [--frames hf|pip]

bundle-dir holds manifest.json, decoder.bin and Phonon-2.mlpackage (encoder functions enc_5s ... enc_35s over one weight blob; each window
runs the smallest function that holds it). The encoder runs on
the Neural Engine; the decoder runs on the CPU through the compiled library that the Swift package also uses (libphonon_tdt.dylib, next to this file). Audio up to
35 s is one window; longer audio is cut into windows of at most 15 s at quiet points, with a 2 s overlap and a word-level stitch when
there is no quiet point. `words` is one {text, start, end} per word in seconds from the start of the file, the pip engine's shape."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from frontend import log_mel, HOP
from segmenter_np import plan as reference_plan, is_silent

FRAME_S = 0.08; SR = 16000

def _load_library():
    import ctypes
    path = os.path.join(HERE, "libphonon_tdt.dylib")
    try: lib = ctypes.CDLL(path)
    except OSError as e: raise SystemExit(f"error: could not load {path} ({e})")
    P, I = ctypes.c_void_p, ctypes.c_int
    lib.phonon2_tdt_create.restype = P; lib.phonon2_tdt_create.argtypes = [I, I, I, I, I, P, I, I] + [P] * 20
    lib.phonon2_tdt_decode_timed.restype = I; lib.phonon2_tdt_decode_timed.argtypes = [P, P, I, P, P, P, I]
    lib.phonon2_tdt_destroy.restype = None; lib.phonon2_tdt_destroy.argtypes = [P]
    return lib

class DecoderBin:
    """decoder.bin (prediction network, joint, vocabulary) and the greedy decoder from the compiled library the Swift package uses."""
    def __init__(self, path):
        d = open(path, "rb").read(); hl = int.from_bytes(d[:4], "little"); self.h = json.loads(d[4:4 + hl]); blob = d[4 + hl:]
        A = {}
        for a in self.h["arrays"]:
            n = int(np.prod(a["shape"]))
            if a["dtype"] == "int6p":
                b = np.frombuffer(blob, np.uint8, count=a["bytes"], offset=a["offset"]).reshape(-1, 3).astype(np.uint32); w = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                u = np.stack([(w >> (6 * k)) & 63 for k in range(4)], 1).reshape(-1)[:n].astype(np.int16); A[a["name"]] = np.where(u >= 32, u - 64, u).astype(np.int8).reshape(a["shape"])
            else: A[a["name"]] = np.frombuffer(blob, dtype=np.dtype(a["dtype"]), count=n, offset=a["offset"]).reshape(a["shape"])
        self._keep = {k: np.ascontiguousarray(v) for k, v in A.items()}
        self.lib = _load_library(); h = self.h
        def q(n): return self._keep[n + ".q"].ctypes.data
        def f16(n): return self._keep[n].view(np.uint16).ctypes.data
        args = [q("decoder.embedding.weight"), f16("decoder.embedding.weight.scale")]
        for l in range(2):
            for kind in ("ih", "hh"):
                w = f"decoder.lstm.weight_{kind}_l{l}"; args += [q(w), f16(w + ".scale"), f16(f"decoder.lstm.bias_{kind}_l{l}")]
        args += [q("decoder.decoder_projector.weight"), f16("decoder.decoder_projector.weight.scale"), f16("decoder.decoder_projector.bias")]
        args += [q("joint.head.weight"), f16("joint.head.weight.scale"), f16("joint.head.bias")]
        self._durs = np.ascontiguousarray(np.array(h["durations"], np.int32))
        self.handle = self.lib.phonon2_tdt_create(h["V"], h["E"], h["H"], h["nhead"], len(h["durations"]), self._durs.ctypes.data, h["blank"], h["max_symbols"], *args)
        if not self.handle: raise RuntimeError("decoder tables could not be loaded")
        self.vocab = h["vocab"]; self.container_sha256 = h["container_sha256"]
    def __del__(self):
        if getattr(self, "handle", None): self.lib.phonon2_tdt_destroy(self.handle); self.handle = None
    def decode(self, enc):
        """-> [(token, frame, duration)]: greedy decode of one window's encoder output (frames x 640)."""
        enc = np.ascontiguousarray(enc, np.float32); T = enc.shape[0]; cap = max(64, 12 * T)
        out, fr, du = (np.zeros(cap, np.int32) for _ in range(3))
        n = self.lib.phonon2_tdt_decode_timed(self.handle, enc.ctypes.data, T, out.ctypes.data, fr.ctypes.data, du.ctypes.data, cap)
        return [(int(out[i]), int(fr[i]), int(du[i])) for i in range(n)]
    def pieces(self, toks):
        """(piece with the word marker as a leading space, start s, duration s) per token, specials dropped."""
        out = []
        for tok, fr, du in toks:
            p = self.vocab[tok]
            if (p.startswith("<|") and p.endswith("|>")) or p in ("<unk>", "<pad>"): continue
            out.append((p.replace("▁", " "), fr * FRAME_S, du * FRAME_S))
        return out

def _is_punct(s): return bool(s) and not any(ch.isalnum() for ch in s)
def words_from_tokens(tokens, offset=0.0, limit=None):
    """pip rule (fermion._speech.segment.words_from_tokens)."""
    words = []; cur = []
    def flush():
        if not cur: return
        text = "".join(t[0] for t in cur).strip()
        if text:
            end = cur[-1][1] + cur[-1][2]
            if limit is not None: end = min(end, limit)
            words.append({"text": text, "start": round(cur[0][1] + offset, 3), "end": round(max(end, cur[0][1]) + offset, 3)})
        cur.clear()
    for piece, start, dur in tokens:
        if piece.startswith(" ") and cur and (not piece.strip() or not _is_punct(piece.strip())): flush()
        cur.append((piece, start, dur))
    flush(); return words

def stitch(kept, incoming, split, jitter=0.25):
    """word-level stitch at an overlapped boundary (see Words.swift)."""
    while kept and kept[-1]["start"] >= split: kept.pop()
    seam = [w for w in kept if w["end"] > split - jitter - 0.5]
    for w in incoming:
        if w["start"] < split - jitter: continue
        dup = False
        for a in seam:
            ov = min(a["end"], w["end"]) - max(a["start"], w["start"])
            if ov <= 1e-9: continue
            if a["text"].lower() == w["text"].lower() or ov > 0.5 * min(a["end"] - a["start"], w["end"] - w["start"]): dup = True; break
        if not dup: kept.append(w)

def plan_windows(audio, window_s=15.0, band_min_s=10.0, overlap_s=2.0, single_shot_max_s=35.0):
    """-> [(start, end, split_or_None)] samples; the Swift Windower, line for line."""
    n = audio.size
    if n <= int(single_shot_max_s * SR): return [(0, n, None)]
    block = int(0.05 * SR); nb = -(-n // block); padded = np.zeros(nb * block, np.float32); padded[:n] = audio
    rms = np.sqrt(np.mean(padded.reshape(nb, block) ** 2, axis=1))
    W = int(window_s * SR) - 400; overlap = int(overlap_s * SR); out = []; start = 0; split = None
    while n - start > W:
        lo, hi = start + int(band_min_s * SR), start + W
        b0, b1, bs = -(-lo // block), min(hi // block, nb), start // block
        cut = None
        if b0 < b1 and b1 > bs:
            gate = max(0.004, 0.18 * float(rms[bs:b1].max())); best = None; i = b0
            while i < b1:
                if rms[i] > gate: i += 1; continue
                j = i
                while j < b1 and rms[j] <= gate: j += 1
                cand = (j - i, -abs((i + j) * block // 2 - hi), (i + j) * block // 2)
                if best is None or cand > best: best = cand
                i = j
            if best: cut = best[2]
        if cut is not None: out.append((start, cut, split)); start = cut; split = None
        else:
            out.append((start, hi, split)); s_lo, s_hi = hi - overlap + int(0.3 * SR), hi - int(0.3 * SR)
            b = -(-s_lo // block); bestb, bestv = -1, np.inf
            while (b + 1) * block <= s_hi and b < nb:
                if rms[b] < bestv: bestv, bestb = rms[b], b
                b += 1
            split = bestb * block + block // 2 if bestb >= 0 else hi - overlap // 2; start = hi - overlap
    out.append((start, n, split)); return out

def speech_blocks(x, block=800):
    """50 ms blocks above the reference gate (max(0.004, 0.18 x peak RMS of the window))."""
    n = x.size // block
    if n == 0: return np.zeros(0, bool)
    rms = np.sqrt((x[:n * block].reshape(n, block) ** 2).mean(1)); return rms > max(0.004, 0.18 * float(rms.max()))

def worst_gap(speech, words):
    """longest run of 50 ms blocks no word covers -> (start s, end s, speech s) or None"""
    covered = np.zeros(speech.size, bool)
    for w in words: covered[max(0, int(w["start"] / 0.05)):min(speech.size, int(w["end"] / 0.05) + 1)] = True
    best = None; i = 0
    while i < speech.size:
        if covered[i]: i += 1; continue
        j = i
        while j < speech.size and not covered[j]: j += 1
        sp = int(speech[i:j].sum())
        if best is None or sp > best[2]: best = (i, j, sp)
        i = j
    return None if best is None else (best[0] * 0.05, best[1] * 0.05, best[2] * 0.05)

class Phonon2CoreML:
    RESCUE_GAP_S = 1.5; RESCUE_MIN_DENSITY = 2.4
    def rescue(self, x, toks, depth=0):
        """long-audio windows only: >= RESCUE_GAP_S of speech energy with no word (or a very sparse window) -> re-decode as two halves
        split at the gap's quietest block; keep the halves when they give more words. Returns new token triples or None."""
        secs = x.size / SR
        if secs < 3.0 or depth >= 2: return None
        speech = speech_blocks(x); speech_s = float(speech.sum()) * 0.05
        if speech_s < 2.0: return None
        words = words_from_tokens(toks, 0.0, secs); gap = worst_gap(speech, words)
        if gap is None or not (gap[2] >= self.RESCUE_GAP_S or len(words) / speech_s < self.RESCUE_MIN_DENSITY): return None
        cut = (gap[0] + gap[1]) / 2
        if gap[1] - gap[0] > 0.4:
            b0 = int(gap[0] / 0.05) + 1; b1 = int(gap[1] / 0.05); e = [(float((x[b * 800:(b + 1) * 800] ** 2).sum()), b) for b in range(b0, b1)]
            if e: cut = min(e)[1] * 0.05 + 0.025
        cut = min(max(cut, 1.0), secs - 1.0); m = int(cut * SR); out = []
        for a, b in ((0, m), (m, x.size)):
            part = x[a:b]; t = [] if is_silent(part) else self.dec.pieces(self.dec.decode(self.encode(part)))
            deeper = self.rescue(part, t, depth + 1)
            if deeper is not None: t = deeper
            out.extend((pc, st + a / SR, du) for pc, st, du in t)
        return out if len(words_from_tokens(out, 0.0, secs)) > len(words) else None

    def __init__(self, bundle, compute="CPU_AND_NE", frames=None, long_audio=None):
        import coremltools as ct
        self.m = json.load(open(os.path.join(bundle, "manifest.json"))); self.dec = DecoderBin(os.path.join(bundle, "decoder.bin"))
        assert self.dec.container_sha256 == self.m["container_sha256"], "decoder.bin and manifest come from different containers"
        self.frames = frames or self.m.get("frames_rule", "hf"); self.cu = getattr(ct.ComputeUnit, compute); self.bundle = bundle; self.ct = ct
        self.long_audio = long_audio or self.m.get("long_audio", "windows15"); self.functions = sorted(self.m["functions_s"]); self.progs = {}; self.pos = {}; self.load_s = 0.0
        self._comp = self._compiled(os.path.join(bundle, self.m["multifunction"]))
    def cache_key(self, pkg):
        """Same key as the Swift package (Transcriber.cacheKey): container + FNV-1a 64 of the program spec + the weight blob's size, so two
        packages of the same weights never share a compiled model, and Swift and Python share one compiled copy."""
        d = os.path.join(pkg, "Data", "com.apple.CoreML"); h = 0xcbf29ce484222325
        for b in open(os.path.join(d, "model.mlmodel"), "rb").read() + os.path.getsize(os.path.join(d, "weights", "weight.bin")).to_bytes(8, "little"):
            h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return self.m["container_sha256"][:16] + "-" + format(h, "016x")
    def _compiled(self, pkg):
        import shutil
        name = os.path.splitext(os.path.basename(pkg))[0] + ".mlmodelc"
        root = os.environ.get("P2_COMPILE_DIR", os.path.expanduser("~/Library/Caches/phonon-coreml")); key = self.cache_key(pkg)
        for d in (os.path.dirname(pkg), os.path.join(root, key)):
            c = os.path.join(d, name)
            if os.path.isdir(c): return c
        tmp = self.ct.models.utils.compile_model(pkg); d = os.path.join(root, key); os.makedirs(d, exist_ok=True); c = os.path.join(d, name); shutil.move(tmp, c); return c
    def prog(self, s):
        if s not in self.progs:
            t0 = time.perf_counter(); self.progs[s] = self.ct.models.CompiledMLModel(self._comp, compute_units=self.cu, function_name=f"{self.m['program_prefix']}{int(s)}s"); self.load_s += time.perf_counter() - t0
        return self.progs[s]
    @staticmethod
    def frames_for(sec): return 1 + int(round(sec * SR)) // HOP
    def pos_table(self, T3):
        if T3 not in self.pos:
            P = 2 * T3 - 1; p = np.arange(T3 - 1, -T3, -1, dtype=np.float64)[:, None]; invf = 1.0 / (10000.0 ** (np.arange(0, 1024, 2) / 1024.0)); f = p * invf[None, :]
            t = np.zeros((1, P, 1024), np.float32); t[0, :, 0::2] = np.sin(f); t[0, :, 1::2] = np.cos(f); self.pos[T3] = t
        return self.pos[T3]
    def encode(self, wave):
        feats = log_mel(wave, self.frames); n = feats.shape[1]; sec = n * HOP / SR + 0.02
        s = next((b for b in self.functions if sec <= b), None); assert s is not None, f"window of {sec:.1f}s exceeds the largest function"
        T = self.frames_for(s); mel = np.zeros((1, 128, T), np.float32); mel[0, :, :n] = feats; lens = [n]; Ts = [T]
        for _ in range(3): lens.append((lens[-1] - 1) // 2 + 1); Ts.append((Ts[-1] - 1) // 2 + 1)
        feed = {"mel": mel}
        for i, name in enumerate(("m1", "m2", "m3")):
            m = np.zeros((1, 1, Ts[i + 1], 1), np.float32); m[:, :, :lens[i + 1]] = 1; feed[name] = m
        km = np.full((1, 1, 1, Ts[3]), -1e4, np.float32); km[..., :lens[3]] = 0; feed["kmask"] = km
        p = self.prog(s)
        pi = self.m.get("pos_input", True); pi = pi.get(str(int(s)), True) if isinstance(pi, dict) else pi   # per function or one flag
        if pi: feed["pos"] = self.pos_table(Ts[3])
        if self.m.get("io") == "float16": feed = {k: v.astype(np.float16) for k, v in feed.items()}   # bundles with 16-bit float inputs
        return np.asarray(p.predict(feed)["enc"], np.float32)[0, :lens[3]]
    def transcribe(self, wave):
        """-> {"text", "words", "segments"}"""
        wave = np.asarray(wave, np.float32).reshape(-1)
        wins = [(a, b, None) for a, b in reference_plan(wave)] if self.long_audio == "reference" else plan_windows(wave)
        words = []; segments = []
        for i, (a, b, split) in enumerate(wins):
            w = wave[a:b]; toks = [] if is_silent(w) else self.dec.pieces(self.dec.decode(self.encode(w)))
            if len(wins) > 1 and self.long_audio != "reference":
                r = self.rescue(w, toks)
                if r is not None: toks = r
            ws = words_from_tokens(toks, a / SR, (b - a) / SR)
            if split is not None: stitch(words, ws, split / SR)
            else: words.extend(ws)
            segments.append({"id": i, "start": round(a / SR, 3), "end": round(b / SR, 3), "text": " ".join(x["text"] for x in ws)})
        return {"text": " ".join(x["text"] for x in words), "words": words, "segments": segments}

def main():
    import argparse, soundfile as sf
    ap = argparse.ArgumentParser(); ap.add_argument("bundle"); ap.add_argument("wavs", nargs="+"); ap.add_argument("--json"); ap.add_argument("--frames", choices=["pip", "hf"]); ap.add_argument("--compute", default="CPU_AND_NE"); ap.add_argument("--reference", action="store_true")
    a = ap.parse_args(); m = Phonon2CoreML(a.bundle, a.compute, a.frames, "reference" if a.reference else None); rows = []; ta = tw = 0.0
    for p in a.wavs:
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        if sr != SR:
            import scipy.signal as ss; w = ss.resample_poly(w, SR, sr)
        t0 = time.perf_counter(); r = m.transcribe(w); dt = time.perf_counter() - t0; ta += len(w) / SR; tw += dt
        print(r["text"]); rows.append({"path": os.path.basename(p), "audio_s": round(len(w) / SR, 2), "wall_s": round(dt, 4), **r})
    if a.json: json.dump({"runtime": "phonon_coreml.py (Core ML encoder, compiled decoder)", "frames_rule": m.frames, "compute": a.compute, "long_audio": m.long_audio, "load_s": round(m.load_s, 2), "audio_s": round(ta, 2), "decode_wall_s": round(tw, 3), "x_realtime": round(ta / tw, 1) if tw else None, "rows": rows}, open(a.json, "w"), indent=1)

if __name__ == "__main__": main()
