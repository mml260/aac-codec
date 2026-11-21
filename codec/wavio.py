"""Minimal WAV I/O using only the standard library + NumPy (16-bit PCM)."""
import wave
import numpy as np


def read_wav(path):
    """Return (samples float64 in [-1, 1), shape (n, ch)), fs)."""
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        fs = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit PCM supported (got {8*sw}-bit)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, ch), fs


def write_wav(path, x, fs):
    """Write mono/stereo float signal as 16-bit PCM. Returns number of clipped samples.

    No gain is applied (playback level is preserved); values outside the
    16-bit range are clipped and counted.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    q = np.round(x * 32768.0)
    clipped = int(np.sum((q > 32767) | (q < -32768)))
    q = np.clip(q, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(q.tobytes())
    return clipped
