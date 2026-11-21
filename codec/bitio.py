"""Bit-level writer/reader and the variable-length codes used by the bitstream.

Codes
  * fixed(n):  n-bit unsigned, MSB first
  * unary(q):  q ones followed by a zero
  * rice(m,k): unary(m >> k) then the k low bits of m
  * eg0(u):    order-0 Exp-Golomb for unsigned u >= 0
  * seg0(s):   signed Exp-Golomb, mapping 0,1,-1,2,-2,... -> 0,1,2,3,4,...
"""
import numpy as np


# ---------- code lengths (used by the encoder's exact bit counting) ----------
def eg0_len(u):
    u = np.asarray(u)
    return 2 * np.floor(np.log2(u + 1)).astype(np.int64) + 1


def seg0_map(s):
    s = np.asarray(s, dtype=np.int64)
    return np.where(s > 0, 2 * s - 1, -2 * s)


def seg0_len(s):
    return eg0_len(seg0_map(s))


# ------------------------------- writer --------------------------------------
class BitWriter:
    def __init__(self):
        self._bits = bytearray()  # one byte per bit (0/1); packed at the end

    def __len__(self):
        return len(self._bits)

    def fixed(self, value, n):
        value = int(value)
        if n == 0:
            return
        if value < 0 or value >= (1 << n):
            raise ValueError(f"value {value} does not fit in {n} bits")
        self._bits.extend((value >> (n - 1 - i)) & 1 for i in range(n))

    def unary(self, q):
        self._bits.extend(b"\x01" * int(q))
        self._bits.append(0)

    def eg0(self, u):
        u = int(u) + 1
        n = u.bit_length()
        self._bits.extend(b"\x00" * (n - 1))
        self.fixed(u, n)

    def seg0(self, s):
        s = int(s)
        self.eg0(2 * s - 1 if s > 0 else -2 * s)

    def rice_block(self, q, k):
        """Rice-code signed integers q with parameter k (sign bit after each nonzero)."""
        k = int(k)
        bits = self._bits
        for v in np.asarray(q, dtype=np.int64).tolist():
            m = -v if v < 0 else v
            bits.extend(b"\x01" * (m >> k))
            bits.append(0)
            for i in range(k - 1, -1, -1):
                bits.append((m >> i) & 1)
            if m:
                bits.append(1 if v < 0 else 0)

    def raw_bits(self, n, value=0):
        self._bits.extend(bytes([value & 1]) * int(n))

    def tobytes(self):
        """Pack to bytes; the final byte is zero-padded (pad bits are counted in the file size)."""
        arr = np.frombuffer(bytes(self._bits), dtype=np.uint8)
        return np.packbits(arr).tobytes()


# ------------------------------- reader --------------------------------------
class BitReader:
    def __init__(self, data):
        self._bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).tobytes()
        self.pos = 0

    def bits_left(self):
        return len(self._bits) - self.pos

    def fixed(self, n):
        if n == 0:
            return 0
        v = 0
        b = self._bits
        p = self.pos
        for i in range(n):
            v = (v << 1) | b[p + i]
        self.pos = p + n
        return v

    def unary(self):
        z = self._bits.find(b"\x00", self.pos)
        if z < 0:
            raise EOFError("unterminated unary code")
        q = z - self.pos
        self.pos = z + 1
        return q

    def eg0(self):
        z = self._bits.find(b"\x01", self.pos)
        n = z - self.pos  # leading zeros
        self.pos = z
        return self.fixed(n + 1) - 1

    def seg0(self):
        u = self.eg0()
        return (u + 1) // 2 if u & 1 else -(u // 2)

    def rice_block(self, n, k):
        out = np.empty(n, dtype=np.int64)
        b = self._bits
        for j in range(n):
            q = self.unary()
            m = q << k
            p = self.pos
            for i in range(k):
                m |= b[p + i] << (k - 1 - i)
            p += k
            if m:
                if b[p]:
                    m = -m
                p += 1
            self.pos = p
            out[j] = m
        return out

    def skip(self, n):
        self.pos += int(n)
