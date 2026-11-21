"""AAC-style perceptual audio codec with baseline and temporal-masking PAMs.

Fixed parameters required by the assignment:
  48 kHz mono, 2048-sample MDCT window, 1024 coefficients, 1024-sample hop.
"""
FS = 48000
N_WIN = 2048
M = 1024  # hop size == number of MDCT coefficients per frame
