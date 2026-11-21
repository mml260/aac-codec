Audio outputs: AAC-style codec with a temporal-masking ear model (Mike Lundy, MUE 610)

Contents
  castanets/, speech/, harpsichord/ (harpsichord was the held-out test excerpt)
    <source>__original.wav                         prepared input: first 10 s of the EBU SQAM
                                                   recording, mono, 48 kHz, 16-bit
    <source>__<model>__target<RRR>k__actual<A>k.pacc   encoded file (this codec's binary format)
    <source>__<model>__target<RRR>k__actual<A>k.wav    decoded file
  index.csv   one row per file: source, model, target rate, actual rate, size, compression ratio

  model    = baseline (MPEG-1 Psychoacoustic Model 1) or improved (temporal masking with pre-echo control)
  target   = 64, 96 or 128 kb/s; actual = file bits x 48000 / 480000 samples
  size     = the whole .pacc file: 24-byte header, side information, coefficients,
             filler bits and final-byte padding
  ratio    = 768 kb/s (48 kHz mono 16-bit PCM) / actual rate

Every decoded WAV is 48 kHz mono 16-bit, exactly 480,000 samples, sample-aligned with
its original and at the original level (no gain or normalization), so files can be
compared or switched between directly.

To decode a .pacc yourself (from the code repository):
  python3 -m codec.decoder castanets/castanets__improved__target064k__actual63.999k.pacc out.wav
