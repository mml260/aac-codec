# AAC-style codec with a temporal-masking ear model

Mike Lundy · MUE 610 Psychoacoustics, University of Miami

A 48 kHz mono perceptual audio codec written in Python with NumPy. It uses a
2048-sample sine-window MDCT (1024 coefficients, 1024-sample hop), band-wise
quantization guided by a psychoacoustic model (PAM), Rice coding, and a bit
reservoir for fixed-bitrate operation, and writes its own binary format
(`.pacc`) that a stand-alone decoder reads.

Two PAMs are compared at matched bitrates, with everything else in the codec
identical:

| | Module | What it is |
|---|---|---|
| **Baseline** | `codec/pam_baseline.py` | MPEG-1 Psychoacoustic Model 1: threshold in quiet, tonal and noise maskers, decimation, level-dependent spreading, global masking threshold → band allowances |
| **Improved** | `codec/pam_temporal.py` | Baseline + temporal masking with pre-echo control: 2.67 ms short-time energies, pre/post-masking, and a tighter threshold where the long window would spread noise into the quiet part before an attack |

The report explains the design and results. This README covers running the
code.

## Quick start

Requires Python 3.9+ and NumPy (Matplotlib only for figures). No audio
libraries or SciPy are needed: WAV I/O and resampling are implemented here.

```bash
python3 -m pip install -r requirements.txt
```

### Decode a file (needs only the `.pacc`)

```bash
python3 -m codec.decoder audio_outputs/castanets/castanets__improved__target064k__actual63.999k.pacc out.wav
```

### Encode a file

Input must be 48 kHz mono 16-bit, such as the prepared excerpts (see below).

```bash
python3 -m codec.encoder data/prepared/speech.wav speech_96k.pacc --rate 96 --model improved
python3 -m codec.decoder speech_96k.pacc speech_96k.wav
```

`--model` is `baseline` or `improved`, and `--rate` is the target in kb/s. Add
`--stats stats.json` to save per-frame bit counts.

## Reproduce every result

1. **Get the source audio.** Download the three original EBU SQAM recordings
   and put them, with these file names, in a folder called `audio excerpts/`
   at the top of the repository:
   * [`27_castanets_single_tone_rhythm_stereo.wav`](https://qoaformat.org/samples/sqam/27_castanets_single_tone_rhythm_stereo.wav)
   * [`40_harpsichord_arpegio_melodious_phrase_stereo.wav`](https://qoaformat.org/samples/sqam/40_harpsichord_arpegio_melodious_phrase_stereo.wav)
   * [`49_female_speech_english_mono.wav`](https://qoaformat.org/samples/sqam/49_female_speech_english_mono.wav)

   They are not included here; see the EBU's [usage terms](https://qc.ebu.io/testmaterials/523/).

2. **Run everything:**

   ```bash
   python3 scripts/reproduce.py
   ```

   This takes about 8 minutes and runs, in order:
   1. input preparation (first 10 s, channels averaged, 44.1 → 48 kHz);
   2. the MDCT reconstruction tests;
   3. the codec tests;
   4. encode and decode of every condition, with metrics;
   5. significance tests;
   6. figures.

   Outputs go to `data/prepared/`, `results/` and `figures/`. The encoder is
   deterministic: the `.pacc` files it writes are byte-identical to those in
   `audio_outputs/`, which was checked on three machines.

Individual steps:

```bash
python3 scripts/prepare_inputs.py          # -> data/prepared/{castanets,harpsichord,speech}.wav
python3 tests/test_mdct.py                 # perfect reconstruction: boundaries, arbitrary lengths
python3 tests/test_codec.py                # exact bit accounting, decoder == encoder, fresh-process decode
python3 scripts/run_all.py --models baseline,improved    # the 18 required conditions -> results/
python3 scripts/significance.py            # -> results/significance.csv
python3 scripts/make_figures.py            # -> figures/
```

## Submitted audio: `audio_outputs/`

For each excerpt (`castanets`, `speech`, and `harpsichord`, which was held out
for final testing), the folder holds the prepared original and, for both PAMs
at 64, 96 and 128 kb/s, the encoded file and the decoded WAV:

```
<source>__<model>__target<RRR>k__actual<AAA.AAA>k.pacc
<source>__<model>__target<RRR>k__actual<AAA.AAA>k.wav
```

`audio_outputs/index.csv` lists every file with its size, actual bitrate and
compression ratio against 768 kb/s. Decoded WAVs are sample-aligned with their
originals and at the original level (no gain or normalization). Each file's
size counts everything in it: the 24-byte header, side information,
coefficients, filler bits and final-byte padding.

## Tests

| Test | Checks |
|---|---|
| `tests/test_mdct.py` | 23 reconstruction cases without quantization: lengths from 1 to 480,001 samples, impulses on the first and last sample, full-scale DC, a step, a sine with abrupt edges, and the three excerpts (max error 5×10⁻¹³) |
| `tests/test_codec.py` | Every frame's bit count matches what is written; the decoder reproduces the encoder's reconstruction exactly; a `.pacc` decodes in a **fresh Python process** in an empty folder holding only the decoder modules and the file |

## Code layout

```
codec/
  mdct.py            orthonormal MDCT/IMDCT, boundary padding, overlap-add
  bands.py           49 AAC scalefactor bands (48 kHz, long window), Bark scale, threshold in quiet
  pam_baseline.py    baseline PAM + mapping of the masking curve to band allowances
  pam_temporal.py    improved PAM (temporal masking with pre-echo control)
  subblocks.py       2.67 ms short-time energies used by the improved PAM
  allocation.py      scalefactors, quantizer, exact Rice bit counting
  encoder.py         two-pass rate control with a 6144-bit bit reservoir
  bitstream.py       file format: header, frame syntax, exact side-info bit counts
  bitio.py           bit writer/reader, Exp-Golomb and Rice codes
  decoder.py         stand-alone decoder
  resample.py        44.1 -> 48 kHz polyphase resampler
  wavio.py           16-bit WAV read/write
evaluation/          independent NMR model, pre-echo measure, paired significance tests
scripts/             prepare_inputs, run_all, significance, make_figures, reproduce
tests/               test_mdct, test_codec (and test_blockswitch, see below)
results/             summary.csv (one row per condition), significance.csv
figures/             all figures (fig1 and fig2a are in the report)
audio_outputs/       the submitted encoded and decoded files
```

## Bitstream format

The header is 24 bytes: magic `PACC`, version, sample rate, hop, original
length, number of frames, target bitrate and model id.

Each frame contains:
1. `max_band` (6 bits);
2. for each band below `max_band`, a 1-bit active flag, followed (if active)
   by the scalefactor and Rice parameter as signed Exp-Golomb differences from
   the previous band, then the band's Rice-coded values (a sign bit after each
   nonzero);
3. a 1-bit filler flag with an optional Exp-Golomb count of filler bits.

Dequantization is `X = q · 2^(sf/4)`. The file ends with up to 7 zero padding
bits. `codec/bitstream.py` has the full definition.

## Extras (not required by the assignment)

* **Control experiment** (`codec/pam_control.py`, model `control`): the
  baseline plus a simple transient detector that lowers all thresholds by
  30 dB on attack frames. The report uses it to test whether the improvement
  comes from the temporal model or simply from giving attack frames more bits.
* **Block switching** (`codec/blockswitch.py`, encoder flag
  `--block-switching`, models `baseline_bs` / `improved_bs`): AAC-style
  switching to eight 256-sample windows on attacks. It is an extension beyond
  the assignment's fixed 2048-sample window and is not part of the report.
  With the flag off, the encoder's output is unchanged.
* **PEAQ** (`scripts/run_peaq.py`, `tools/peaq/`): objective quality grades
  (ITU-R BS.1387-1, basic version) from the open-source GstPEAQ core, built
  without GStreamer. It needs git, a C compiler and glib
  (`brew install glib pkg-config` on macOS):

  ```bash
  git clone https://github.com/HSU-ANT/gstpeaq tools/peaq/gstpeaq
  python3 scripts/run_peaq.py --build     # -> results/peaq.csv
  ```

To run every model, including the extras:
`python3 scripts/run_all.py --models baseline,improved,control,baseline_bs,improved_bs`.
