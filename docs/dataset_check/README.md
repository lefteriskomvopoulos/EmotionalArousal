# POPANE data-quality check (2026-09-16)

Source: POPANE Study 1 free sample (3 participants), OSF https://osf.io/94bpx/ (DOI 10.17605/OSF.IO/94BPX), CC-BY 4.0.
Paper: Behnke et al. 2022, Sci Data 9:10.

## From the file header
- Channels: timestamp, affect, ECG, EDA, temp, respiration, SBP, DBP, marker — all 1000 Hz
- EDA: GSR Amp (ADInstruments), microsiemens, finger electrodes (index + ring, left hand per paper)
- affect: Response Meter (ADInstruments), continuous 0–10 slider (corrected 2026-11-23: the Study 6 recordings reach 0; the header gives no range)
- marker: per-sample block code (-1 = baseline, 3-digit codes = stimulus phases, empty = transitions)

## Measured on the sample (EDA downsampled to 10 Hz, high-passed 0.05 Hz)
| | P1 | P2 | P3 |
|---|---|---|---|
| duration | 15.4 min | 15.7 min | 15.0 min |
| EDA range (µS) | -0.35–2.68 | 0–20.5 | -0.33–2.16 |
| distinct bursts ≥0.02 µS | 72 (4.7/min) | 137 (8.7/min) | 76 (5.1/min) |
| median burst (µS) | 0.028 | 0.065 | 0.038 |
| quiet-stretch noise | 2.7 nS | 782 nS (first 30 s) | 3.6 nS |
| missing samples | 0% | 3.1% | 0% |

## Observations
- Bursts are clearly visible with the classic fast-rise / slow-decay shape → deconvolvable.
- Amplitude is quantised to 0.01 µS steps in the exported CSV; bursts below ~0.02 µS will be lost.
- Small negative EDA values in P1/P3 = amplifier offset; constant shift, harmless.
- P2: noisy first 30 s and 3% dropped samples → trim / interpolate in preprocessing.
- Affect slider is live throughout and changes within stimulus blocks.

Files: popane_check.png (full sessions), popane_zoom.png (60 s at 1000 Hz), metadata_R1.xlsx (official per-study metadata).
