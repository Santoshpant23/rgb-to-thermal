# Round 2 — beating the winners (results)
*Goal: push past round-1's best (ensemble 19.11 dB on the official unseen 202) without overfitting.*

## New winner 🏆
**Flip-TTA + a val-tuned weighted ensemble → PSNR 19.28 dB** on the official unseen 202
(was 19.11 in round 1). Every individual model also improved ~+0.3 dB from TTA alone, for free
(no retraining, no new parameters → zero overfitting risk). Adding a full-res solar model and a
second RGB seed did NOT move the needle (19.26→19.28), confirming we're at the data-limited ceiling.

### v2 official-202 leaderboard (TTA applied)
| model | MAE↓ | PSNR↑ | SSIM↑ | corr↑ | cLPIPS↓ |
|---|---|---|---|---|---|
| **[ensemble, weighted]** | **0.0696** | **19.26** | **0.711** | **0.907** | 0.360 |
| [ensemble, equal top-3] | 0.0697 | 19.23 | 0.710 | 0.907 | 0.359 |
| a2_gan + TTA | 0.0725 | 18.91 | 0.694 | 0.899 | **0.316** |
| a1_rgb + TTA | 0.0733 | 18.73 | 0.703 | 0.896 | 0.368 |
| a4_physics8 + TTA | 0.0748 | 18.42 | 0.696 | 0.890 | 0.391 |
| (round-1 ensemble, no TTA) | 0.0705 | 19.11 | 0.705 | 0.904 | 0.350 |

## What I tested to go further — and the honest verdicts

**1. Your live-weather idea — and it paid off.** We *already* use the capture-time weather; it's
near-constant (one ~1h window) so it can't help. The real per-image physical signal is the **sun's
position** (from timestamp+lat/long): elevation 62–64° (near noon), azimuth ~25°. I *expected* this
to be redundant with the shadows already visible in the RGB — but the experiment said otherwise:

| model (res 320, identical settings, held-out val) | val-MAE↓ |
|---|---|
| a1_rgb2 — RGB only (control) | 0.0881 |
| **a1_solar — + sun elevation/azimuth** | **0.0820** |

→ At **reduced 320-res** solar helped (~7% lower val-MAE). **BUT I then retrained at full 512-res**
and it **reversed**: a1_solar_512 val-MAE **0.0827** vs RGB-only 512 **0.0745–0.0757**, and on the
official 202 it ranked **near the bottom (17.51 dB)**. So the low-res "gain" was a *crutch* — when
the RGB is under-resolved, the sun hint helps; at full res the RGB already encodes shadow/orientation,
so the extra channels are redundant and add noise. **Net: solar does NOT help the strong model — do
not use it.** (Honest reversal; this is exactly why we re-tested at full res before shipping it.)

**2. Better registration (the round-1 ceiling).** Tested per-image scale+shift vs the fixed
0.65 crop: edge-alignment improved only **0.180 → 0.194**, and per-image best scales clustered at
0.62–0.66 (the fixed crop is already near-optimal). The leftover misalignment is **parallax**
(building-height-dependent, non-rigid) which affine/similarity transforms can't fix. → A full
affine rebuild+retrain is **not worth it**; would need depth-aware (3D) registration.

**3. Bigger models / more dense priors.** Already shown in round 1 to not help (ConvNeXt-S < -T;
depth & AlphaEarth didn't beat RGB-only). With 336 training images, capacity isn't the bottleneck.

## Where we are: at the ceiling for this data
The final push (full-res solar + a deep-ensemble member) moved the winner only **19.26 → 19.28** —
a tie. Solar was rejected at full res; the extra RGB seed added ~nothing (ensemble already saturated).
This strongly says **~19.3 dB is the achievable ceiling for this dataset + approach family**, and the
limit is now the **data** (336 images, one afternoon, residual parallax misalignment), not the model.

## The lever that should actually move it next
- **More and more-varied data** — the real ceiling. Two concrete paths:
- **Transfer learning**: pretrain on a public aligned aerial RGB-thermal set (e.g. **Kust4K**,
  ~4,000 pairs; **CART**) then fine-tune on our 336. 10× the thermal-translation examples →
  better features and *less* overfitting. (Deferred here: GPU was occupied by other users + dataset
  download/setup; flagged as the top next step.)
- **Capture across conditions** (times of day, days, seasons) so the model can actually learn
  weather/sun effects — which this single-afternoon dataset fundamentally cannot teach.

## No overfitting — re-confirmed
All headline numbers are on the **official 202**, which was never used for training *or* tuning
(ensemble weights/TTA were tuned only on the internal val split). Official-test ≈ internal val/test
across both rounds → the models generalize; they didn't memorize.

## Files
Local `rgb2thermal_out/`: `leaderboard_v2.csv`, `v2_official_gallery.png` (RGB | GT | weighted
ensemble | a1_rgb | a2_gan | a4_physics8), `APPROACH_v2.md`, this report.
Cluster: `rgb2thermal_v2/outputs/`, scripts `rgb2thermal/eval_v2.py`, `compute_solar.py`.
