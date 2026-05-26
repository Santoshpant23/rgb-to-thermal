# Round 2 — how do we beat the current winners? (target: > 19.1 dB PSNR on the official 202, no overfitting)

## Current bests (official unseen 202): ensemble 19.11 / a1_rgb 18.45 / a2_gan 18.59 (best perceptual).

## First, your live-weather idea — honest answer
We **already use the capture-time weather** (`drone_and_weather_metadata.json` is Open-Meteo
"live" weather at each image's timestamp/lat-long). The problem isn't that it's missing — it's
that it **barely varies** (all 418 images were shot in one ~1-hour window: wind 7.9–9.0, temp
29.1–29.9). So it carries almost no signal to learn from; we proved this in the perturbation
study. Adding more weather variables won't help *on this dataset*. It would only help with data
captured across many conditions.

**BUT there is a per-image physical signal we are NOT yet using: the SUN's position.**
We have timestamp + lat/long per image → we can compute **solar elevation & azimuth**. Across the
hour and across orientations these *do* change, and the sun is the main thing heating surfaces
(sun-facing vs shaded). This is the right "physical" lever to test instead of weather. → Experiment.

## The two real ceilings (from round 1)
1. **Residual RGB↔thermal misalignment.** We only did a global crop + translation. Leftover
   scale/rotation/parallax caps pixel accuracy. → **Better registration (affine ECC) is the #1 lever.**
2. **Data scarcity / single-condition data** (336 train imgs, one afternoon). → **More data via
   transfer learning** is the principled fix and *reduces* overfitting.

## Ranked levers (by expected gain × reliability, all low overfit-risk)
| lever | idea | risk |
|---|---|---|
| **TTA** | average predictions over flips → free boost | none (no new params) |
| **Weighted ensemble** | tune ensemble weights on val, add more diverse members | none |
| **Affine registration** | per-image ECC affine RGB→thermal, rebuild train data + eval | model-neutral; reduces a real error source |
| **Solar geometry input** | add sun elevation/azimuth (2 ch) from timestamp+latlon | +2 channels only |
| **Alignment-robust loss** | shift-tolerant / perceptual loss (less punished by misalignment) | none |
| **Transfer learning** (stretch) | pretrain on public aerial RGB-T (Kust4K/CART) then fine-tune | more data → less overfit |

## What we will NOT do (would overfit / didn't help)
- Bigger encoders (ConvNeXt-S already lost to -T), more dense priors (depth/AlphaEarth didn't help),
  longer training without more data. With 336 images, capacity is not the bottleneck.

## Plan (evaluate everything on the official unseen 202; keep a held-out val for tuning)
- **Step 1 (free wins, no training):** TTA + weighted ensemble of existing v1 models → new leaderboard.
- **Step 2 (registration):** `data_prep_affine.py` rebuilds train RGB with per-image affine ECC
  (fallback to translation); retrain the top arches (a1_rgb, a2_gan, a4_physics8) on the better-aligned data.
- **Step 3 (physics):** add solar elevation/azimuth channels; train a1_solar; honest test of the weather/sun idea.
- **Step 4 (stretch):** transfer-learn from a public aligned aerial RGB-T set if a quick reliable pull works.
- **Step 5:** final official-202 leaderboard with TTA + weighted ensemble across the strongest members;
  report new winners + confirm no overfit (val ≈ official test).

## Anti-overfitting rules for this round
- All headline numbers on the **official 202** (never used for training/tuning).
- Tune ensemble weights / TTA only on the internal **val** split, then report on official test.
- Prefer methods that add data or fix data, not parameters.
