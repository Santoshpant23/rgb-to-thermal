# Turning normal photos into heat maps — explained simply

## What we're trying to do
A drone flies over a city and takes two kinds of pictures:
- a **normal color photo** (what your eyes see), and
- a **thermal photo** (a "heat map" that shows which surfaces are hot 🔥 or cool 🧊).

Thermal cameras are expensive. So we asked: **can a computer look at just the cheap color
photo and "imagine" the heat map?** If yes, cities could find their hottest rooftops and
streets — the places that make heat waves dangerous — without buying thermal cameras everywhere.

Think of it like a weather app that guesses the temperature of every rooftop from a Google-Maps
photo.

## The picture we're going for
```
   COLOR PHOTO            →   AI   →        HEAT MAP
 (roofs, roads, trees)              (bright = hot, dark = cool)
```

## Two sneaky mistakes the original team made (and we fixed)
Our teammates did great and came 2nd. But when I looked closely, I found two hidden problems
that were quietly holding everything back:

**Mistake 1 — They were grading against the wrong answer key. 🗝️**
The heat map isn't a temperature number stored in the file — it's **color-coded** (like a mood
ring, or those red-blue weather maps). The original code grabbed just the "redness" of each pixel
and called that the heat. But in this color code, "redness" doesn't always mean hotter — so it's
like **grading a math test with the wrong answer key**. We figured out the exact color code and
turned the colors back into the *real* heat values. (Our recovered answer key was 8× more
accurate than the guess they used.)

**Mistake 2 — The two photos didn't line up. 🎯**
The color camera and the heat camera are two *different* cameras pointing at slightly different
"windows" of the ground (different zoom). The original setup just stretched both to the same size
and assumed they matched — but they didn't. Imagine printing two slightly-shifted photos on
tracing paper and trying to grade one against the other: every score is unfair.
We found that the heat camera sees the **middle ~65%** of the color photo, so we **cropped the
color photo to match** before teaching the AI. Now they line up.

> These two fixes mattered *more* than any fancy AI. Fixing the data beat fixing the model.

## What we actually built
We taught the AI three different ways (so we could compare), like asking three students to solve
the same problem:
- **The careful art student** (a normal "predict-the-heat" network) — neat and accurate.
- **The competing artists** (a "GAN": one paints, one critiques) — makes the sharpest, most
  realistic-looking heat maps.
- **The materials expert** (a "physics-style" model) — it first guesses what each surface is
  (roof, road, tree, shadow) and gives each a typical temperature. Bonus: you can *see* its
  reasoning as a labeled map. 🧱🛣️🌳
- **The team vote** (an "ensemble"): ask all three and average — usually the best of all.

## How we made sure we weren't cheating
The biggest danger in AI is **overfitting** — that's when the AI *memorizes* the practice photos
instead of *learning*, like a student who memorizes last year's exam answers but flunks a new
test. To check, we:
1. Hid some photos during learning, and
2. Tested on a **brand-new set of 202 photos the AI had never seen in any way.**

It scored **the same on the new photos as on the practice ones** → it truly learned, it didn't
memorize. ✅

## The scores (simple version)
Two common "report-card" numbers:
- **PSNR** — how close the predicted heat is, pixel by pixel. **Higher = better** (think of it
  like a percentage grade).
- **SSIM** — how well the *shapes and patterns* match (0 = nothing alike, 1 = identical twins).

| | Old hackathon | **Ours (best, on unseen photos)** |
|---|---|---|
| PSNR | ~13–14 | **~19** |
| SSIM | ~0.45 | **~0.70** |

Roughly: we went from a **C** to an **A−**, *and* we proved it on photos the AI had never seen.

## The honest catch ⚠️
Every training photo was taken on **one afternoon in one city** (Ann Arbor). So the AI is great at
places like what it studied — but it might struggle somewhere very different (another city, winter,
nighttime), because it never saw those. That's not a flaw in the AI; it just needs more varied
photos to learn from. (Fun related finding: because the weather barely changed across all photos,
the AI never really learned how *wind* or *temperature* change the heat map — it only ever saw
one kind of weather.)

## One-line summary
**We taught a computer to turn cheap color drone photos into accurate heat maps — and the biggest
wins came from fixing the data (the right answer key + lining up the cameras), not from fancier AI.**
