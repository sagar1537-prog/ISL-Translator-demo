# ISL Translator Mk III: two-way

| Tab | What it does |
|---|---|
| **Sign → English** | Sign at your natural speed. Words appear as you sign. A short pause turns them into an English sentence, which is spoken and summarised. |
| **English → Sign** | Speak (microphone) or type. A 3D avatar signs it back in ISL word order, using **real signers' recorded movements**. |
| **Teach** | Record yourself signing any word three times, including words the dataset lacks ("want", "where", "go"). Then run `finetune.bat`. |

## Start (Windows)

| Step | File | When |
|---|---|---|
| 1 | `setup.bat` | Once: Python environment, packages, body-tracking model |
| 2 | `train.bat` | Once: trains on the whole Kaggle dataset. It asks for the `keypoints` folder the first time. |
| 3 | `run.bat` | Every time: opens http://127.0.0.1:8000 (Chrome or Edge) |
| + | `finetune.bat` | After recording signs in Teach mode (a few minutes) |

## About accuracy

No sign-language recogniser is 100% accurate. People sign differently, cameras and lighting differ, and some ISL signs differ only slightly. Mark III is built to **avoid wrong words** rather than guess:

1. **Calibrated confidence.** After training, the model is calibrated on videos it never saw, so "85%" means it is right about 85% of the time.
2. **Clear winners only.** A word is shown only if it reaches the **Sensitivity** level (default 75%), is at least **25 points ahead of the runner-up**, and wins two readings in a row. Near-ties, where mix-ups happen, are not shown.
3. **"None" class.** Resting hands and the movement between signs are learned explicitly, so in-between moments do not become words.
4. **Your own signing.** The dataset was recorded by other people with other cameras. A few Teach-mode takes of the words you use make the model match *your* hands, camera and room. This is the biggest single improvement for real use. Your takes count 8× in training.
5. **Honest numbers.** `train.bat` writes `model/report.txt` with:
   - top-1 accuracy on held-back videos,
   - **accuracy at each confidence level** (e.g. "at ≥80%: 97% correct, 85% of words shown"),
   - the weakest words and the most-confused pairs.

   Use it to choose the Sensitivity level, and to decide which words to record in Teach mode.

## Tracking fast movement

- **60 fps camera** when your webcam supports it (shown in the status chip). That is twice the frames of most setups, with less motion blur.
- **Hands, body and arms every frame.** Elbows and wrists come from the body tracker, which still follows the arm when the hand itself is blurred.
- **Gap filling.** When a fast movement makes the hand tracker lose a hand for a moment (up to 0.15 s), the hand is carried along its path and pinned to the arm's wrist. On screen it is drawn faded.
- **Adaptive smoothing (One Euro filter).** Steady when still, near-zero lag when moving fast.
- **Speed-varied training.** The model learns each sign sped up, slowed down and with uneven pacing.

## English → Sign

- Speech input uses the browser's speech recognition (Chrome/Edge, needs internet). Typing works offline.
- English is converted to ISL order:
  - articles and "to be" are dropped;
  - greetings come first, then time words, and question words go last;
  - adjectives follow their noun;
  - synonyms are mapped (ticket → train ticket, mom → mother, phone → cell phone…).
- Each sign is a real recorded signer. For every word, the training picks the clearest recording the model is most sure about; your own Teach-mode take is used first if you recorded one.
- Words with no sign appear as dashed text chips. Teach them in Teach mode and they become signs.

## Testing checklist

1. **Framing:** head, both shoulders and room above your head in view; light in front of you. The red banner on the camera must be gone.
2. **Single words with Test mode on:** the right word should be top, well above the second guess. Note the words that fail.
3. **Teach the failing words:** record 3 takes each, run `finetune.bat`, reload, and test again.
4. **Sentences:** sign 2–4 words that work, then put your hands down for about a second.
5. **English → Sign:** type "Good morning, I need a train ticket" and watch the avatar sign GOOD MORNING · TRAIN TICKET.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `ISL_THRESHOLD` | `0.75` | starting Sensitivity (also on the page) |
| `ISL_MARGIN` | `0.25` | how far the winner must lead the runner-up |
| `ISL_LLM_MODEL` | off | optional local LLM (Ollama) to polish English |
| `PORT` | `8000` | web port |

Training options: `train.bat --epochs 100`, `--aspect 1.3333` (4:3 dataset videos), `--fps 30` (dataset frame rate, used for avatar speed).

## Files

| Path | Purpose |
|---|---|
| `train.py`, `features3.py` | Training (dataset + your takes), calibration, report, avatar sign library |
| `app.py` | Server: page, config, Teach-mode recordings, optional LLM |
| `static/js/app.js` | Camera, tracking + gap filling, word decoder, sentences, Teach mode, tabs |
| `static/js/engine3.js`, `worker.js` | The model in the browser, on a background thread (identical to PyTorch) |
| `static/js/language.js` | Signs → English and the summary |
| `static/js/english2isl.js` | English → ISL word order and sign lookup |
| `static/js/avatar.js` | The signing avatar (three.js) |
| `static/js/scene3d.js` | The live 3D hands on the Sign tab |
| `static/signs/` | One recorded signer per word (written by training) |
| `personal/` | Your Teach-mode takes (stay on your computer) |
