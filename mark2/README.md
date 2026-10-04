# ISL Translator Mk II

Sign full sentences at your natural speed. Mk II recognises **all 262 words** of the Kaggle ISL keypoint dataset, turns them into English sentences, speaks them, and keeps a running summary of what the person is telling you. A 3D hand on screen mirrors your hands live.

## Start (Windows)

| Step | File | When |
|---|---|---|
| 1 | **`setup.bat`** | Once. Creates the Python environment, installs the packages and downloads the body-tracking model. |
| 2 | **`train.bat`** | Once, and again after the dataset changes. It asks for the dataset's `keypoints` folder the first time (default `C:\Users\VSJ\Downloads\archive\keypoints`). The first run reads all videos (5–15 min), then trains (15–40 min on a laptop CPU). |
| 3 | **`run.bat`** | Every time. Opens http://127.0.0.1:8000 in your browser. Use Chrome or Edge. |

## Using it

- Select **Start camera** and sign. Each recognised word pops up under **Now signing** and is added to the sentence chips.
- **Pause** (hands down for about a second) to finish a sentence. It is turned into English, spoken, added to **Conversation**, and the **Summary** updates.
- **Speak sentences / Speak words / Speech off** cycles the speech mode. **Clear** starts over.
- **Sensitivity**: lower shows words sooner, higher makes fewer mistakes.

Examples of what the rules produce:

| Signs | English |
|---|---|
| I · sick · doctor · hospital · today | I am sick and need a doctor at the hospital today. |
| hello · train ticket | Hello. I need a train ticket. |
| mother · sick | My mother is sick. |
| father · doctor | My father is a doctor. |
| car · red | The car is red. |
| train station · Monday | I am going to the train station on Monday. |

## How it works (and why it stays fast)

```
camera → MediaPipe hands (+ body) → smoothing → 135 numbers per frame
       → two sliding windows (0.8 s for fast signing, 1.4 s for normal) → sign model in a background thread
       → word decoder → sentence on pause → English rules (+ optional local LLM) → speech + summary
```

- **Body position:** the model sees both hands and where they are relative to your shoulders. Many ISL signs differ only by location, which Mk I could not see.
- **Continuous signing:** the model was trained with a **"none" class** made from transitions between signs and resting hands. That is what splits a fast stream of signs into words.
- **Either hand:** works for left- and right-handed signers. Training mirrors every example.
- **Fast:** everything runs in the browser. The model takes a few milliseconds in a **Web Worker**, so the video, tracking and 3D never wait for it. The server only sends the page.
- **Slow computers:** the 3D view drops to half rate automatically when the computer is slow, so recognition keeps full speed.
- **Privacy:** video never leaves the computer.

### Optional: smoother English with a local LLM

The built-in rules work offline with no extra software. If you run [Ollama](https://ollama.com), the app can polish each sentence and the summary:

```
ollama pull llama3.2:3b
set ISL_LLM_MODEL=llama3.2:3b
run.bat
```

If Ollama isn't running, the app simply keeps the rule-based text.

## Honest limits

- **Vocabulary:** the dataset has **no verbs or question words** ("want", "go", "where"). The English is built from nouns, adjectives, pronouns, places and times: "I · ticket" becomes "I need a ticket". Recording a few verbs yourself would widen it.
- **Training data:** every dataset video is one word, signed by other people with other cameras. Continuous recognition comes from the sliding windows and the "none" training, so expect some missed or extra words in very fast signing.
- **Accuracy:** `train.bat` prints the top-1 and top-5 accuracy on held-back videos. That is the honest number for single words.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | web port |
| `ISL_THRESHOLD` | `0.55` | starting sensitivity (also adjustable on the page) |
| `ISL_LLM_MODEL`, `OLLAMA_URL` | off, `http://127.0.0.1:11434` | optional local LLM |

Training options: `train.bat --epochs 80`, `train.bat --aspect 1.3333` (if the dataset videos are 4:3).

## Files

| Path | Purpose |
|---|---|
| `setup.bat`, `train.bat`, `run.bat` | Windows one-click scripts |
| `train.py` | Reads the dataset, trains the model, writes `static/model/model.json` |
| `features2.py` | The 135 features per frame (mirrored exactly in `static/js/engine2.js`) |
| `app.py` | Small Flask server (page, config, optional LLM polish) |
| `static/js/app.js` | Camera, tracking, word decoder, sentences, speech |
| `static/js/engine2.js`, `worker.js` | The model in the browser, run on a background thread |
| `static/js/language.js` | Signs → English, and the conversation summary |
| `static/js/scene3d.js` | The 3D hand stage (three.js) |
| `static/vendor/` | MediaPipe and three.js, bundled so nothing loads from the internet |
