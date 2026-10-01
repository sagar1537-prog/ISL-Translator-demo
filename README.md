# ISL Counter (web app)

Indian Sign Language to text in the browser, for a public service counter.
The trained model comes from your `gestures.csv` (14 words, 4 takes each).

## Run it (Windows)

Double-click **`run.bat`**. The first run creates a small Python environment and installs Flask and NumPy.
Then the page opens at <http://127.0.0.1:5000>. Use **Chrome or Edge** and select **Start camera**.

Manual alternative:

```
python -m venv .venv_web
.venv_web\Scripts\activate
pip install -r requirements.txt
python app.py
```

## How it works

```
browser:  webcam -> MediaPipe hand tracking -> smoothing + hand tracking
          -> the same 63 features as the desktop app -> classifier (runs in the page, ~0.1 ms)
          -> voting -> word board, conversation strip, speech
server:   serves the page and the model file (/api/model). /api/predict remains as a fallback.
```

- **Privacy:** the video never leaves the computer, and neither do the hand points. Recognition happens in the browser.
- **Works offline:** MediaPipe, the hand model and the font are all inside `static/`. Nothing loads from the internet.
- **Accepting a word:** a reading counts only at **≥ 70% confidence**. A word is shown when it wins **6 of the last 9** readings,
  or at once after **4 identical readings in a row, each ≥ 92%**. The word is spoken the moment it appears,
  and signing it again after a pause speaks it again immediately.
- **Speed:** the word appears about 0.15 to 0.2 seconds after the sign is held steady (measured in the browser test:
  ~140 ms), with no network wait, because the model runs on the device. Speech uses a voice installed on the computer
  (online voices add a delay) and is warmed up when you press Start camera.
- **Why it's light on the server:** the server only sends the page and a 90 KB model file once. Large files are compressed
  (the 11 MB hand-tracking engine downloads as 3.4 MB) and cached by the browser for a week.

## Settings

Set these as environment variables before starting:

| Variable | Default | Meaning |
|---|---|---|
| `ISL_THRESHOLD` | `0.70` | minimum confidence per reading |
| `ISL_VOTES` / `ISL_BUFFER` | `6` / `9` | votes needed out of recent readings |
| `ISL_FAST_RUN` / `ISL_FAST_CONF` | `4` / `0.92` | express lane: identical readings in a row, each this confident |
| `ISL_MODEL` | `model/isl_model.npz` | model file |
| `PORT` / `HOST` | `5000` / `127.0.0.1` | where the server listens |

Example: `set ISL_THRESHOLD=0.8` then `python app.py`.

Display names (e.g. `storeorshop` → "Store / shop") are set in `DISPLAY_NAMES` near the top of `app.py`.

## Updating the model after recording more signs

1. In the main project folder, record with `collect_data.py`, then train:
   `python train_classifier.py --csv gestures.csv`
   (add `--model gru --window 30` for movement-based signs; the web app supports both).
2. Copy the new `isl_model.npz` from the project folder into `isl_web\model\`, replacing the old one.
3. Restart `app.py`.

Record a few takes of **"none"** (key 0 in the recorder: hands resting or moving between signs) before training.
The current model has no "none" class, so any hand in view is matched to one of its 14 words. The 70% threshold
and the voting filter out most of this, but a "none" class removes it properly.

## Deploying online (free tier)

The server is light enough for free hosting (Render, Railway, Fly.io, PythonAnywhere):

- Start command: `gunicorn app:app --workers 1 --threads 8 --bind 0.0.0.0:$PORT` (already in `Procfile`).
- The site **must use HTTPS**, because browsers only allow the camera on `https://` pages or on `localhost`. Hosting providers give you HTTPS automatically.
- A free Render service sleeps after about 15 minutes without visits, and the next visit takes 30 to 60 seconds to wake it. Open the page once before a demo, or ping `/healthz` every 10 minutes with a free monitor such as UptimeRobot to keep it awake.
- Opening `http://<your-PC-IP>:5000` from a phone on your Wi-Fi will **not** get camera access, because that isn't HTTPS.

## Files

| Path | Purpose |
|---|---|
| `app.py` | Flask server: page, `/api/config`, `/api/model`, `/api/predict` (fallback), `/healthz` |
| `engine.py`, `features.py` | Model runner and features, shared with the desktop app (NumPy only) |
| `model/isl_model.npz` | Trained on your `gestures.csv` |
| `templates/index.html`, `static/style.css`, `static/app.js` | The web page |
| `static/engine.js` | The model and features in JavaScript (identical results to `engine.py`, checked to 4e-7) |
| `static/vendor/mediapipe/`, `static/models/hand_landmarker.task` | Hand tracking, bundled for offline use (Apache 2.0) |
| `static/fonts/` | Atkinson Hyperlegible, a typeface designed for legibility (SIL Open Font License) |
| `run.bat` | One-click start on Windows |
