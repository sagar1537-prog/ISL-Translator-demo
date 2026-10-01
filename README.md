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
browser:  webcam -> MediaPipe hand tracking (in the page) -> smoothing + hand tracking
          -> 21 hand points per frame -> POST /api/predict
server:   points -> the same 63 features as the desktop app -> classifier (NumPy) -> word + confidence
browser:  voting -> word board, conversation strip, speech
```

- **Privacy:** the video never leaves the computer. Only the 21 hand points are sent, about 1 KB per reading.
- **Works offline:** MediaPipe, the hand model and the font are all inside `static/`. Nothing loads from the internet.
- **Accepting a word:** a reading counts only at **≥ 70% confidence**, and a word is shown when it wins **10 of the last 15** readings.
  Each word is spoken once and is not repeated until the sign is released.
- **Speed:** a request takes about 10 ms. The server needs only Flask and NumPy, using about 40 MB of RAM and no PyTorch.

## Settings

Set these as environment variables before starting:

| Variable | Default | Meaning |
|---|---|---|
| `ISL_THRESHOLD` | `0.70` | minimum confidence per reading |
| `ISL_VOTES` / `ISL_BUFFER` | `10` / `15` | votes needed out of recent readings |
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
- Opening `http://<your-PC-IP>:5000` from a phone on your Wi-Fi will **not** get camera access, because that isn't HTTPS.

## Files

| Path | Purpose |
|---|---|
| `app.py` | Flask server: page, `/api/config`, `/api/predict`, `/healthz` |
| `engine.py`, `features.py` | Model runner and features, shared with the desktop app (NumPy only) |
| `model/isl_model.npz` | Trained on your `gestures.csv` |
| `templates/index.html`, `static/style.css`, `static/app.js` | The web page |
| `static/vendor/mediapipe/`, `static/models/hand_landmarker.task` | Hand tracking, bundled for offline use (Apache 2.0) |
| `static/fonts/` | Atkinson Hyperlegible, a typeface designed for legibility (SIL Open Font License) |
| `run.bat` | One-click start on Windows |
