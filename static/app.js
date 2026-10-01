// ISL Counter - browser side.
// Webcam -> MediaPipe HandLandmarker (in the page) -> smoothing + hand tracking
// -> /api/predict (server classifier) -> voting -> word board, transcript, speech.
import { FilesetResolver, HandLandmarker } from "./vendor/mediapipe/vision_bundle.mjs";
import { LocalModel, features } from "./engine.js";

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), canvas: $("overlay"), wrap: $("videoWrap"),
  startPanel: $("startPanel"), startBtn: $("startBtn"), startMsg: $("startMsg"),
  reading: $("reading"), board: $("board"), word: $("word"), meter: $("meterFill"),
  meta: $("boardMeta"), transcript: $("transcript"), speakBtn: $("speakBtn"),
  clearBtn: $("clearBtn"), rule: $("rule"),
  stCamera: $("stCamera"), stHand: $("stHand"), stServer: $("stServer"),
};
const ctx = els.canvas.getContext("2d");

const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
               [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const COLORS = { bone: "#FFFFFF", outline: "rgba(18,26,51,.85)", joint: "#FFFFFF", live: "#F2B705" };

let cfg = null;
let landmarker = null;

// ------------------------------------------------------------------ status
function setStatus(el, state, text) {
  el.dataset.state = state;
  el.querySelector(".txt").textContent = text;
}

// ------------------------------------------------------------------ smoothing
// One Euro filter: steady when the hand is still, almost no lag when it moves.
class OneEuro {
  constructor(minCutoff = 1.0, beta = 30, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null, t: 0 });
  }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  reset() { this.x = null; }
  filter(v, t) {
    if (this.x === null) { this.x = Float32Array.from(v); this.dx = new Float32Array(v.length); this.t = t; return this.x; }
    const dt = Math.max(t - this.t, 1e-3); this.t = t;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    for (let i = 0; i < v.length; i++) {
      const d = (v[i] - this.x[i]) / dt;
      this.dx[i] = ad * d + (1 - ad) * this.dx[i];
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      this.x[i] = a * v[i] + (1 - a) * this.x[i];
    }
    return this.x;
  }
}

// Keeps each physical hand's identity, smooths it, and votes left/right over
// time so a one-frame handedness flip can't change the prediction.
class HandTracker {
  constructor() { this.tracks = []; this.nextId = 0; this.primaryId = -1; }
  static center(p) { // mean of wrist + three knuckles
    let x = 0, y = 0; for (const i of [0, 5, 9, 17]) { x += p[i * 3]; y += p[i * 3 + 1]; } return [x / 4, y / 4];
  }
  update(dets, t) {
    this.tracks = this.tracks.filter((tr) => t - tr.seen <= 0.4);
    const free = new Set(this.tracks), assigned = new Map(), pairs = [];
    dets.forEach((d, i) => { const c = HandTracker.center(d.pts);
      for (const tr of free) { const o = HandTracker.center(tr.raw); pairs.push([Math.hypot(c[0] - o[0], c[1] - o[1]), i, tr]); } });
    pairs.sort((a, b) => a[0] - b[0]);
    for (const [dist, i, tr] of pairs) {
      if (dist > 0.2 || assigned.has(i) || !free.has(tr)) continue;
      assigned.set(i, tr); free.delete(tr);
    }
    const out = dets.map((d, i) => {
      let tr = assigned.get(i);
      if (!tr) { tr = { id: this.nextId++, f: new OneEuro(), pLeft: d.pLeft, seen: t, raw: d.pts }; this.tracks.push(tr); }
      else tr.pLeft = 0.75 * tr.pLeft + 0.25 * d.pLeft;
      if (t - tr.seen > 0.15) tr.f.reset();
      tr.seen = t; tr.raw = d.pts;
      const pts = Float32Array.from(tr.f.filter(d.pts, t));
      return { id: tr.id, pts, isLeft: tr.pLeft > 0.5, area: area(pts) * (tr.id === this.primaryId ? 1.3 : 1) };
    });
    const primary = out.reduce((best, h) => (!best || h.area > best.area ? h : best), null);
    this.primaryId = primary ? primary.id : -1;
    return { hands: out, primary };
  }
}
function area(p) {
  const w = els.video.videoWidth || 1, h = els.video.videoHeight || 1;
  let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
  for (let i = 0; i < 21; i++) { x0 = Math.min(x0, p[i*3]); x1 = Math.max(x1, p[i*3]); y0 = Math.min(y0, p[i*3+1]); y1 = Math.max(y1, p[i*3+1]); }
  return (x1 - x0) * w * (y1 - y0) * h;
}

// ------------------------------------------------------------------ voting
class Voter {
  // A word is shown when it wins `need` of the last `size` readings, or at once when
  // `fastRun` identical readings in a row are each at least `fastConf` confident.
  constructor(size, need, fastRun, fastConf) {
    Object.assign(this, { size, need: Math.min(need, size), fastRun, fastConf, buf: [] });
  }
  push(v) { this.buf.push(v); if (this.buf.length > this.size) this.buf.shift(); }
  result() {
    const b = this.buf;
    if (this.fastRun && b.length >= this.fastRun) {
      const tail = b.slice(-this.fastRun), l = tail[0] && tail[0].label;
      if (l && tail.every((v) => v && v.label === l && v.confidence >= this.fastConf))
        return { label: l, confidence: tail.reduce((s, v) => s + v.confidence, 0) / tail.length };
    }
    const count = new Map();
    for (const v of b) if (v) count.set(v.label, (count.get(v.label) || 0) + 1);
    let best = null, n = 0;
    for (const [l, c] of count) if (c > n) { best = l; n = c; }
    if (!best || n < this.need) return null;
    const confs = b.filter((v) => v && v.label === best).map((v) => v.confidence);
    return { label: best, confidence: confs.reduce((x, y) => x + y, 0) / confs.length };
  }
}

// ------------------------------------------------------------------ speech
// In the Android app, window.AndroidTTS speaks through the phone's own text-to-speech
// engine (offline and instant). In a browser, the Web Speech API is used.
const nativeTTS = window.AndroidTTS || null;
const hasWebSpeech = "speechSynthesis" in window;
const speech = {
  on: !!nativeTTS || hasWebSpeech, voice: null,
  pickVoice() {
    if (!hasWebSpeech) return;
    // Prefer a voice installed on the device: online ("Google ...") voices stream the
    // audio first and add a noticeable delay before the word is heard.
    const vs = speechSynthesis.getVoices();
    const rank = (v) => (v.localService ? 0 : 10) + (/^en-IN/i.test(v.lang) ? 0 : /^en/i.test(v.lang) ? 1 : 5);
    this.voice = vs.length ? [...vs].sort((a, b) => rank(a) - rank(b))[0] : null;
  },
  prime() {                       // warm up the speech engine while the camera is starting
    if (!this.on || nativeTTS || !hasWebSpeech) return;
    this.pickVoice();
    const u = new SpeechSynthesisUtterance(" "); u.volume = 0; speechSynthesis.speak(u);
  },
  say(text) {                     // speaks immediately; the caller decides when a word is new
    if (!this.on) return;
    if (nativeTTS) { try { nativeTTS.speak(text, 1.05); } catch (e) { console.warn(e); } return; }
    if (!hasWebSpeech) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    if (this.voice) { u.voice = this.voice; u.lang = this.voice.lang; } else u.lang = "en-IN";
    if (speechSynthesis.speaking || speechSynthesis.pending) {
      speechSynthesis.cancel();   // drop a previous word that is still playing
      setTimeout(() => speechSynthesis.speak(u), 0);
    } else speechSynthesis.speak(u);
  },
  stop() {
    if (nativeTTS) { try { nativeTTS.stop(); } catch {} }
    else if (hasWebSpeech) speechSynthesis.cancel();
  },
};
if (hasWebSpeech && !nativeTTS) { speechSynthesis.addEventListener("voiceschanged", () => speech.pickVoice()); speech.pickVoice(); }
if (!speech.on) { els.speakBtn.disabled = true; els.speakBtn.textContent = "No speech"; }
els.speakBtn.addEventListener("click", () => {
  speech.on = !speech.on;
  els.speakBtn.setAttribute("aria-pressed", String(speech.on));
  els.speakBtn.textContent = speech.on ? "Speech on" : "Speech off";
  if (!speech.on) speech.stop();
});

// ------------------------------------------------------------------ transcript
const transcript = [];
function renderTranscript() {
  els.transcript.replaceChildren(...(transcript.length
    ? transcript.map((w) => Object.assign(document.createElement("li"), { textContent: w }))
    : [Object.assign(document.createElement("li"), { className: "empty", textContent: "Words appear here in the order they are signed." })]));
}
els.clearBtn.addEventListener("click", () => { transcript.length = 0; renderTranscript(); });

// ------------------------------------------------------------------ board
let shownLabel = undefined;
function showWord(stable, raw) {
  const label = stable ? stable.label : null;
  if (label !== shownLabel) {
    shownLabel = label;
    els.board.dataset.state = label ? "live" : "idle";
    els.word.textContent = label ? cfg.display[label] : "Waiting for a sign";
    els.word.classList.remove("flip"); void els.word.offsetWidth; els.word.classList.add("flip");
    document.querySelectorAll(".vocab li").forEach((li) =>
      li.classList.toggle("is-active", !!label && li.textContent === cfg.display[label]));
  }
  const conf = stable ? stable.confidence : 0;
  els.meter.style.width = `${Math.round(conf * 100)}%`;
  if (stable) {
    const alt = raw && raw.top ? raw.top.find((t) => t.label !== stable.label && t.confidence >= 0.05) : null;
    els.meta.textContent = `${Math.round(conf * 100)}% confident` +
      (alt ? `. Next closest: ${cfg.display[alt.label]} (${Math.round(alt.confidence * 100)}%)` : "");
  } else {
    els.meta.textContent = handVisible ? "Reading your sign…" : "Hold a sign steady in view of the camera.";
  }
}

// ------------------------------------------------------------------ prediction
let voter, inFlight = false, lastRaw = null, handVisible = false, announced = null, quietFrames = 0;
let seq = [];
let latencyAvg = 0, serverOk = null;
let local = null;                  // LocalModel when recognition runs in the browser

function record(r) {
  lastRaw = r;
  voter.push(r.accepted ? r : null);
  els.reading.textContent = handVisible ? `Reading: ${cfg.display[r.label]} ${Math.round(r.confidence * 100)}%` : "";
}

function predictLocal(hand) {
  const w = els.video.videoWidth, h = els.video.videoHeight;
  const f = features(hand.pts, hand.isLeft, w, h, local.feature);
  let p;
  if (local.window) {
    seq.push(f); if (seq.length > local.window) seq.shift();
    if (seq.length < local.window) return false;
    p = local.probs(seq);
  } else p = local.probs(f);
  const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const [conf, idx] = order[0], label = local.labels[idx];
  record({ label, confidence: conf, accepted: conf >= cfg.threshold && label !== cfg.none_label,
           top: order.map(([c, i]) => ({ label: local.labels[i], confidence: c })) });
  return true;
}

// Fallback when the model can't run in the page: ask the server (one request at a time).
async function predictRemote(hand) {
  const body = { is_left: hand.isLeft, width: els.video.videoWidth, height: els.video.videoHeight };
  const pts = Array.from({ length: 21 }, (_, i) => [hand.pts[i*3], hand.pts[i*3+1], hand.pts[i*3+2]]);
  if (cfg.window) {
    seq.push(pts); if (seq.length > cfg.window) seq.shift();
    if (seq.length < cfg.window) return;
    body.sequence = seq;
  } else body.points = pts;

  inFlight = true;
  const t0 = performance.now();
  try {
    const res = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
    const r = await res.json();
    const ms = performance.now() - t0;
    latencyAvg = latencyAvg ? 0.85 * latencyAvg + 0.15 * ms : ms;
    if (serverOk !== true || Math.random() < 0.1) setStatus(els.stServer, "on", `Server ${Math.round(latencyAvg)} ms`);
    serverOk = true;
    record(r);
  } catch (err) {
    if (serverOk !== false) setStatus(els.stServer, "error", "Server offline");
    serverOk = false;
    voter.push(null);
    console.warn("predict failed:", err);
  } finally {
    inFlight = false;
  }
}

function afterVote() {
  const stable = voter.result();
  if (!stable) { if (++quietFrames >= 4) announced = null; }   // re-arm once the sign has really gone
  else quietFrames = 0;
  if (stable && stable.label !== announced) {
    announced = stable.label;
    transcript.push(cfg.display[stable.label]);
    if (transcript.length > 30) transcript.shift();
    renderTranscript();
    speech.say(cfg.display[stable.label]);
  }
  showWord(stable, lastRaw);
}

// ------------------------------------------------------------------ drawing
function draw(hands, primary) {
  const { width: W, height: H } = els.canvas;
  ctx.clearRect(0, 0, W, H);
  const s = W / 1280;
  for (const h of hands) {
    const main = primary && h.id === primary.id;
    const P = (i) => [h.pts[i*3] * W, h.pts[i*3+1] * H];
    ctx.lineCap = "round";
    for (const [lw, color] of [[8 * s, COLORS.outline], [4.4 * s, main ? COLORS.bone : "rgba(255,255,255,.55)"]]) {
      ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.beginPath();
      for (const [a, b] of BONES) { const [x1, y1] = P(a), [x2, y2] = P(b); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); }
      ctx.stroke();
    }
    const jointColor = main && shownLabel ? COLORS.live : COLORS.joint;
    for (let i = 0; i < 21; i++) {
      const [x, y] = P(i), r = (i === 0 ? 8 : [4, 8, 12, 16, 20].includes(i) ? 6.6 : 5.4) * s;
      ctx.beginPath(); ctx.arc(x, y, r + 2.2 * s, 0, Math.PI * 2); ctx.fillStyle = COLORS.outline; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = main ? jointColor : "rgba(255,255,255,.7)"; ctx.fill();
    }
  }
}

// ------------------------------------------------------------------ main loop
const tracker = new HandTracker();
let lastVideoTime = -1;

function loop() {
  const v = els.video;
  if (v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    const now = performance.now();
    const res = landmarker.detectForVideo(v, now);
    const dets = res.landmarks.map((lms, i) => {
      const c = res.handedness && res.handedness[i] && res.handedness[i][0];
      // The model assumes a mirrored image; on the raw camera frame "Right" is the signer's left hand.
      const pLeft = c ? (c.categoryName === "Right" ? c.score : 1 - c.score) : 0.5;
      return { pts: Float32Array.from(lms.flatMap((p) => [p.x, p.y, p.z])), pLeft };
    });
    const { hands, primary } = tracker.update(dets, now / 1000);
    handVisible = !!primary;
    setStatus(els.stHand, primary ? "on" : "off", primary ? (primary.isLeft ? "Left hand" : "Right hand") : "No hand");
    if (primary) {
      if (local) { predictLocal(primary); afterVote(); }
      else if (!inFlight) predictRemote(primary).then(afterVote);
    } else { voter.push(null); seq = []; els.reading.textContent = ""; afterVote(); }
    draw(hands, primary);
  }
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) v.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ start-up
async function loadConfig() {
  try {
    const res = await fetch("api/config");
    cfg = await res.json();
    voter = new Voter(cfg.buffer, cfg.votes, cfg.fast_run, cfg.fast_conf);
    els.rule.textContent = `A word is shown when it reaches ${Math.round(cfg.threshold * 100)}% confidence in ${cfg.votes} of the last ${cfg.buffer} readings, or in ${cfg.fast_run} in a row at ${Math.round(cfg.fast_conf * 100)}%.`;
    try {                               // recognition runs in the browser: no network wait per reading
      local = new LocalModel(await (await fetch("api/model")).json());
      setStatus(els.stServer, "on", "On-device");
    } catch (e) {
      console.warn("on-device model unavailable, using the server:", e);
      setStatus(els.stServer, "on", "Server ready");
    }
    serverOk = true;
  } catch {
    setStatus(els.stServer, "error", "Server offline");
    throw new Error("The recognition server is not running. Start it with python app.py, then reload this page.");
  }
}

async function loadLandmarker() {
  const files = await FilesetResolver.forVisionTasks(new URL("./vendor/mediapipe/wasm", import.meta.url).href);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: new URL("./models/hand_landmarker.task", import.meta.url).href, delegate },
    runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await HandLandmarker.createFromOptions(files, opts("GPU")); }
  catch (e) { console.warn("GPU unavailable, using CPU:", e); return HandLandmarker.createFromOptions(files, opts("CPU")); }
}

let stream = null, cameraWanted = false;

// Match the overlay and the box to the video's real shape. Phones deliver portrait video
// when held upright and switch when rotated, so this runs again on every change.
function syncVideoSize() {
  const { videoWidth: w, videoHeight: h } = els.video;
  if (!w || !h) return;
  if (els.canvas.width !== w || els.canvas.height !== h) { els.canvas.width = w; els.canvas.height = h; }
  els.wrap.style.aspectRatio = `${w} / ${h}`;
  els.wrap.style.setProperty("--ar", (w / h).toFixed(4));
  els.wrap.classList.toggle("is-portrait", h > w);
  setStatus(els.stCamera, "on", `Camera ${Math.min(w, h)}p`);
}
els.video.addEventListener("resize", syncVideoSize);
els.video.addEventListener("loadedmetadata", syncVideoSize);

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    throw new Error("This browser can't use the camera here. Open the page as http://127.0.0.1:5000 (or over https) in Chrome or Edge.");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" },
    });
  } catch (e) {
    const inApp = !!nativeTTS;
    if (e.name === "NotAllowedError") throw new Error(inApp
      ? "Camera access is off. Allow the camera when asked, or turn it on in the app's settings, then select Start camera again."
      : "Camera access is blocked. Allow the camera from the icon in the address bar, then select Start camera again.");
    if (e.name === "NotFoundError") throw new Error("No camera was found. Connect a webcam, then select Start camera again.");
    if (e.name === "NotReadableError") throw new Error("The camera is being used by another app. Close it, then select Start camera again.");
    throw e;
  }
  cameraWanted = true;
  els.video.srcObject = stream;
  await els.video.play();
  syncVideoSize();
  stream.getVideoTracks()[0].addEventListener("ended", () => {
    setStatus(els.stCamera, "warn", "Camera paused");
    if (document.visibilityState === "visible") resumeCamera();
  });
}

// The phone stops the camera when the app goes to the background; restart it on return.
let resuming = false;
async function resumeCamera() {
  if (!cameraWanted || resuming) return;
  const live = stream && stream.getVideoTracks().some((t) => t.readyState === "live");
  if (live) { if (els.video.paused) els.video.play().catch(() => {}); return; }
  resuming = true;
  try { await startCamera(); } catch (e) { setStatus(els.stCamera, "error", "Camera off"); console.warn(e); }
  finally { resuming = false; }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") setTimeout(resumeCamera, 300);
  else speech.stop();
});

let landmarkerLoading = null;       // started when the page opens, so Start camera is quick
function preload() { landmarkerLoading = loadLandmarker().then((l) => (landmarker = l), () => { landmarkerLoading = null; }); }

els.startBtn.addEventListener("click", async () => {
  els.startBtn.disabled = true; els.startMsg.textContent = "";
  try {
    els.startBtn.textContent = "Loading hand tracking…";
    if (!cfg) await loadConfig();
    speech.prime();
    if (!landmarker) { if (landmarkerLoading) await landmarkerLoading; if (!landmarker) landmarker = await loadLandmarker(); }
    els.startBtn.textContent = "Starting camera…";
    await startCamera();
    els.startPanel.hidden = true;
    loop();
  } catch (e) {
    els.startMsg.textContent = e.message || String(e);
    els.startBtn.textContent = "Start camera";
    els.startBtn.disabled = false;
  }
});

loadConfig().catch(() => {});
preload();
renderTranscript();
