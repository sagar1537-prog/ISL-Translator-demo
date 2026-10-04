// ISL Translator Mk II - browser pipeline
// camera -> MediaPipe hands (+ pose) -> smoothing/tracking -> 135 features per frame
// -> two sliding windows (0.8 s and 1.4 s) -> sign model in a Web Worker
// -> word decoder (continuous signing) -> sentence on pause -> English + summary + speech
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "../vendor/mediapipe/vision_bundle.mjs";
import { frameFeatures, FEATURE_DIM, windowFrom } from "./engine2.js";
import { toEnglish, summarize } from "./language.js";
import { HandStage } from "./scene3d.js";

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), videoBox: $("videoBox"), tilt: $("tilt"), gate: $("gate"),
  startBtn: $("startBtn"), gateMsg: $("gateMsg"), liveWord: $("liveWord"), meter: $("meter"),
  sentence: $("sentence"), gloss: $("gloss"), summary: $("summary"), topics: $("topics"), history: $("history"),
  speakBtn: $("speakBtn"), clearBtn: $("clearBtn"), sens: $("sens"), sensVal: $("sensVal"), perf: $("perf"),
  stCam: $("stCam"), stHands: $("stHands"), stModel: $("stModel"),
  frameHint: $("frameHint"), guesses: $("guesses"), testMode: $("testMode"),
};
const ctx = els.overlay.getContext("2d");
const status = (el, s, text) => { el.dataset.s = s; el.querySelector("span").textContent = text; };

const WINDOWS = [0.8, 1.4];          // seconds: fast and normal signing
const SENTENCE_PAUSE = 1.1;          // seconds without hands that end a sentence
const IDLE_END = 3.5;                // ... or this long with hands up but no new word
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
               [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

let cfg = null, spec = null, worker = null, workerReady = false, T = 24;
let hands = null, pose = null, stage = null;
let threshold = 0.55;

// ------------------------------------------------------------------ smoothing + tracking
class OneEuro {
  constructor(minCutoff = 1.0, beta = 30, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null, t: 0 }); }
  static a(c, dt) { const tau = 1 / (2 * Math.PI * c); return 1 / (1 + tau / dt); }
  reset() { this.x = null; }
  filter(v, t) {
    if (this.x === null) { this.x = Float32Array.from(v); this.dx = new Float32Array(v.length); this.t = t; return this.x; }
    const dt = Math.max(t - this.t, 1e-3); this.t = t; const ad = OneEuro.a(this.dCutoff, dt);
    for (let i = 0; i < v.length; i++) {
      const d = (v[i] - this.x[i]) / dt; this.dx[i] = ad * d + (1 - ad) * this.dx[i];
      const a = OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      this.x[i] = a * v[i] + (1 - a) * this.x[i];
    }
    return this.x;
  }
}
class Tracker {          // keeps each hand's identity; votes left/right over time
  constructor() { this.tracks = []; this.next = 0; }
  static c(p) { return [(p[0] + p[15] + p[27] + p[51]) / 4, (p[1] + p[16] + p[28] + p[52]) / 4]; }
  update(dets, t) {
    this.tracks = this.tracks.filter((k) => t - k.seen <= 0.4);
    const free = new Set(this.tracks), got = new Map(), pairs = [];
    dets.forEach((d, i) => { const c = Tracker.c(d.pts); for (const k of free) { const o = Tracker.c(k.raw); pairs.push([Math.hypot(c[0] - o[0], c[1] - o[1]), i, k]); } });
    pairs.sort((a, b) => a[0] - b[0]);
    for (const [dist, i, k] of pairs) { if (dist > 0.2 || got.has(i) || !free.has(k)) continue; got.set(i, k); free.delete(k); }
    return dets.map((d, i) => {
      let k = got.get(i);
      if (!k) { k = { id: this.next++, f: new OneEuro(), pLeft: d.pLeft, seen: t, raw: d.pts }; this.tracks.push(k); }
      else k.pLeft = 0.75 * k.pLeft + 0.25 * d.pLeft;
      if (t - k.seen > 0.15) k.f.reset();
      k.seen = t; k.raw = d.pts;
      return { id: k.id, pts: Float32Array.from(k.f.filter(d.pts, t)), isLeft: k.pLeft > 0.5 };
    });
  }
}
const tracker = new Tracker();

// signer's right / left hand (raw frame: the signer's right hand is on the image's left)
function assignSlots(hs) {
  let right = null, left = null;
  if (hs.length === 2 && hs[0].isLeft === hs[1].isLeft) {
    const [a, b] = hs[0].pts[0] < hs[1].pts[0] ? [hs[0], hs[1]] : [hs[1], hs[0]];
    return { right: a.pts, left: b.pts };
  }
  for (const h of hs) { if (h.isLeft) left = left || h.pts; else right = right || h.pts; }
  return { right, left };
}

// ------------------------------------------------------------------ speech
const nativeTTS = window.AndroidTTS || null;
const speech = {
  modes: ["sentences", "words", "off"], mode: "sentences", voice: null,
  label() { return { sentences: "Speak sentences", words: "Speak words", off: "Speech off" }[this.mode]; },
  pick() {
    if (!("speechSynthesis" in window)) return;
    const vs = speechSynthesis.getVoices();
    const rank = (v) => (v.localService ? 0 : 10) + (/^en-IN/i.test(v.lang) ? 0 : /^en/i.test(v.lang) ? 1 : 5);
    this.voice = vs.length ? [...vs].sort((a, b) => rank(a) - rank(b))[0] : null;
  },
  say(text) {
    if (!text) return;
    if (nativeTTS) { try { nativeTTS.speak(text, 1.0); } catch {} return; }
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text); u.rate = 1.02;
    if (this.voice) { u.voice = this.voice; u.lang = this.voice.lang; } else u.lang = "en-IN";
    if (speechSynthesis.speaking || speechSynthesis.pending) { speechSynthesis.cancel(); setTimeout(() => speechSynthesis.speak(u), 0); }
    else speechSynthesis.speak(u);
  },
  word(t) { if (this.mode === "words") this.say(t); },
  sentence(t) { if (this.mode === "sentences") this.say(t); },
  stop() { if (nativeTTS) { try { nativeTTS.stop(); } catch {} } else if ("speechSynthesis" in window) speechSynthesis.cancel(); },
};
if ("speechSynthesis" in window) { speechSynthesis.addEventListener("voiceschanged", () => speech.pick()); speech.pick(); }
els.speakBtn.textContent = speech.label();
els.speakBtn.addEventListener("click", () => {
  speech.mode = speech.modes[(speech.modes.indexOf(speech.mode) + 1) % 3];
  els.speakBtn.textContent = speech.label();
  els.speakBtn.setAttribute("aria-pressed", String(speech.mode !== "off"));
  if (speech.mode === "off") speech.stop();
});

// ------------------------------------------------------------------ sentences
let current = [];                     // tokens of the sentence being signed
const history = [];
let lastWordAt = 0, lastHandAt = 0;

const token = (label) => ({ label, text: spec.display[label] || label, cat: spec.category[label] || "" });

function renderGloss(tokens) {
  els.gloss.textContent = "";
  for (const t of tokens) { const li = document.createElement("li"); li.textContent = t.text; els.gloss.appendChild(li); }
}
function showSentence(text, draft) {
  els.sentence.textContent = "";
  const s = document.createElement("span");
  if (draft) s.className = "draft";
  s.textContent = text;
  els.sentence.appendChild(s);
  if (!draft) { els.sentence.classList.remove("settle"); void els.sentence.offsetWidth; els.sentence.classList.add("settle"); }
}
function renderSummary() {
  const s = summarize(history);
  const polished = history.length && history[history.length - 1].summary;
  els.summary.textContent = polished || s.text || "A short summary of the conversation builds up here.";
  els.summary.classList.toggle("muted", !s.text);
  els.topics.textContent = "";
  const tags = [...(s.urgent ? ["needs attention"] : []), ...(s.topics || [])];
  for (const t of tags) { const li = document.createElement("li"); li.textContent = t; if (t === "needs attention") li.className = "alert"; els.topics.appendChild(li); }
}
function renderHistory() {
  els.history.textContent = "";
  if (!history.length) { const li = document.createElement("li"); li.className = "muted empty"; li.textContent = "Finished sentences are listed here."; els.history.appendChild(li); return; }
  for (const h of [...history].reverse()) {
    const li = document.createElement("li"); li.textContent = h.english;
    const sm = document.createElement("small"); sm.textContent = h.tokens.map((t) => t.text).join(" · "); li.appendChild(sm);
    els.history.appendChild(li);
  }
}

function setIdle() { els.liveWord.textContent = "Ready"; els.liveWord.classList.add("idle"); els.liveWord.classList.remove("pop"); els.meter.style.width = "0"; }

function addWord(label, p) {
  const tk = token(label);
  current.push(tk);
  lastWordAt = performance.now() / 1000;
  els.liveWord.textContent = tk.text; els.liveWord.classList.remove("pop", "idle"); void els.liveWord.offsetWidth; els.liveWord.classList.add("pop");
  els.meter.style.width = `${Math.round(p * 100)}%`;
  renderGloss(current);
  showSentence(toEnglish(current).english, true);
  speech.word(tk.text);
}

async function finishSentence(force = false) {
  if (!current.length) return;
  // a lone "I"/"you" is the start of a sentence, not a sentence: keep it unless the signer clearly stopped
  if (!force && current.every((t) => t.cat === "pronouns")) return;
  const tokens = current; current = [];
  const { english } = toEnglish(tokens);
  if (!english) return;
  const item = { english, tokens };
  history.push(item);
  showSentence(english, false);
  speech.sentence(english);
  renderHistory(); renderSummary();
  setIdle();
  if (cfg.llm) {                                  // optional: a local LLM polishes the wording
    try {
      const r = await fetch("api/polish", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gloss: tokens.map((t) => t.label), english, history: history.slice(0, -1).map((h) => h.english) }) });
      if (r.ok) {
        const j = await r.json();
        if (j.sentence) { item.english = j.sentence; if (history[history.length - 1] === item && !current.length) showSentence(j.sentence, false); }
        if (j.summary) item.summary = j.summary;
        renderHistory(); renderSummary();
      }
    } catch { /* keep the rule-based text */ }
  }
}

els.clearBtn.addEventListener("click", () => {
  current = []; history.length = 0; renderGloss([]); renderHistory(); renderSummary();
  els.sentence.innerHTML = '<span class="hint">Cleared. Sign the next sentence.</span>';
  setIdle();
});

// ------------------------------------------------------------------ word decoder (continuous signing)
const dec = { recent: [], lastLabel: null, lastAt: 0, released: true, quiet: 0 };
let testMode = false;
function showGuesses(r, none) {
  if (!testMode) return;
  els.guesses.textContent = "";
  for (const [i, p] of r.top || []) {
    const li = document.createElement("li");
    const name = spec.display[spec.labels[i]] || "(none: resting / between signs)";
    li.innerHTML = `<b></b><span></span>`;
    li.firstChild.textContent = `${Math.round(p * 100)}%`;
    li.lastChild.textContent = name;
    if (p < threshold || spec.labels[i] === "none") li.className = "low";
    els.guesses.appendChild(li);
  }
}
function onResult(r) {
  if (r.top) showGuesses(r);
  const none = spec.labels[r.best] === "none" || r.p < threshold;
  els.meter.style.width = `${Math.round((none ? 0 : r.p) * 100)}%`;
  if (none) {
    if (++dec.quiet >= 2) dec.released = true;
    dec.recent.push(null); if (dec.recent.length > 3) dec.recent.shift();
    return;
  }
  dec.quiet = 0;
  const label = spec.labels[r.best];
  dec.recent.push(label); if (dec.recent.length > 3) dec.recent.shift();
  const votes = dec.recent.filter((l) => l === label).length;
  const now = performance.now() / 1000;
  if (votes >= 2 || r.p >= 0.9) {
    const repeat = label === dec.lastLabel;
    const inSentence = current.some((t) => t.label === label);
    if ((!repeat && !inSentence) || (dec.released && now - dec.lastAt > 1.5 && !inSentence)) {
      addWord(label, r.p);
      dec.lastLabel = label; dec.lastAt = now; dec.released = false;
    }
  }
}

// ------------------------------------------------------------------ frame loop
const buffer = [];                    // {t, f}
let lastVideoTime = -1, frameNo = 0, lastPose = null, inflight = false, scaleIx = 0, runId = 0;
let fps = 0, lastFrameT = 0, modelMs = 0;

function draw(hs, poseNow) {
  const W = els.overlay.width, H = els.overlay.height, s = W / 1280;
  ctx.clearRect(0, 0, W, H);
  if (poseNow) {
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 3 * s; ctx.setLineDash([6 * s, 8 * s]);
    ctx.beginPath(); ctx.moveTo(poseNow[1][0] * W, poseNow[1][1] * H); ctx.lineTo(poseNow[2][0] * W, poseNow[2][1] * H); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const h of hs) {
    const P = (i) => [h.pts[i*3] * W, h.pts[i*3+1] * H];
    for (const [lw, c] of [[8 * s, "rgba(14,20,48,.8)"], [4.2 * s, "#ffffff"]]) {
      ctx.lineWidth = lw; ctx.strokeStyle = c; ctx.lineCap = "round"; ctx.beginPath();
      for (const [a, b] of BONES) { const [x1, y1] = P(a), [x2, y2] = P(b); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); }
      ctx.stroke();
    }
    ctx.fillStyle = h.isLeft ? "#FF9F1C" : "#7C8BFF";
    for (let i = 0; i < 21; i++) { const [x, y] = P(i); ctx.beginPath(); ctx.arc(x, y, (i % 4 === 0 ? 6.5 : 5) * s, 0, 7); ctx.fill(); }
  }
}

function schedule(now) {
  if (!workerReady || inflight) return;
  const handRecently = now - lastHandAt < 0.35;
  if (!handRecently) { onResult({ best: spec.labels.length - 1, p: 1 }); return; }
  const span = WINDOWS[scaleIx]; scaleIx = (scaleIx + 1) % WINDOWS.length;
  if (!buffer.length || buffer[0].t > now - span * 0.7) return;     // not enough history yet
  const win = windowFrom(buffer, now - span, now, T);
  const data = new Float32Array(T * FEATURE_DIM);
  win.forEach((f, i) => data.set(f, i * FEATURE_DIM));
  inflight = true;
  worker.postMessage({ type: "run", id: ++runId, scale: span, data }, [data.buffer]);
}

function loop() {
  const v = els.video;
  if (v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime; frameNo++;
    const nowMs = performance.now(), now = nowMs / 1000;
    const res = hands.detectForVideo(v, nowMs);
    if (pose && frameNo % 2 === 0) {                 // the body moves slowly: every other frame is plenty
      const pr = pose.detectForVideo(v, nowMs);
      const lm = pr.landmarks && pr.landmarks[0];
      lastPose = lm && (lm[11].visibility ?? 1) > 0.3 && (lm[12].visibility ?? 1) > 0.3
        ? [[lm[0].x, lm[0].y], [lm[11].x, lm[11].y], [lm[12].x, lm[12].y]] : null;
    }
    const dets = res.landmarks.map((lms, i) => {
      const c = res.handedness && res.handedness[i] && res.handedness[i][0];
      const pLeft = c ? (c.categoryName === "Right" ? c.score : 1 - c.score) : 0.5;   // raw frame: "Right" = signer's left
      return { pts: Float32Array.from(lms.flatMap((p) => [p.x, p.y, p.z])), pLeft };
    });
    const hs = tracker.update(dets, now);
    if (hs.length) lastHandAt = now;
    const { right, left } = assignSlots(hs);
    const aspect = v.videoWidth / v.videoHeight;
    buffer.push({ t: now, f: frameFeatures(right, left, lastPose, aspect) });
    while (buffer.length && buffer[0].t < now - 2.5) buffer.shift();
    // framing check: the model needs your head and shoulders in view
    if (frameNo % 15 === 0) {
      const msg = !lastPose ? "Move back or tilt the camera down: your head and both shoulders must be in view."
        : lastPose[1][1] > 0.92 || lastPose[2][1] > 0.92 ? "Tilt the camera down a little: your shoulders are at the edge."
        : null;
      els.frameHint.hidden = !msg || !pose;
      if (msg) els.frameHint.textContent = msg;
    }
    status(els.stHands, hs.length ? "on" : "off", hs.length === 2 ? "Both hands" : hs.length ? (hs[0].isLeft ? "Left hand" : "Right hand") : "No hands");
    draw(hs, lastPose);
    stage.setHands(hs.map((h) => h.pts), aspect);
    if (frameNo % 2 === 0) schedule(now);
    // end of sentence: a pause with hands down, or no new word for a while
    if (current.length && ((now - lastHandAt > SENTENCE_PAUSE) || (now - lastWordAt > IDLE_END && dec.released)))
      finishSentence(now - lastHandAt > 3 || now - lastWordAt > 6);
    if (lastFrameT) fps = fps ? 0.9 * fps + 0.1 / Math.max(now - lastFrameT, 1e-3) : 1 / (now - lastFrameT);
    lastFrameT = now;
    if (frameNo % 30 === 0) stage.lowPower = fps > 0 && fps < 20;
    if (frameNo % 30 === 0) els.perf.textContent = `${Math.round(fps)} fps · model ${modelMs.toFixed(1)} ms (background)`;
  }
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) v.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ start-up
async function loadAll() {
  cfg = await (await fetch("api/config")).json();
  threshold = cfg.threshold; els.sens.value = threshold; els.sensVal.textContent = `${Math.round(threshold * 100)}%`;
  if (!cfg.model.ready) {
    status(els.stModel, "bad", "No model: run train.bat");
    throw new Error("No trained model yet. Run train.bat once (it uses your Kaggle dataset), then reload this page.");
  }
  spec = await (await fetch("static/model/model.json")).json();
  T = spec.T;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { workerReady = true; status(els.stModel, "on", `${spec.labels.length - 1} signs`); }
    else if (m.type === "error") status(els.stModel, "bad", "Model failed");
    else if (m.type === "result") { inflight = false; modelMs = m.ms; onResult(m); }
  };
  worker.postMessage({ type: "load", spec });
}

async function loadTrackers() {
  const files = await FilesetResolver.forVisionTasks(new URL("../vendor/mediapipe/wasm", import.meta.url).href);
  const make = async (Cls, file, extra) => {
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: new URL(`../models/${file}`, import.meta.url).href, delegate }, runningMode: "VIDEO", ...extra });
    try { return await Cls.createFromOptions(files, opts("GPU")); } catch { return Cls.createFromOptions(files, opts("CPU")); }
  };
  hands = await make(HandLandmarker, "hand_landmarker.task", { numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
  if (cfg.pose) {
    try { pose = await make(PoseLandmarker, "pose_landmarker_lite.task", { numPoses: 1 }); }
    catch (e) { console.warn("body tracking unavailable:", e); pose = null; }
  }
}

let stream = null, cameraWanted = false;
function sync() {
  const { videoWidth: w, videoHeight: h } = els.video; if (!w || !h) return;
  if (els.overlay.width !== w || els.overlay.height !== h) { els.overlay.width = w; els.overlay.height = h; }
  els.videoBox.style.aspectRatio = `${w} / ${h}`;
  status(els.stCam, "on", `Camera ${Math.min(w, h)}p`);
}
els.video.addEventListener("resize", sync);
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" } });
  } catch (e) {
    if (e.name === "NotAllowedError") throw new Error("Camera access is blocked. Allow the camera (address bar icon), then select Start camera again.");
    if (e.name === "NotFoundError") throw new Error("No camera found. Connect a webcam, then select Start camera again.");
    if (e.name === "NotReadableError") throw new Error("Another app is using the camera. Close it, then select Start camera again.");
    throw e;
  }
  cameraWanted = true;
  els.video.srcObject = stream; await els.video.play(); sync();
}
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState !== "visible" || !cameraWanted) return;
  if (stream && stream.getVideoTracks().some((t) => t.readyState === "live")) { if (els.video.paused) els.video.play().catch(() => {}); return; }
  try { await startCamera(); } catch { status(els.stCam, "bad", "Camera off"); }
});

els.testMode.addEventListener("change", () => { testMode = els.testMode.checked; els.guesses.hidden = !testMode; });
els.sens.addEventListener("input", () => { threshold = Number(els.sens.value); els.sensVal.textContent = `${Math.round(threshold * 100)}%`; });

// subtle 3D tilt of the camera card following the pointer
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
  window.addEventListener("pointermove", (e) => {
    const x = e.clientX / innerWidth - 0.5, y = e.clientY / innerHeight - 0.5;
    els.tilt.style.setProperty("--ry", `${(4 - x * 6).toFixed(2)}deg`);
    els.tilt.style.setProperty("--rx", `${(2 + y * 4).toFixed(2)}deg`);
  }, { passive: true });
}

stage = new HandStage($("three"));
const ready = loadAll();
const trackersReady = ready.then(loadTrackers);
ready.catch((e) => { els.gateMsg.textContent = e.message; });
trackersReady.catch(() => {});

els.startBtn.addEventListener("click", async () => {
  els.startBtn.disabled = true; els.gateMsg.textContent = "";
  try {
    els.startBtn.textContent = "Loading…";
    await ready; await trackersReady;
    if (!hands) await loadTrackers();
    els.startBtn.textContent = "Starting camera…";
    await startCamera();
    els.gate.hidden = true;
    if ("speechSynthesis" in window && !nativeTTS) { const u = new SpeechSynthesisUtterance(" "); u.volume = 0; speechSynthesis.speak(u); }
    els.sentence.innerHTML = '<span class="hint">Sign now. Pause briefly to finish a sentence.</span>';
    loop();
  } catch (e) {
    els.gateMsg.textContent = e.message || String(e);
    els.startBtn.textContent = "Start camera"; els.startBtn.disabled = false;
  }
});
