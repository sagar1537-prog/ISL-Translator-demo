// ISL Translator Mk III - browser
//   Sign -> English : camera -> hands + body (every frame) -> gap-filling tracker -> 143 features
//                     -> two windows (0.8 s, 1.4 s) -> model in a Web Worker -> strict word decoder
//                     -> sentence on pause -> English + summary + speech
//   English -> Sign : microphone or typing -> ISL gloss -> avatar replays real signers
//   Teach           : record your own takes of any word (new words too), then fine-tune
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "../vendor/mediapipe/vision_bundle.mjs";
import { frameFeatures, FEATURE_DIM, POSE_IDS, windowFrom } from "./engine3.js";
import { toEnglish, summarize } from "./language.js";
import { HandStage } from "./scene3d.js";
import { EnglishToISL } from "./english2isl.js";
import { SignAvatar } from "./avatar.js";

const $ = (id) => document.getElementById(id);
const els = Object.fromEntries([
  "video", "overlay", "videoBox", "tilt", "gate", "startBtn", "gateMsg", "liveWord", "meter", "sentence", "gloss",
  "summary", "topics", "history", "speakBtn", "clearBtn", "sens", "sensVal", "perf", "stCam", "stHands", "stModel",
  "frameHint", "guesses", "testMode", "three", "avatar", "avatarWord", "e2sInput", "e2sGo", "micBtn", "micState",
  "e2sGloss", "e2sSpeed", "e2sSpeedVal", "e2sReplay", "e2sNote", "teachWord", "teachList", "teachRec", "teachState",
  "teachCount", "mySigns", "recOverlay", "wordList",
].map((id) => [id, $(id)]));
const ctx = els.overlay.getContext("2d");
const status = (el, s, text) => { el.dataset.s = s; el.querySelector("span").textContent = text; };

const WINDOWS = [0.8, 1.4];
const INFER_EVERY = 1 / 15;            // model runs 15x per second, whatever the camera frame rate
const SENTENCE_PAUSE = 1.1, IDLE_END = 3.5;
const GAP = 0.15;                      // seconds a lost hand is predicted (fast movement / motion blur)
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
               [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

let cfg = null, spec = null, worker = null, workerReady = false, T = 24;
let hands = null, pose = null, stage = null, avatar = null, e2i = null;
let threshold = 0.75, margin = 0.25;

// ------------------------------------------------------------------ smoothing + tracking (with gap filling)
class OneEuro {
  constructor(minCutoff = 1.0, beta = 40, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null, t: 0 }); }
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
class Tracker {
  constructor() { this.tracks = []; this.next = 0; }
  static c(p) { return [(p[0] + p[15] + p[27] + p[51]) / 4, (p[1] + p[16] + p[28] + p[52]) / 4]; }
  /** dets: [{pts, pLeft}], poseWrists: {left:[x,y]|null, right:[x,y]|null} */
  update(dets, t, poseWrists) {
    this.tracks = this.tracks.filter((k) => t - k.seen <= 0.5);
    const free = new Set(this.tracks), got = new Map(), pairs = [];
    dets.forEach((d, i) => { const c = Tracker.c(d.pts); for (const k of free) {
      const o = Tracker.c(k.pts); const dt = t - k.seen;            // compare against where it should be by now
      pairs.push([Math.hypot(c[0] - (o[0] + k.vx * dt), c[1] - (o[1] + k.vy * dt)), i, k]); } });
    pairs.sort((a, b) => a[0] - b[0]);
    for (const [dist, i, k] of pairs) { if (dist > 0.25 || got.has(i) || !free.has(k)) continue; got.set(i, k); free.delete(k); }
    const out = dets.map((d, i) => {
      let k = got.get(i);
      if (!k) { k = { id: this.next++, f: new OneEuro(), pLeft: d.pLeft, seen: t, pts: d.pts, vx: 0, vy: 0 }; this.tracks.push(k); }
      else k.pLeft = 0.75 * k.pLeft + 0.25 * d.pLeft;
      if (t - k.seen > GAP + 0.05) k.f.reset();
      const prev = Tracker.c(k.pts), pts = Float32Array.from(k.f.filter(d.pts, t)), now = Tracker.c(pts);
      const dt = Math.max(t - k.seen, 1e-3);
      if (t > k.seen) { k.vx = 0.6 * k.vx + 0.4 * (now[0] - prev[0]) / dt; k.vy = 0.6 * k.vy + 0.4 * (now[1] - prev[1]) / dt; }
      k.seen = t; k.pts = pts;
      return { id: k.id, pts, isLeft: k.pLeft > 0.5, predicted: false };
    });
    // hands lost for a moment (fast movement blurs them): carry them along their path,
    // pinned to the body tracker's wrist when it still sees the arm
    for (const k of free) {
      const dt = t - k.seen;
      if (dt > GAP || dt <= 0) continue;
      const wrist = k.pLeft > 0.5 ? poseWrists.left : poseWrists.right;
      const dx = wrist ? wrist[0] - k.pts[0] : k.vx * dt, dy = wrist ? wrist[1] - k.pts[1] : k.vy * dt;
      const pts = Float32Array.from(k.pts);
      for (let i = 0; i < 21; i++) { pts[i*3] += dx; pts[i*3+1] += dy; }
      out.push({ id: k.id, pts, isLeft: k.pLeft > 0.5, predicted: true });
    }
    return out;
  }
}
const tracker = new Tracker();

function assignSlots(hs) {
  let right = null, left = null;
  if (hs.length >= 2 && hs[0].isLeft === hs[1].isLeft) {
    const [a, b] = hs[0].pts[0] < hs[1].pts[0] ? [hs[0], hs[1]] : [hs[1], hs[0]];   // raw frame: signer's right is on the image's left
    return { right: a.pts, left: b.pts };
  }
  for (const h of hs) { if (h.isLeft) left = left || h.pts; else right = right || h.pts; }
  return { right, left };
}

// ------------------------------------------------------------------ speech output
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

// ------------------------------------------------------------------ sentences (sign -> English)
let current = [];
const history = [];
let lastWordAt = 0, lastHandAt = 0;
const token = (label) => ({ label, text: spec.display[label] || label, cat: spec.category[label] || "" });

function renderGloss(tokens) {
  els.gloss.textContent = "";
  for (const t of tokens) { const li = document.createElement("li"); li.textContent = t.text; els.gloss.appendChild(li); }
}
function showSentence(text, draft) {
  els.sentence.textContent = "";
  const s = document.createElement("span"); if (draft) s.className = "draft"; s.textContent = text; els.sentence.appendChild(s);
  if (!draft) { els.sentence.classList.remove("settle"); void els.sentence.offsetWidth; els.sentence.classList.add("settle"); }
}
function renderSummary() {
  const s = summarize(history);
  const polished = history.length && history[history.length - 1].summary;
  els.summary.textContent = polished || s.text || "A short summary of the conversation builds up here.";
  els.summary.classList.toggle("muted", !s.text);
  els.topics.textContent = "";
  for (const t of [...(s.urgent ? ["needs attention"] : []), ...(s.topics || [])]) {
    const li = document.createElement("li"); li.textContent = t; if (t === "needs attention") li.className = "alert"; els.topics.appendChild(li);
  }
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
  current.push(tk); lastWordAt = performance.now() / 1000;
  els.liveWord.textContent = tk.text; els.liveWord.classList.remove("pop", "idle"); void els.liveWord.offsetWidth; els.liveWord.classList.add("pop");
  els.meter.style.width = `${Math.round(p * 100)}%`;
  renderGloss(current); showSentence(toEnglish(current).english, true); speech.word(tk.text);
}
async function finishSentence(force = false) {
  if (!current.length) return;
  if (!force && current.every((t) => t.cat === "pronouns")) return;
  const tokens = current; current = [];
  const { english } = toEnglish(tokens);
  if (!english) return;
  const item = { english, tokens };
  history.push(item); showSentence(english, false); speech.sentence(english);
  renderHistory(); renderSummary(); setIdle();
  if (cfg.llm) {
    try {
      const r = await fetch("api/polish", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gloss: tokens.map((t) => t.label), english, history: history.slice(0, -1).map((h) => h.english) }) });
      if (r.ok) { const j = await r.json();
        if (j.sentence) { item.english = j.sentence; if (history[history.length - 1] === item && !current.length) showSentence(j.sentence, false); }
        if (j.summary) item.summary = j.summary; renderHistory(); renderSummary(); }
    } catch { /* keep the rule-based text */ }
  }
}
els.clearBtn.addEventListener("click", () => {
  current = []; history.length = 0; renderGloss([]); renderHistory(); renderSummary(); setIdle();
  els.sentence.innerHTML = '<span class="hint">Cleared. Sign the next sentence.</span>';
});

// ------------------------------------------------------------------ word decoder: only clear wins become words
const dec = { recent: [], lastLabel: null, lastAt: 0, released: true, quiet: 0 };
let testMode = false;
function showGuesses(r) {
  if (!testMode) return;
  els.guesses.textContent = "";
  for (const [i, p] of r.top || []) {
    const li = document.createElement("li"), b = document.createElement("b"), s = document.createElement("span");
    b.textContent = `${Math.round(p * 100)}%`;
    s.textContent = spec.labels[i] === "none" ? "(resting / between signs)" : spec.display[spec.labels[i]] || spec.labels[i];
    li.append(b, s); if (p < threshold || spec.labels[i] === "none") li.className = "low"; els.guesses.appendChild(li);
  }
}
function onResult(r) {
  if (r.top) showGuesses(r);
  if (teach.recording) return;                                   // no words while recording a take
  const p2 = r.top && r.top[1] ? r.top[1][1] : 0;
  // a word must be confident AND clearly ahead of the runner-up: near-ties are where mix-ups happen
  const none = spec.labels[r.best] === "none" || r.p < threshold || r.p - p2 < margin;
  els.meter.style.width = `${Math.round((spec.labels[r.best] === "none" ? 0 : r.p) * 100)}%`;
  if (none) { if (++dec.quiet >= 2) dec.released = true; dec.recent.push(null); if (dec.recent.length > 3) dec.recent.shift(); return; }
  dec.quiet = 0;
  const label = spec.labels[r.best];
  dec.recent.push(label); if (dec.recent.length > 3) dec.recent.shift();
  const votes = dec.recent.filter((l) => l === label).length;
  const now = performance.now() / 1000;
  if (votes >= 2) {                                              // two agreeing readings in a row (~130 ms)
    const inSentence = current.some((t) => t.label === label);
    const repeat = label === dec.lastLabel;
    if ((!repeat && !inSentence) || (dec.released && now - dec.lastAt > 1.5 && !inSentence)) {
      addWord(label, r.p); dec.lastLabel = label; dec.lastAt = now; dec.released = false;
    }
  }
}

// ------------------------------------------------------------------ frame loop
const buffer = [];
let lastVideoTime = -1, frameNo = 0, lastPose = null, inflight = false, scaleIx = 0, runId = 0, lastInfer = 0;
let fps = 0, lastFrameT = 0, modelMs = 0;

function draw(hs, poseNow) {
  const W = els.overlay.width, H = els.overlay.height, s = W / 1280;
  ctx.clearRect(0, 0, W, H);
  if (poseNow) {
    ctx.lineCap = "round"; ctx.lineWidth = 5 * s; ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.beginPath();
    const seg = (a, b) => { if (poseNow[a] && poseNow[b]) { ctx.moveTo(poseNow[a][0] * W, poseNow[a][1] * H); ctx.lineTo(poseNow[b][0] * W, poseNow[b][1] * H); } };
    seg(1, 2); seg(1, 3); seg(3, 5); seg(2, 4); seg(4, 6); ctx.stroke();
  }
  for (const h of hs) {
    const P = (i) => [h.pts[i*3] * W, h.pts[i*3+1] * H];
    ctx.globalAlpha = h.predicted ? 0.45 : 1;
    for (const [lw, c] of [[8 * s, "rgba(14,20,48,.8)"], [4.2 * s, "#ffffff"]]) {
      ctx.lineWidth = lw; ctx.strokeStyle = c; ctx.lineCap = "round"; ctx.beginPath();
      for (const [a, b] of BONES) { const [x1, y1] = P(a), [x2, y2] = P(b); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); }
      ctx.stroke();
    }
    ctx.fillStyle = h.isLeft ? "#FF9F1C" : "#7C8BFF";
    for (let i = 0; i < 21; i++) { const [x, y] = P(i); ctx.beginPath(); ctx.arc(x, y, (i % 4 === 0 ? 6.5 : 5) * s, 0, 7); ctx.fill(); }
    ctx.globalAlpha = 1;
  }
}

function schedule(now) {
  if (!workerReady || inflight || now - lastInfer < INFER_EVERY) return;
  lastInfer = now;
  if (now - lastHandAt >= 0.35) { onResult({ best: spec.labels.length - 1, p: 1, top: null }); return; }
  const span = WINDOWS[scaleIx]; scaleIx = (scaleIx + 1) % WINDOWS.length;
  if (!buffer.length || buffer[0].t > now - span * 0.7) return;
  const win = windowFrom(buffer, now - span, now, T);
  const data = new Float32Array(T * FEATURE_DIM);
  win.forEach((f, i) => data.set(f, i * FEATURE_DIM));
  inflight = true;
  worker.postMessage({ type: "run", id: ++runId, scale: span, data }, [data.buffer]);
}

function loop() {
  const v = els.video;
  if (mode !== "speak" && v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime; frameNo++;
    const nowMs = performance.now(), now = nowMs / 1000;
    if (pose) {                                                  // every frame: arms carry fast movement
      const pr = pose.detectForVideo(v, nowMs), lm = pr.landmarks && pr.landmarks[0];
      lastPose = lm ? POSE_IDS.map((id) => ((lm[id].visibility ?? 1) > 0.4 ? [lm[id].x, lm[id].y] : null)) : null;
      if (lastPose && (!lastPose[1] || !lastPose[2])) lastPose = null;
    }
    const res = hands.detectForVideo(v, nowMs + 0.5);
    const dets = res.landmarks.map((lms, i) => {
      const c = res.handedness && res.handedness[i] && res.handedness[i][0];
      const pLeft = c ? (c.categoryName === "Right" ? c.score : 1 - c.score) : 0.5;   // raw frame: "Right" = signer's left
      return { pts: Float32Array.from(lms.flatMap((p) => [p.x, p.y, p.z])), pLeft };
    });
    const hs = tracker.update(dets, now, { left: lastPose && lastPose[5], right: lastPose && lastPose[6] });
    if (hs.length) lastHandAt = now;
    const { right, left } = assignSlots(hs);
    const aspect = v.videoWidth / v.videoHeight;
    const f = frameFeatures(right, left, lastPose, aspect);
    buffer.push({ t: now, f, raw: { right, left, pose: lastPose, aspect } });
    while (buffer.length && buffer[0].t < now - 3.5) buffer.shift();
    const real = hs.filter((h) => !h.predicted);
    status(els.stHands, real.length ? "on" : hs.length ? "warn" : "off",
      real.length === 2 ? "Both hands" : real.length ? (real[0].isLeft ? "Left hand" : "Right hand") : hs.length ? "Following hand" : "No hands");
    if (frameNo % 15 === 0) {
      const msg = !pose ? null : !lastPose ? "Move back or tilt the camera down: your head and both shoulders must be in view."
        : lastPose[1][1] > 0.92 || lastPose[2][1] > 0.92 ? "Tilt the camera down a little: your shoulders are at the edge." : null;
      els.frameHint.hidden = !msg; if (msg) els.frameHint.textContent = msg;
    }
    draw(hs, lastPose);
    stage.setHands(hs.map((h) => h.pts), aspect);
    teach.tick(now);
    schedule(now);
    if (!teach.recording && current.length && ((now - lastHandAt > SENTENCE_PAUSE) || (now - lastWordAt > IDLE_END && dec.released)))
      finishSentence(now - lastHandAt > 3 || now - lastWordAt > 6);
    if (lastFrameT) fps = fps ? 0.9 * fps + 0.1 / Math.max(now - lastFrameT, 1e-3) : 1 / (now - lastFrameT);
    lastFrameT = now;
    if (frameNo % 30 === 0) { stage.lowPower = fps > 0 && fps < 20; els.perf.textContent = `${Math.round(fps)} fps · model ${modelMs.toFixed(1)} ms (background)`; }
  }
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) v.requestVideoFrameCallback(loop);
  else requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ Teach mode: record your own takes
// library format for the avatar: origin = middle of the shoulders, unit = shoulder width, y down
function libraryFrame(raw) {
  const { right, left, pose: p, aspect: a } = raw;
  if (!p) return null;
  const sx = (p[1][0] + p[2][0]) / 2 * a, sy = (p[1][1] + p[2][1]) / 2;
  const s = Math.hypot((p[1][0] - p[2][0]) * a, p[1][1] - p[2][1]); if (s < 1e-4) return null;
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const hand = (h) => { if (!h) return null; const o = []; for (let i = 0; i < 21; i++) o.push(r3((h[i*3] * a - sx) / s), r3((h[i*3+1] - sy) / s), r3(h[i*3+2] * a / s)); return o; };
  return { r: hand(right), l: hand(left), p: p.flatMap((q) => (q ? [r3((q[0] * a - sx) / s), r3((q[1] - sy) / s)] : [null, null])) };
}
const teach = {
  recording: false, phase: "", until: 0, start: 0, takes: 0, want: 3, label: "",
  begin() {
    const word = els.teachWord.value.trim();
    if (!word) { els.teachState.textContent = "Type or pick the word you will sign."; return; }
    if (!cameraWanted) { els.teachState.textContent = "Start the camera first (Sign → English tab)."; return; }
    this.label = word; this.takes = 0; this.next();
  },
  next() {
    this.phase = "countdown"; this.until = performance.now() / 1000 + 2; this.recording = true;
    els.teachRec.disabled = true; els.recOverlay.hidden = false;
  },
  tick(now) {
    if (!this.recording) return;
    const handsUp = now - lastHandAt < 0.05;
    if (this.phase === "countdown") {
      const left = Math.ceil(this.until - now);
      els.recOverlay.textContent = `Take ${this.takes + 1} of ${this.want}: get ready… ${left}`;
      if (now >= this.until) { this.phase = "wait"; this.until = now + 6; }
    } else if (this.phase === "wait") {                 // wait for the hands to come up
      els.recOverlay.textContent = `Take ${this.takes + 1} of ${this.want}: sign "${this.label}" now`;
      if (handsUp) { this.phase = "rec"; this.start = now - 0.2; this.downSince = 0; this.until = now + 4; }
      else if (now >= this.until) { els.teachState.textContent = "No hands seen. Keep your hands and shoulders in view and try again."; this.finish(); }
    } else if (this.phase === "rec") {                  // record until the hands go down (or 4 s)
      els.recOverlay.textContent = `● Recording take ${this.takes + 1}: lower your hands when the sign is done`;
      if (!handsUp) { this.downSince = this.downSince || now; } else this.downSince = 0;
      if ((this.downSince && now - this.downSince > 0.35) || now >= this.until) this.save(now);
    }
  },
  async save(now) {
    this.phase = "saving";
    const part = buffer.filter((b) => b.t >= this.start && b.t <= now);
    const active = part.filter((b) => b.f[0] > 0.5 || b.f[66] > 0.5).length;
    if (part.length < 10 || active < part.length * 0.4) {
      els.teachState.textContent = "No hands seen in that take. Keep your hands and shoulders in view and try again.";
      return this.finish();
    }
    const span = part[part.length - 1].t - part[0].t || 1;
    const body = { label: this.label, display: this.label, fps: part.length / span, features: part.map((b) => Array.from(b.f, (v) => Math.round(v * 10000) / 10000)),
      raw: { fps: part.length / span, frames: part.map((b) => libraryFrame(b.raw)).filter(Boolean) } };
    try {
      const r = await fetch("api/personal", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "save failed");
      this.takes++;
      els.teachState.textContent = `Saved take ${this.takes} of ${this.want} for "${this.label}".`;
      loadMySigns();
      if (this.takes < this.want) return this.next();
      els.teachState.textContent = `Done: ${this.want} takes of "${this.label}" saved. Run finetune.bat to teach the model (a few minutes).`;
    } catch (e) { els.teachState.textContent = `Could not save: ${e.message}`; }
    this.finish();
  },
  finish() { this.recording = false; this.phase = ""; els.teachRec.disabled = false; els.recOverlay.hidden = true; },
};
els.teachRec.addEventListener("click", () => teach.begin());
async function loadMySigns() {
  try {
    const { items } = await (await fetch("api/personal")).json();
    const by = new Map();
    for (const it of items) { if (!by.has(it.label)) by.set(it.label, []); by.get(it.label).push(it); }
    els.mySigns.textContent = "";
    els.teachCount.textContent = items.length ? `${items.length} takes of ${by.size} words` : "none yet";
    for (const [label, takes] of by) {
      const li = document.createElement("li"), name = document.createElement("span"), del = document.createElement("button");
      const known = spec && spec.labels.includes(label);
      name.textContent = `${takes[0].display} · ${takes.length} take${takes.length > 1 ? "s" : ""}${known ? "" : " · new word"}`;
      del.className = "btn link"; del.type = "button"; del.textContent = "Delete";
      del.addEventListener("click", async () => { for (const t of takes) await fetch(`api/personal/${t.id}`, { method: "DELETE" }); loadMySigns(); });
      li.append(name, del); els.mySigns.appendChild(li);
    }
    if (e2i) e2i.addWords([...by.entries()].map(([label, t]) => ({ label, display: t[0].display })));
    if (avatar) avatar.personalOnly = new Set([...by.keys()].filter((l) => !(spec && spec.labels.includes(l))));
  } catch { /* server not reachable */ }
}

// ------------------------------------------------------------------ English -> Sign
let lastItems = [];
function renderE2S(items, active, okFlags) {
  els.e2sGloss.textContent = "";
  items.forEach((it, i) => {
    const li = document.createElement("li");
    li.textContent = it.kind === "sign" ? it.text : it.text;
    li.className = it.kind === "sign" && (!okFlags || okFlags[i]) ? "sign" : "spell";
    if (it.kind !== "sign" || (okFlags && !okFlags[i])) li.title = "No sign for this word yet: it is shown as text. Teach it in Teach mode.";
    if (active && active === it) li.classList.add("now");
    if (it.kind === "sign") li.addEventListener("click", () => playItems([it]));
    els.e2sGloss.appendChild(li);
  });
}
async function playItems(items) {
  if (!items.length) return;
  const ok = await avatar.play(items);
  renderE2S(items === lastItems ? lastItems : items, null, ok);
  const missing = items.filter((it, i) => it.kind !== "sign" || !ok[i]).map((it) => it.text);
  els.e2sNote.textContent = missing.length ? `Shown as text (no sign yet): ${missing.join(", ")}. You can teach these in Teach mode.` : "";
}
function translateAndPlay(text) {
  if (!e2i) return;
  lastItems = e2i.translate(text);
  if (!lastItems.length) { els.e2sNote.textContent = "Nothing to sign in that sentence."; return; }
  renderE2S(lastItems);
  playItems(lastItems);
}
els.e2sGo.addEventListener("click", () => translateAndPlay(els.e2sInput.value));
els.e2sInput.addEventListener("keydown", (e) => { if (e.key === "Enter") translateAndPlay(els.e2sInput.value); });
els.e2sReplay.addEventListener("click", () => playItems(lastItems));
els.e2sSpeed.addEventListener("input", () => { avatar.speed = Number(els.e2sSpeed.value); els.e2sSpeedVal.textContent = `${avatar.speed.toFixed(2).replace(/0$/, "")}×`; });

const Recog = window.SpeechRecognition || window.webkitSpeechRecognition;
let recog = null, listening = false;
if (!Recog) { els.micBtn.disabled = true; els.micState.textContent = "Speech input needs Chrome or Edge. You can type instead."; }
els.micBtn.addEventListener("click", () => {
  if (!Recog) return;
  if (listening) { recog.stop(); return; }
  recog = new Recog(); recog.lang = "en-IN"; recog.interimResults = true; recog.continuous = false;
  recog.onstart = () => { listening = true; els.micBtn.classList.add("on"); els.micBtn.textContent = "Listening… tap to stop"; els.micState.textContent = ""; };
  recog.onresult = (e) => {
    let text = ""; let final = false;
    for (const r of e.results) { text += r[0].transcript; final = final || r.isFinal; }
    els.e2sInput.value = text;
    if (final) translateAndPlay(text);
  };
  recog.onerror = (e) => { els.micState.textContent = e.error === "not-allowed" ? "Microphone access is blocked. Allow it from the address bar." : e.error === "network" ? "Speech recognition needs an internet connection. You can type instead." : `Speech input stopped (${e.error}).`; };
  recog.onend = () => { listening = false; els.micBtn.classList.remove("on"); els.micBtn.textContent = "Speak"; };
  recog.start();
});

// ------------------------------------------------------------------ modes
let mode = "sign";
function setMode(m) {
  mode = m; document.body.dataset.mode = m;
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  if (m === "speak") avatar.resize(); else stage.resize();
  if (m === "teach" && !teach.recording)
    els.teachState.textContent = cameraWanted ? "Camera is on. Type a word, then select Record 3 takes." : "Select Start camera (on the left), then type a word and record.";
}
document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

// ------------------------------------------------------------------ start-up
async function loadAll() {
  cfg = await (await fetch("api/config")).json();
  threshold = cfg.threshold; margin = cfg.margin;
  els.sens.value = threshold; els.sensVal.textContent = `${Math.round(threshold * 100)}%`;
  if (!cfg.model.ready) { status(els.stModel, "bad", "No model: run train.bat"); throw new Error("No trained model yet. Run train.bat once (it uses your Kaggle dataset), then reload this page."); }
  spec = await (await fetch("static/model/model.json")).json();
  T = spec.T;
  e2i = new EnglishToISL(spec.labels, spec.display, spec.category);
  els.wordList.textContent = "";
  for (const l of spec.labels) if (l !== "none") { const o = document.createElement("option"); o.value = spec.display[l] || l; els.wordList.appendChild(o); }
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") { workerReady = true; status(els.stModel, "on", `${spec.labels.length - 1} signs`); }
    else if (m.type === "error") status(els.stModel, "bad", "Model failed");
    else if (m.type === "result") { inflight = false; modelMs = m.ms; onResult(m); }
  };
  worker.postMessage({ type: "load", spec });
  loadMySigns();
}
async function loadTrackers() {
  const files = await FilesetResolver.forVisionTasks(new URL("../vendor/mediapipe/wasm", import.meta.url).href);
  const make = async (Cls, file, extra) => {
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: new URL(`../models/${file}`, import.meta.url).href, delegate }, runningMode: "VIDEO", ...extra });
    try { return await Cls.createFromOptions(files, opts("GPU")); } catch { return Cls.createFromOptions(files, opts("CPU")); }
  };
  // lower tracking thresholds keep hold of a hand through fast movement; the gap filler covers the rest
  hands = await make(HandLandmarker, "hand_landmarker.task", { numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
  if (cfg.pose) {
    try { pose = await make(PoseLandmarker, "pose_landmarker_lite.task", { numPoses: 1, minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5 }); }
    catch (e) { console.warn("body tracking unavailable:", e); pose = null; }
  }
}

let stream = null, cameraWanted = false;
function sync() {
  const { videoWidth: w, videoHeight: h } = els.video; if (!w || !h) return;
  if (els.overlay.width !== w || els.overlay.height !== h) { els.overlay.width = w; els.overlay.height = h; }
  els.videoBox.style.aspectRatio = `${w} / ${h}`;
  const fr = stream && stream.getVideoTracks()[0] && stream.getVideoTracks()[0].getSettings().frameRate;
  status(els.stCam, "on", `Camera ${Math.min(w, h)}p${fr ? ` · ${Math.round(fr)} fps` : ""}`);
}
els.video.addEventListener("resize", sync);
async function startCamera() {
  try {
    // 60 fps when the webcam can: twice the frames for fast signs, less motion blur
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 24 }, facingMode: "user" } });
  } catch (e) {
    if (e.name === "OverconstrainedError") stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    else if (e.name === "NotAllowedError") throw new Error("Camera access is blocked. Allow the camera (address bar icon), then select Start camera again.");
    else if (e.name === "NotFoundError") throw new Error("No camera found. Connect a webcam, then select Start camera again.");
    else if (e.name === "NotReadableError") throw new Error("Another app is using the camera. Close it, then select Start camera again.");
    else throw e;
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
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
  window.addEventListener("pointermove", (e) => {
    const x = e.clientX / innerWidth - 0.5, y = e.clientY / innerHeight - 0.5;
    els.tilt.style.setProperty("--ry", `${(4 - x * 6).toFixed(2)}deg`); els.tilt.style.setProperty("--rx", `${(2 + y * 4).toFixed(2)}deg`);
  }, { passive: true });
}

stage = new HandStage(els.three);
avatar = new SignAvatar(els.avatar);
avatar.onWord = (it) => {
  els.avatarWord.textContent = !it ? "" : it.kind === "sign" ? it.text : `“${it.text}” (no sign yet)`;
  els.avatarWord.classList.toggle("spelled", !!it && it.kind !== "sign");
  if (lastItems.length) renderE2S(lastItems, it);
};
const ready = loadAll();
const trackersReady = ready.then(loadTrackers);
ready.catch((e) => { els.gateMsg.textContent = e.message; els.e2sNote.textContent = e.message; });
trackersReady.catch(() => {});

let looping = false;
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
    if (!looping) { looping = true; loop(); }
  } catch (e) {
    els.gateMsg.textContent = e.message || String(e);
    els.startBtn.textContent = "Start camera"; els.startBtn.disabled = false;
  }
});
setMode("sign");
