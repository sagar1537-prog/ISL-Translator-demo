// Signing avatar: replays REAL signers' recorded movements (from the dataset, or your own Teach-mode
// takes) on a clean 3D figure: head, shoulders, arms and two full hands (palm + fingers).
// Library frames are normalized: origin = middle of the shoulders, unit = shoulder width, y down.
//
// Recorded keypoints are imperfect, so every clip is cleaned before it is played:
//   - hands the tracker labelled left/right the wrong way round are put back on the right arm
//   - one- or two-frame flickers of a hand are removed, short drop-outs are filled in
//   - jitter is smoothed out (centred, so no lag)
// and playback interpolates between recorded frames, so 25 fps recordings move smoothly on a 60 Hz screen.
import * as THREE from "../vendor/three/three.module.js";

const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const SEGMENTS = FINGERS.flatMap((f) => [[0, f[0]], [f[0], f[1]], [f[1], f[2]], [f[2], f[3]]].slice(f[0] === 1 ? 0 : 1));
const PALM = [0, 1, 5, 9, 13, 17];                       // palm outline (fan from the wrist)
const NOSE = 0, LSH = 1, RSH = 2, LEL = 3, REL = 4, LWR = 5, RWR = 6;
const K = 1.15;                                          // world units per shoulder width
const BLEND = 8;                                         // frames to glide from one sign into the next
const REST = { p: [0, -1.05, 0.5, 0, -0.5, 0, 0.56, 0.95, -0.56, 0.95, 0.42, 1.75, -0.42, 1.75], r: null, l: null };
const v3 = (x, y, z = 0) => new THREE.Vector3(x * K, -y * K, -z * K);
// Recorded hand depth is relative to the wrist, not the body: signing hands are held in front of the
// chest, so hands (and the forearm ends) are placed this far in front of the torso.
const HAND_Z = 0.62 * K, ELBOW_Z = 0.22 * K;
const vh = (x, y, z = 0) => new THREE.Vector3(x * K, -y * K, -z * K + HAND_Z);

// ------------------------------------------------------------------ clip cleaning
const wristOf = (h) => [h[0], h[1]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function cleanClip(frames) {
  const n = frames.length;
  if (!n) return frames;
  const P = frames.map((f) => (f.p || []).slice());
  // 1. body points: fill unseen values from the neighbours
  for (let k = 0; k < 14; k++) fillSeries(P.map((p) => p[k]), (i, v) => { P[i][k] = v; }, Infinity, REST.p[k]);
  let R = frames.map((f) => (f.r && f.r.length === 63 ? f.r.slice() : null));
  let L = frames.map((f) => (f.l && f.l.length === 63 ? f.l.slice() : null));
  // 2. left/right put back on the correct arm (closest wrist + continuity with the previous frame)
  let pr = null, pl = null;
  for (let i = 0; i < n; i++) {
    const rw = [P[i][RWR * 2], P[i][RWR * 2 + 1]], lw = [P[i][LWR * 2], P[i][LWR * 2 + 1]];
    const cost = (r, l) => (r ? dist(wristOf(r), rw) + (pr ? 0.5 * dist(wristOf(r), pr) : 0) : 0)
                         + (l ? dist(wristOf(l), lw) + (pl ? 0.5 * dist(wristOf(l), pl) : 0) : 0);
    if ((R[i] || L[i]) && cost(L[i], R[i]) + 0.15 < cost(R[i], L[i])) [R[i], L[i]] = [L[i], R[i]];
    if (R[i]) pr = wristOf(R[i]);
    if (L[i]) pl = wristOf(L[i]);
  }
  // 3. remove flickers (a hand seen for 1-2 frames inside a long absence), 4. fill short drop-outs
  R = repairHand(R); L = repairHand(L);
  // 5. smooth jitter: centred 5-tap kernel, only inside runs where the hand is present
  smoothRuns(R); smoothRuns(L);
  smoothPose(P);
  return frames.map((_, i) => ({ p: P[i], r: R[i], l: L[i] }));
}

function fillSeries(vals, set, maxGap, fallback) {
  const n = vals.length, known = [];
  vals.forEach((v, i) => { if (v !== null && v !== undefined && Number.isFinite(v)) known.push(i); });
  if (!known.length) { for (let i = 0; i < n; i++) set(i, fallback); return; }
  for (let i = 0; i < n; i++) {
    if (vals[i] !== null && vals[i] !== undefined && Number.isFinite(vals[i])) continue;
    let a = -1, b = -1;
    for (const k of known) { if (k < i) a = k; else { b = k; break; } }
    if (a < 0) set(i, vals[b]); else if (b < 0) set(i, vals[a]);
    else if (b - a - 1 <= maxGap) set(i, vals[a] + (vals[b] - vals[a]) * (i - a) / (b - a));
  }
}

function runs(arr) {
  const out = []; let s = -1;
  for (let i = 0; i <= arr.length; i++) {
    const on = i < arr.length && !!arr[i];
    if (on && s < 0) s = i;
    if (!on && s >= 0) { out.push([s, i - 1]); s = -1; }
  }
  return out;
}

function repairHand(H) {
  const n = H.length;
  // flickers: runs of <= 2 frames with >= 3 empty frames on both sides (or at the clip edge)
  for (const [a, b] of runs(H)) {
    if (b - a + 1 > 2) continue;
    const before = a === 0 ? Infinity : a - (runs(H.slice(0, a)).pop()?.[1] ?? -1) - 1;
    const after = b === n - 1 ? Infinity : (runs(H.slice(b + 1))[0]?.[0] ?? Infinity);
    if (before >= 3 && after >= 3) for (let i = a; i <= b; i++) H[i] = null;
  }
  // drop-outs of up to 10 frames between two sightings: interpolate every coordinate
  const rs = runs(H);
  for (let k = 0; k + 1 < rs.length; k++) {
    const a = rs[k][1], b = rs[k + 1][0], gap = b - a - 1;
    if (gap < 1 || gap > 10) continue;
    for (let i = a + 1; i < b; i++) {
      const t = (i - a) / (b - a);
      H[i] = H[a].map((v, j) => v + (H[b][j] - v) * t);
    }
  }
  return H;
}

function smoothRuns(H) {
  const w = [1, 4, 6, 4, 1];
  for (const [a, b] of runs(H)) {
    if (b - a < 2) continue;
    const src = H.slice(a, b + 1).map((h) => h.slice());
    for (let i = a; i <= b; i++) {
      const out = new Array(63).fill(0); let ws = 0;
      for (let k = -2; k <= 2; k++) {
        const j = i + k; if (j < a || j > b) continue;
        const s = src[j - a]; ws += w[k + 2];
        for (let c = 0; c < 63; c++) out[c] += w[k + 2] * s[c];
      }
      H[i] = out.map((v) => v / ws);
    }
  }
}

function smoothPose(P) {
  const w = [1, 4, 6, 4, 1], src = P.map((p) => p.slice()), n = P.length;
  for (let i = 0; i < n; i++) for (let c = 0; c < 14; c++) {
    let s = 0, ws = 0;
    for (let k = -2; k <= 2; k++) { const j = i + k; if (j < 0 || j >= n) continue; s += w[k + 2] * src[j][c]; ws += w[k + 2]; }
    P[i][c] = s / ws;
  }
}

function mix(a, b, t, ease = true) {
  const e = ease ? t * t * (3 - 2 * t) : t;
  const p = a.p.map((v, i) => v + (b.p[i] - v) * e);
  // a hand that is only on one side travels with its arm's wrist instead of floating loose
  const hand = (ha, hb, w) => {
    if (ha && hb) return ha.map((v, i) => v + (hb[i] - v) * e);
    const h = ha || hb; if (!h) return null;
    if ((ha && e > 0.75) || (hb && e < 0.25)) return null;      // fade window
    const dx = p[w * 2] - h[0], dy = p[w * 2 + 1] - h[1], k = ha ? e : 1 - e;
    return h.map((v, i) => (i % 3 === 0 ? v + dx * k : i % 3 === 1 ? v + dy * k : v));
  };
  return { p, r: hand(a.r, b.r, RWR), l: hand(a.l, b.l, LWR) };
}

// ------------------------------------------------------------------ the figure
export class SignAvatar {
  constructor(canvas) {
    this.canvas = canvas;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
    this.camera.position.set(0, -0.15, 5.4);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xc9d1ff, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(1.5, 2.5, 4); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xffb85c, 1.2); rim.position.set(-3, 0, -2); this.scene.add(rim);

    const skin = new THREE.MeshStandardMaterial({ color: 0x2337c6, roughness: 0.45, metalness: 0.1 });
    const palmMat = new THREE.MeshStandardMaterial({ color: 0x3348d6, roughness: 0.5, side: THREE.DoubleSide });
    const body = new THREE.MeshStandardMaterial({ color: 0xdfe5fb, roughness: 0.6 });
    const tipMat = new THREE.MeshStandardMaterial({ color: 0xff9f1c, roughness: 0.3, emissive: 0x6b3a00, emissiveIntensity: 0.25 });
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.42 * K, 32, 24), body);
    this.torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.42 * K, 1.0 * K, 8, 24), body);
    this.neck = new THREE.Mesh(new THREE.CylinderGeometry(0.13 * K, 0.15 * K, 1, 16), body);
    this.eyes = [0, 1].map(() => { const e = new THREE.Mesh(new THREE.SphereGeometry(0.05 * K, 12, 8), new THREE.MeshStandardMaterial({ color: 0x0e1430 })); this.scene.add(e); return e; });
    this.scene.add(this.head, this.torso, this.neck);
    const limb = () => { const m = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 16), skin); m.frustumCulled = false; m.matrixAutoUpdate = false; this.scene.add(m); return m; };
    this.limbs = [0, 1, 2, 3].map(limb);
    this.elbows = [0, 1].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), skin); this.scene.add(m); return m; });
    this.shoulders = [0, 1].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.13 * K, 16, 12), skin); this.scene.add(m); return m; });
    this.hands = [0, 1].map(() => {
      const seg = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10), skin, SEGMENTS.length);
      const knuckles = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 10), skin, 21);
      const tips = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 10), tipMat, 5);
      const palmGeo = new THREE.BufferGeometry();
      palmGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(PALM.length * 3), 3));
      palmGeo.setIndex([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]);
      const palm = new THREE.Mesh(palmGeo, palmMat);
      for (const m of [seg, knuckles, tips, palm]) { m.frustumCulled = false; this.scene.add(m); }
      return { seg, knuckles, tips, palm, vis: 0, pts: null };
    });
    this.tmp = { m: new THREE.Matrix4(), q: new THREE.Quaternion(), s: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0) };

    this.cache = new Map();
    this.frames = []; this.pos = 0; this.speed = 1; this.playing = false; this.onWord = null; this.onEnd = null;
    this.cur = REST; this.t0 = performance.now(); this.lastT = performance.now(); this.lastItem = undefined;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const loop = (t) => { if (!document.hidden) this.step(t); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth || 1, h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h;
    // frame head to waist (and both hands at full reach) whatever the box shape
    const half = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const needV = 1.8 * K, needH = 1.55 * K;                  // half-height / half-width to keep in view
    const z = Math.max(needV / half, needH / (half * this.camera.aspect)) + 0.4;
    this.camera.position.set(0, -0.3 * K, z); this.camera.lookAt(0, -0.3 * K, 0);
    this.camera.updateProjectionMatrix();
  }

  async _load(label) {
    if (this.cache.has(label)) return this.cache.get(label);
    let data = null;
    const urls = [`static/signs/${label}.json`, `api/personal/sign/${label}`];
    if (this.personalOnly && this.personalOnly.has(label)) urls.reverse();
    for (const url of urls) {
      try { const r = await fetch(url); if (r.ok) { data = await r.json(); break; } } catch { /* next */ }
    }
    if (data && data.frames && data.frames.length) data = { ...data, frames: cleanClip(data.frames) };
    this.cache.set(label, data);
    return data;
  }

  /** items: [{label, text, kind}] from EnglishToISL. Plays them in order. Returns which ones have a sign. */
  async play(items) {
    this.stop();
    const seq = [];
    for (const it of items) {
      const d = it.kind === "sign" ? await this._load(it.label) : null;
      seq.push({ item: it, data: d && d.frames && d.frames.length ? d : null });
    }
    // one timeline: glide into each sign, play it at its own frame rate, glide back to rest at the end
    const frames = [];
    let prev = this.cur;
    for (const { item, data } of seq) {
      const fps = data ? data.fps || 25 : 25;
      const first = data ? data.frames[0] : REST;
      for (let k = 1; k <= BLEND; k++) frames.push({ f: mix(prev, first, k / BLEND), dt: 1 / 30, item });
      if (data) for (const fr of data.frames) frames.push({ f: fr, dt: 1 / fps, item });
      else for (let k = 0; k < Math.round(fps * 0.8); k++) frames.push({ f: REST, dt: 1 / fps, item });
      prev = data ? data.frames[data.frames.length - 1] : REST;
    }
    for (let k = 1; k <= BLEND * 2; k++) frames.push({ f: mix(prev, REST, k / (BLEND * 2)), dt: 1 / 30, item: null });
    // cumulative times, so playback can sample between recorded frames
    let t = 0;
    for (const fr of frames) { fr.t = t; t += fr.dt; }
    this.frames = frames; this.duration = t; this.pos = 0; this.playing = true; this.lastItem = undefined;
    this.lastT = performance.now();
    return seq.map((s) => !!s.data);
  }

  stop() { this.playing = false; this.frames = []; }

  step(t) {
    const dt = Math.min(0.1, Math.max(0, (t - this.lastT) / 1000)); this.lastT = t;
    if (this.playing && this.frames.length) {
      this.pos += dt * this.speed;
      const fr = this.frames;
      let i = this._i || 0;
      if (i >= fr.length || fr[i].t > this.pos) i = 0;
      while (i + 1 < fr.length && fr[i + 1].t <= this.pos) i++;
      this._i = i;
      const a = fr[i], b = fr[Math.min(i + 1, fr.length - 1)];
      const u = b === a ? 0 : Math.min(1, (this.pos - a.t) / (b.t - a.t));
      this.cur = mix(a.f, b.f, u, false);                    // smooth between recorded frames
      if (a.item !== this.lastItem) { this.lastItem = a.item; if (this.onWord) this.onWord(a.item); }
      if (this.pos >= this.duration) {
        this.playing = false; this.cur = REST;
        if (this.lastItem !== null && this.onWord) this.onWord(null);
        if (this.onEnd) this.onEnd();
      }
    }
    this.render((t - this.t0) / 1000, dt);
  }

  _cyl(mesh, a, b, r, instance = -1) {
    const { m, q, s, up } = this.tmp;
    const d = b.clone().sub(a), len = Math.max(d.length(), 1e-4);
    q.setFromUnitVectors(up, d.divideScalar(len));
    m.compose(a.clone().add(b).multiplyScalar(0.5), q, s.set(r, len, r));
    if (instance < 0) { mesh.matrix.copy(m); mesh.matrixWorldNeedsUpdate = true; } else mesh.setMatrixAt(instance, m);
  }

  _hand(h, src, dt, wristTarget) {
    // fade a hand in/out instead of popping; keep its last shape while it fades
    h.vis += ((src ? 1 : 0) - h.vis) * Math.min(1, dt * 12);
    if (src) h.pts = src;
    else if (h.pts && wristTarget) {                           // fading out: follow the arm
      const dx = wristTarget[0] - h.pts[0], dy = wristTarget[1] - h.pts[1];
      h.pts = h.pts.map((v, i) => (i % 3 === 0 ? v + dx : i % 3 === 1 ? v + dy : v));
    }
    const show = h.vis > 0.02 && h.pts;
    for (const m of [h.seg, h.knuckles, h.tips, h.palm]) m.visible = !!show;
    if (!show) return null;
    // hands drawn 25% larger around the wrist (as signing avatars do) so handshapes read at a glance
    const S = 1.25, w0 = h.pts;
    const P = []; for (let i = 0; i < 21; i++) P.push(vh(w0[0] + (w0[i * 3] - w0[0]) * S, w0[1] + (w0[i * 3 + 1] - w0[1]) * S, w0[i * 3 + 2] * S));
    // thickness follows the hand's own size, so every recording looks like a real hand
    const palm = (P[0].distanceTo(P[5]) + P[0].distanceTo(P[9]) + P[0].distanceTo(P[17])) / 3;
    const r = Math.max(0.018, Math.min(0.06, palm * 0.16)) * h.vis;
    const { m, q, s } = this.tmp;
    SEGMENTS.forEach(([a, b], i) => this._cyl(h.seg, P[a], P[b], r * (b % 4 === 0 ? 0.8 : 1), i));
    for (let i = 0; i < 21; i++) { const rr = i === 0 ? r * 1.4 : r * (i % 4 === 0 ? 0.8 : 1); m.compose(P[i], q.identity(), s.set(rr, rr, rr)); h.knuckles.setMatrixAt(i, m); }
    [4, 8, 12, 16, 20].forEach((i, k) => { const rr = r * 0.95; m.compose(P[i], q.identity(), s.set(rr, rr, rr)); h.tips.setMatrixAt(k, m); });
    const pos = h.palm.geometry.attributes.position;
    PALM.forEach((idx, k) => pos.setXYZ(k, P[idx].x, P[idx].y, P[idx].z));
    pos.needsUpdate = true; h.palm.geometry.computeVertexNormals(); h.palm.geometry.computeBoundingSphere();
    h.seg.instanceMatrix.needsUpdate = h.knuckles.instanceMatrix.needsUpdate = h.tips.instanceMatrix.needsUpdate = true;
    return P[0];
  }

  render(time, dt = 1 / 60) {
    const f = this.cur, p = f.p, P = (i) => v3(p[i * 2], p[i * 2 + 1], 0);
    const breathe = this.reduced ? 0 : Math.sin(time * 1.4) * 0.01;
    const ls = P(LSH), rs = P(RSH), mid = ls.clone().add(rs).multiplyScalar(0.5), nose = P(NOSE);
    const head = new THREE.Vector3(nose.x, Math.max(nose.y, mid.y + 0.75 * K) + breathe, 0);
    this.head.position.copy(head);
    this.eyes[0].position.set(head.x - 0.15 * K, head.y + 0.05 * K, 0.38 * K);
    this.eyes[1].position.set(head.x + 0.15 * K, head.y + 0.05 * K, 0.38 * K);
    this.neck.position.set(mid.x, mid.y + 0.25 * K, 0); this.neck.scale.set(1, 0.5 * K, 1);
    this.torso.position.set(mid.x, mid.y - 0.62 * K + breathe, -0.05);
    this.torso.scale.set(Math.max(ls.distanceTo(rs) / K, 0.8), 1, 0.6);
    this.shoulders[0].position.copy(ls); this.shoulders[1].position.copy(rs);
    // hands first, so each forearm ends exactly at its hand's wrist
    const wl = this._hand(this.hands[1], f.l, dt, [p[LWR * 2], p[LWR * 2 + 1]]);
    const wr = this._hand(this.hands[0], f.r, dt, [p[RWR * 2], p[RWR * 2 + 1]]);
    [[LSH, LEL, LWR, wl, this.hands[1]], [RSH, REL, RWR, wr, this.hands[0]]].forEach(([s, e, w, handWrist, h], n) => {
      const sp = P(s), ep = P(e), pw = P(w);
      ep.z += ELBOW_Z; pw.z += HAND_Z * 0.85;
      const wp = handWrist ? pw.clone().lerp(handWrist, Math.min(1, h.vis)) : pw;
      this._cyl(this.limbs[n * 2], sp, ep, 0.1 * K);
      this._cyl(this.limbs[n * 2 + 1], ep, wp, 0.085 * K);
      this.elbows[n].position.copy(ep); this.elbows[n].scale.setScalar(0.095 * K);
    });
    this.scene.rotation.y = this.reduced ? 0 : Math.sin(time * 0.3) * 0.08;
    this.renderer.render(this.scene, this.camera);
  }
}
