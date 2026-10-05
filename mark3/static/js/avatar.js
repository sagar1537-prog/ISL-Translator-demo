// Signing avatar: replays REAL signers' recorded movements (from the dataset, or your own Teach-mode
// takes) on a clean 3D figure - head, shoulders, arms and both hands with all 21 joints each.
// Library frames are normalized: origin = middle of the shoulders, unit = shoulder width, y down.
import * as THREE from "../vendor/three/three.module.js";

const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
               [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const NOSE = 0, LSH = 1, RSH = 2, LEL = 3, REL = 4, LWR = 5, RWR = 6;
const K = 1.15;                                    // world units per shoulder width
const BLEND = 7;                                   // frames to glide from one sign into the next

// a relaxed pose with the hands resting in front of the body
const REST = { p: [0, -1.05, 0.5, 0, -0.5, 0, 0.56, 0.95, -0.56, 0.95, 0.42, 1.75, -0.42, 1.75], r: null, l: null };

const v3 = (x, y, z = 0) => new THREE.Vector3(x * K, -y * K, -z * K);

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
    const body = new THREE.MeshStandardMaterial({ color: 0xdfe5fb, roughness: 0.6 });
    const joint = new THREE.MeshStandardMaterial({ color: 0xff9f1c, roughness: 0.3, emissive: 0x6b3a00, emissiveIntensity: 0.25 });
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.42 * K, 32, 24), body);
    this.torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.42 * K, 1.0 * K, 8, 24), body);
    this.neck = new THREE.Mesh(new THREE.CylinderGeometry(0.13 * K, 0.15 * K, 1, 16), body);
    this.eyes = [0, 1].map(() => { const e = new THREE.Mesh(new THREE.SphereGeometry(0.05 * K, 12, 8), new THREE.MeshStandardMaterial({ color: 0x0e1430 })); this.scene.add(e); return e; });
    this.scene.add(this.head, this.torso, this.neck);
    this.limbs = [0, 1, 2, 3].map(() => { const m = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 16), skin); m.frustumCulled = false; m.matrixAutoUpdate = false; this.scene.add(m); return m; });
    this.shoulders = [0, 1].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.13 * K, 16, 12), skin); this.scene.add(m); return m; });
    this.hands = [0, 1].map(() => {
      const j = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 10), joint, 21);
      const b = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 8), skin, BONES.length);
      j.frustumCulled = false; b.frustumCulled = false;   // instances move every frame: never cull them
      this.scene.add(j, b);
      return { j, b, last: null };
    });
    this.tmp = { m: new THREE.Matrix4(), q: new THREE.Quaternion(), s: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0) };

    this.cache = new Map();
    this.queue = []; this.frames = []; this.fi = 0; this.speed = 1; this.playing = false; this.onWord = null; this.onEnd = null;
    this.cur = REST; this.t0 = performance.now(); this.lastT = 0; this.acc = 0;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const loop = (t) => { if (!document.hidden) this.step(t); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth || 1, h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  async _load(label) {
    if (this.cache.has(label)) return this.cache.get(label);
    let data = null;
    const urls = [`static/signs/${label}.json`, `api/personal/sign/${label}`];
    if (this.personalOnly && this.personalOnly.has(label)) urls.reverse();   // your new words: no library file yet
    for (const url of urls) {
      try { const r = await fetch(url); if (r.ok) { data = await r.json(); break; } } catch { /* next */ }
    }
    this.cache.set(label, data);
    return data;
  }

  /** items: [{label, text, kind}] from EnglishToISL. Plays them in order. */
  async play(items) {
    this.stop();
    const seq = [];
    for (const it of items) {
      const d = it.kind === "sign" ? await this._load(it.label) : null;
      seq.push({ item: it, data: d && d.frames && d.frames.length ? d : null });
    }
    // build one frame list with smooth glides between signs; words without a sign hold a pause with a caption
    const frames = [];
    let prev = this.cur;
    for (const { item, data } of seq) {
      const fps = data ? data.fps || 25 : 25;
      const src = data ? data.frames : Array(Math.round(fps * 0.9)).fill(null);
      const first = data ? fill(data.frames[0], prev) : REST;
      for (let k = 1; k <= BLEND; k++) frames.push({ f: mix(prev, first, k / BLEND), dt: 1 / 30, item, start: k === 1 });
      let last = first;
      for (const fr of src) { last = fr ? fill(fr, last) : last; frames.push({ f: last, dt: 1 / fps, item }); }
      prev = last;
    }
    for (let k = 1; k <= BLEND * 2; k++) frames.push({ f: mix(prev, REST, k / (BLEND * 2)), dt: 1 / 30, item: null });
    this.frames = frames; this.fi = 0; this.acc = 0; this.playing = true; this.lastT = performance.now();
    return seq.map((s) => !!s.data);
  }

  stop() { this.playing = false; this.frames = []; }

  step(t) {
    const dt = Math.min(0.1, (t - this.lastT) / 1000); this.lastT = t;
    if (this.playing && this.frames.length) {
      this.acc += dt * this.speed;
      while (this.fi < this.frames.length && this.acc >= this.frames[this.fi].dt) {
        this.acc -= this.frames[this.fi].dt;
        const fr = this.frames[this.fi];
        if (fr.item && (fr.start || this.fi === 0) && this.onWord) this.onWord(fr.item);
        this.cur = fr.f; this.fi++;
      }
      if (this.fi >= this.frames.length) { this.playing = false; if (this.onWord) this.onWord(null); if (this.onEnd) this.onEnd(); }
    }
    this.render((t - this.t0) / 1000);
  }

  _limb(mesh, a, b, r) {
    const { m, q, s, up } = this.tmp;
    const d = b.clone().sub(a), len = Math.max(d.length(), 1e-4);
    q.setFromUnitVectors(up, d.normalize());
    m.compose(a.clone().add(b).multiplyScalar(0.5), q, s.set(r, len, r));
    mesh.matrix.copy(m); mesh.matrixWorldNeedsUpdate = true;
  }

  render(time) {
    const f = this.cur, p = f.p, P = (i) => v3(p[i * 2], p[i * 2 + 1], 0);
    const breathe = this.reduced ? 0 : Math.sin(time * 1.4) * 0.01;
    const ls = P(LSH), rs = P(RSH), mid = ls.clone().add(rs).multiplyScalar(0.5);
    const nose = P(NOSE);
    // head sits above the shoulders, following the nose sideways
    const head = new THREE.Vector3(nose.x, Math.max(nose.y, mid.y + 0.75 * K) + breathe, 0);
    this.head.position.copy(head);
    this.eyes[0].position.set(head.x - 0.15 * K, head.y + 0.05 * K, 0.38 * K);
    this.eyes[1].position.set(head.x + 0.15 * K, head.y + 0.05 * K, 0.38 * K);
    this.neck.position.set(mid.x, mid.y + 0.25 * K, 0); this.neck.scale.set(1, 0.5 * K, 1);
    this.torso.position.set(mid.x, mid.y - 0.62 * K + breathe, -0.05);
    this.torso.scale.set(Math.max(ls.distanceTo(rs) / K, 0.8), 1, 0.6);
    this.shoulders[0].position.copy(ls); this.shoulders[1].position.copy(rs);
    // arms: shoulder -> elbow -> wrist (the hand's own wrist when the hand is visible)
    const sides = [[LSH, LEL, LWR, f.l], [RSH, REL, RWR, f.r]];
    sides.forEach(([s, e, w, hand], n) => {
      const sp = P(s), ep = P(e), wp = hand ? v3(hand[0], hand[1], hand[2]) : P(w);
      this._limb(this.limbs[n * 2], sp, ep, 0.1 * K);
      this._limb(this.limbs[n * 2 + 1], ep, wp, 0.085 * K);
    });
    // hands
    [f.r, f.l].forEach((hand, n) => {
      const h = this.hands[n];
      h.j.visible = h.b.visible = !!hand;
      if (!hand) return;
      const { m, q, s, up } = this.tmp, pts = [];
      for (let i = 0; i < 21; i++) pts.push(v3(hand[i * 3], hand[i * 3 + 1], hand[i * 3 + 2]));
      pts.forEach((pt, i) => { const r = (i === 0 ? 0.075 : 0.045) * K; m.compose(pt, q.identity(), s.set(r, r, r)); h.j.setMatrixAt(i, m); });
      BONES.forEach(([a, b], i) => {
        const d = pts[b].clone().sub(pts[a]), len = Math.max(d.length(), 1e-4);
        q.setFromUnitVectors(up, d.normalize());
        m.compose(pts[a].clone().add(pts[b]).multiplyScalar(0.5), q, s.set(0.028 * K, len, 0.028 * K));
        h.b.setMatrixAt(i, m);
      });
      h.j.instanceMatrix.needsUpdate = true; h.b.instanceMatrix.needsUpdate = true;
    });
    this.scene.rotation.y = this.reduced ? 0 : Math.sin(time * 0.3) * 0.08;
    this.renderer.render(this.scene, this.camera);
  }
}

// fill missing pose points from the previous frame so the figure never jumps
function fill(fr, prev) {
  const p = fr.p.map((v, i) => (v === null || v === undefined ? prev.p[i] : v));
  return { p, r: fr.r || null, l: fr.l || null };
}
function mix(a, b, t) {
  const e = t * t * (3 - 2 * t);                    // smooth step
  const lerp = (x, y) => (x === null || y === null ? (e < 0.5 ? x : y) : x + (y - x) * e);
  const p = a.p.map((v, i) => lerp(v, b.p[i]));
  const hand = (ha, hb) => (ha && hb ? ha.map((v, i) => v + (hb[i] - v) * e) : e < 0.5 ? ha : hb);
  return { p, r: hand(a.r, b.r), l: hand(a.l, b.l) };
}
