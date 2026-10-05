// 3D stage: your hands rebuilt in 3D from the tracked landmarks (lapis bones, saffron joints),
// floating over a few slow shapes. With no hands in view, an idle hand waves gently.
import * as THREE from "../vendor/three/three.module.js";

const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
               [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TIPS = new Set([4, 8, 12, 16, 20]);

function idleHand() {          // an open right hand, palm facing the viewer, in image-like coords
  const p = new Float32Array(63), set = (i, x, y) => { p[i*3] = 0.5 + x; p[i*3+1] = 0.62 + y; p[i*3+2] = 0; };
  set(0, 0, 0);
  const fingers = [[1, -0.075, -0.035, -0.6], [5, -0.04, -0.12, -0.2], [9, 0, -0.13, 0], [13, 0.035, -0.12, 0.15], [17, 0.065, -0.1, 0.32]];
  for (const [base, bx, by, ang] of fingers) {
    set(base, bx, by);
    const len = base === 1 ? 0.045 : base === 17 ? 0.04 : 0.05;
    for (let k = 1; k <= 3; k++) set(base + k, bx + Math.sin(ang) * len * k, by - Math.cos(ang) * len * k);
  }
  return p;
}

export class HandStage {
  constructor(canvas) {
    this.canvas = canvas;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 50);
    this.camera.position.set(0, 0, 6.2);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xc9d1ff, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(2, 3, 4); this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xffb85c, 1.4); rim.position.set(-3, -1, -2); this.scene.add(rim);

    // slow background shapes
    this.deco = new THREE.Group();
    const decoMat = new THREE.MeshStandardMaterial({ color: 0x2337c6, roughness: 0.35, metalness: 0.1, transparent: true, opacity: 0.12 });
    const shapes = [[new THREE.TorusGeometry(0.9, 0.22, 24, 64), -2.4, 1.2, -3],
                    [new THREE.IcosahedronGeometry(0.75, 0), 2.6, -1.0, -2.5],
                    [new THREE.TorusKnotGeometry(0.45, 0.12, 96, 12), 2.2, 1.6, -4]];
    for (const [g, x, y, z] of shapes) { const m = new THREE.Mesh(g, decoMat); m.position.set(x, y, z); this.deco.add(m); }
    this.scene.add(this.deco);

    // two hands: instanced joints and bones
    this.hands = [0, 1].map(() => {
      const joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12),
        new THREE.MeshStandardMaterial({ color: 0xff9f1c, roughness: 0.3, metalness: 0.05, emissive: 0x6b3a00, emissiveIntensity: 0.25 }), 21);
      const bones = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10),
        new THREE.MeshStandardMaterial({ color: 0x2337c6, roughness: 0.4, metalness: 0.15 }), BONES.length);
      joints.frustumCulled = false; bones.frustumCulled = false;
      const g = new THREE.Group(); g.add(joints, bones); this.scene.add(g);
      return { joints, bones, group: g, cur: null, vis: 0 };
    });
    this.idle = idleHand();
    this.targets = [null, null];
    this.tmp = { m: new THREE.Matrix4(), q: new THREE.Quaternion(), s: new THREE.Vector3(), a: new THREE.Vector3(), b: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0) };
    this.t0 = performance.now();
    this.view = { cx: 0.5, cy: 0.56, k: 5.2 };   // eased auto-framing
    this.lowPower = false; this.tick = 0;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    // on slow machines draw every other frame, so hand tracking keeps full speed
    const loop = () => { if (!document.hidden && (!this.lowPower || (this.tick++ & 1))) this.render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth || 1, h = this.canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.aspect = w / h;
  }

  /** hands: up to two Float32Array(63) in raw camera coords (un-mirrored), aspect = camera width/height */
  setHands(hands, aspect) { this.targets = [hands[0] || null, hands[1] || null]; this.camAspect = aspect || 16 / 9; }

  _toWorld(p, i, out, centerX, centerY, k) {
    // mirror x for a selfie view; z towards the viewer
    out.set(-(p[i*3] - centerX) * this.camAspect * k, -(p[i*3+1] - centerY) * k, -p[i*3+2] * this.camAspect * k);
    return out;
  }

  render() {
    const t = (performance.now() - this.t0) / 1000, mo = this.reduced ? 0 : 1;
    const any = this.targets[0] || this.targets[1];
    // frame the visible hands: centre on them and zoom so they fill about two thirds of the stage
    let tx = 0.5, ty = 0.56, tk = 5.2;
    if (any) {
      let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
      for (const p of this.targets) if (p) for (let i = 0; i < 21; i++) {
        x0 = Math.min(x0, p[i*3]); x1 = Math.max(x1, p[i*3]); y0 = Math.min(y0, p[i*3+1]); y1 = Math.max(y1, p[i*3+1]);
      }
      tx = (x0 + x1) / 2; ty = (y0 + y1) / 2 + 0.02;
      const span = Math.max((x1 - x0) * this.camAspect, y1 - y0, 0.08);
      tk = Math.min(Math.max(2.6 / span, 3), 22);
    }
    const v = this.view; v.cx += (tx - v.cx) * 0.15; v.cy += (ty - v.cy) * 0.15; v.k += (tk - v.k) * 0.12;
    const k = v.k;
    this.hands.forEach((h, n) => {
      let target = this.targets[n];
      if (!any && n === 0) {              // idle: a gently waving hand
        target = Float32Array.from(this.idle);
        const wave = Math.sin(t * 1.6) * 0.035 * mo;
        for (let i = 1; i < 21; i++) { target[i*3] += wave * (0.5 - (target[i*3+1] - 0.62) * 4); }
      }
      h.vis += ((target ? 1 : 0) - h.vis) * 0.2;
      if (target) {
        if (!h.cur) h.cur = Float32Array.from(target);
        for (let i = 0; i < 63; i++) h.cur[i] += (target[i] - h.cur[i]) * 0.55;   // ease toward the new pose
      }
      h.group.visible = h.vis > 0.02 && !!h.cur;
      if (!h.group.visible) return;
      const { m, q, s, a, b, up } = this.tmp, r = 0.075 * Math.max(h.vis, 0.001);
      const cx = v.cx, cy = v.cy;
      for (let i = 0; i < 21; i++) {
        this._toWorld(h.cur, i, a, cx, cy, k);
        const rr = (i === 0 ? 1.35 : TIPS.has(i) ? 1.15 : 0.95) * r;
        m.compose(a, q.identity(), s.set(rr, rr, rr)); h.joints.setMatrixAt(i, m);
      }
      BONES.forEach(([i, j], n2) => {
        this._toWorld(h.cur, i, a, cx, cy, k); this._toWorld(h.cur, j, b, cx, cy, k);
        const len = a.distanceTo(b); const mid = a.clone().add(b).multiplyScalar(0.5);
        q.setFromUnitVectors(up, b.clone().sub(a).normalize());
        m.compose(mid, q, s.set(r * 0.42, len, r * 0.42)); h.bones.setMatrixAt(n2, m);
      });
      h.joints.instanceMatrix.needsUpdate = true; h.bones.instanceMatrix.needsUpdate = true;
    });
    // gentle drift so the scene feels alive
    this.scene.rotation.y = Math.sin(t * 0.35) * 0.18 * mo;
    this.scene.rotation.x = Math.sin(t * 0.27) * 0.05 * mo;
    this.deco.children.forEach((d, i) => { d.rotation.x = t * (0.1 + i * 0.05) * mo; d.rotation.y = t * (0.15 + i * 0.04) * mo; d.position.y += Math.sin(t + i) * 0.0008 * mo; });
    this.renderer.render(this.scene, this.camera);
  }
}
