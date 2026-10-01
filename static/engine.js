// On-device recognition: the same maths as features.py + engine.py, so a reading
// takes ~0.05 ms instead of a network round-trip. The server keeps /api/predict
// as a fallback, and both give identical answers (checked in the tests).

const b64f32 = (s) => {
  const bin = atob(s), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
};

// x (B... flattened single row of length n) @ w (n x m) + b
function dense(x, w, b, n, m, out) {
  for (let j = 0; j < m; j++) out[j] = b[j];
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = i * m;
    for (let j = 0; j < m; j++) out[j] += xi * w[row + j];
  }
  return out;
}
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

export function features(pts, isLeft, width, height, cfg) {
  const out = new Float32Array(63);
  out.set(pts);
  if (Number(cfg.version || 1) < 2) {                 // legacy: wrist-relative only
    const wx = out[0], wy = out[1], wz = out[2];
    for (let i = 0; i < 21; i++) { out[i*3] -= wx; out[i*3+1] -= wy; out[i*3+2] -= wz; }
    return out;
  }
  if (cfg.aspect_correct !== false && width > 0 && height > 0) {
    const a = width / height;
    for (let i = 0; i < 21; i++) { out[i*3] *= a; out[i*3+2] *= a; }
  }
  const wx = out[0], wy = out[1], wz = out[2];
  for (let i = 0; i < 21; i++) { out[i*3] -= wx; out[i*3+1] -= wy; out[i*3+2] -= wz; }
  if (cfg.mirror_left !== false && isLeft) for (let i = 0; i < 21; i++) out[i*3] *= -1;
  if (cfg.scale !== false) {
    let palm = 0;
    for (const k of [5, 9, 17]) palm += Math.hypot(out[k*3], out[k*3+1], out[k*3+2]);
    palm /= 3;
    if (palm > 1e-6) for (let i = 0; i < 63; i++) out[i] /= palm;
  }
  return out;
}

export class LocalModel {
  constructor(spec) {
    this.arch = spec.arch; this.labels = spec.labels; this.feature = spec.feature;
    this.window = spec.arch.type === "gru" ? (spec.arch.window || null) : null;
    if (spec.arch.type === "mlp") {
      this.layers = spec.layers.map((l) => ({ w: b64f32(l.w), b: b64f32(l.b), n: l.n, m: l.m }));
      this.bufs = this.layers.map((l) => new Float32Array(l.m));
    } else if (spec.arch.type === "gru") {
      this.gru = spec.gru.map((g) => ({ wih: b64f32(g.wih), whh: b64f32(g.whh), bih: b64f32(g.bih), bhh: b64f32(g.bhh), n: g.n, h: g.h }));
      this.head = { w: b64f32(spec.head.w), b: b64f32(spec.head.b), n: spec.head.n, m: spec.head.m };
    } else throw new Error("unsupported model type");
  }

  logits(x) {                                           // x: Float32Array(63) or array of them (GRU)
    if (this.arch.type === "mlp") {
      let cur = x;
      this.layers.forEach((l, i) => {
        dense(cur, l.w, l.b, l.n, l.m, this.bufs[i]);
        if (i < this.layers.length - 1) for (let j = 0; j < l.m; j++) if (this.bufs[i][j] < 0) this.bufs[i][j] = 0;
        cur = this.bufs[i];
      });
      return Float32Array.from(cur);
    }
    // GRU over a window; input = [features, velocity]
    let seq = x.map((f, t) => {
      const v = new Float32Array(126); v.set(f);
      if (t > 0) for (let i = 0; i < 63; i++) v[63 + i] = f[i] - x[t - 1][i];
      return v;
    });
    let h = null;
    for (const g of this.gru) {
      const H = g.h; h = new Float32Array(H);
      const outs = [], gx = new Float32Array(3 * H), gh = new Float32Array(3 * H);
      for (const inp of seq) {
        dense(inp, g.wih, g.bih, g.n, 3 * H, gx);
        dense(h, g.whh, g.bhh, H, 3 * H, gh);
        const nh = new Float32Array(H);
        for (let j = 0; j < H; j++) {
          const r = sigmoid(gx[j] + gh[j]), z = sigmoid(gx[H + j] + gh[H + j]);
          const n = Math.tanh(gx[2 * H + j] + r * gh[2 * H + j]);
          nh[j] = (1 - z) * n + z * h[j];
        }
        h = nh; outs.push(nh);
      }
      seq = outs;
    }
    return dense(h, this.head.w, this.head.b, this.head.n, this.head.m, new Float32Array(this.head.m));
  }

  probs(x) {
    const z = this.logits(x);
    let mx = -Infinity; for (const v of z) mx = Math.max(mx, v);
    let sum = 0; const e = Array.from(z, (v) => { const q = Math.exp(v - mx); sum += q; return q; });
    return e.map((q) => q / sum);
  }
}
