// Mark III recognition in the browser: same features as features3.py, same network as train.py.

export const NUM = 21, HAND = 1 + 2 + NUM * 3, BODY = 11, FEATURE_DIM = 2 * HAND + BODY;
export const R0 = 0, L0 = HAND, P0 = 2 * HAND;
// pose points used, in this order (MediaPipe Pose ids): nose 0, shoulders 11/12, elbows 13/14, wrists 15/16
export const POSE_IDS = [0, 11, 12, 13, 14, 15, 16];

const b64f32 = (s) => {
  const bin = atob(s), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
};

function handBlock(pts, center, scale, out, off, a) {
  // pts: Float32Array(63) raw normalized x,y,z or null
  if (!pts) return;
  const p = new Float32Array(63);
  for (let i = 0; i < 21; i++) { p[i*3] = pts[i*3] * a; p[i*3+1] = pts[i*3+1]; p[i*3+2] = pts[i*3+2] * a; }
  const wx = p[0], wy = p[1], wz = p[2];
  for (let i = 0; i < 21; i++) { p[i*3] -= wx; p[i*3+1] -= wy; p[i*3+2] -= wz; }
  let palm = 0;
  for (const k of [5, 9, 17]) palm += Math.hypot(p[k*3], p[k*3+1], p[k*3+2]);
  palm /= 3;
  if (palm > 1e-6) for (let i = 0; i < 63; i++) p[i] /= palm;
  out[off] = 1;
  if (center) { out[off+1] = (wx - center[0]) / scale; out[off+2] = (wy - center[1]) / scale; }
  out.set(p, off + 3);
}

/** right/left: Float32Array(63) of the signer's right/left hand or null;
 *  pose: array of 7 [x, y] (null for unseen points) in POSE_IDS order, or null; aspect = width/height. */
export function frameFeatures(right, left, pose, aspect) {
  const out = new Float32Array(FEATURE_DIM);
  let center = null, scale = 0;
  if (pose && pose[1] && pose[2]) {
    const q = pose.map((p) => (p ? [p[0] * aspect, p[1]] : null));
    const cx = (q[1][0] + q[2][0]) / 2, cy = (q[1][1] + q[2][1]) / 2;
    const s = Math.hypot(q[1][0] - q[2][0], q[1][1] - q[2][1]);
    if (s > 1e-4) {
      center = [cx, cy]; scale = s; out[P0] = 1;
      [0, 3, 4, 5, 6].forEach((idx, k) => {
        if (q[idx]) { out[P0 + 1 + 2 * k] = (q[idx][0] - cx) / s; out[P0 + 2 + 2 * k] = (q[idx][1] - cy) / s; }
      });
    }
  }
  handBlock(right, center, scale, out, R0, aspect);
  handBlock(left, center, scale, out, L0, aspect);
  return out;
}

export class SignModel {
  constructor(spec) {
    this.spec = spec; this.labels = spec.labels; this.T = spec.T; this.F = spec.feature_dim;
    this.H = spec.hidden; this.C = spec.labels.length;
    const w = spec.w, H = this.H;
    this.inpW = b64f32(w.inp_w); this.inpB = b64f32(w.inp_b);
    this.outW = b64f32(w.out_w); this.outB = b64f32(w.out_b);
    this.temperature = spec.temperature || 1;
    // conv weights (out, in, k) -> (k, in, out) so the inner loop runs over contiguous outputs
    const re = (src) => { const s = b64f32(src), d = new Float32Array(s.length);
      for (let o = 0; o < H; o++) for (let i = 0; i < H; i++) for (let k = 0; k < 5; k++) d[(k*H + i)*H + o] = s[(o*H + i)*5 + k];
      return d; };
    this.convs = w.convs.map((c) => ({ W: re(c.w), B: b64f32(c.b), d: c.dilation }));
    this.bufA = new Float32Array(this.T * H); this.bufB = new Float32Array(this.T * H);
    this.h0 = this.bufA;
    this.z = new Float32Array(2 * this.F);
  }

  _conv(src, dst, W, B, dil) {
    const T = this.T, H = this.H, pad = 2 * dil;
    for (let t = 0; t < T; t++) {
      const o0 = t * H;
      for (let o = 0; o < H; o++) dst[o0 + o] = B[o];
      for (let k = 0; k < 5; k++) {
        const ts = t + k * dil - pad;
        if (ts < 0 || ts >= T) continue;
        const s0 = ts * H, wk = k * H * H;
        for (let i = 0; i < H; i++) {
          const v = src[s0 + i];
          if (v === 0) continue;
          const wr = wk + i * H;
          for (let o = 0; o < H; o++) dst[o0 + o] += v * W[wr + o];
        }
      }
      for (let o = 0; o < H; o++) { const r = dst[o0 + o]; dst[o0 + o] = (r > 0 ? r : 0) + src[o0 + o]; }
    }
  }

  /** frames: array of T Float32Array(F) -> probabilities (Float32Array C) */
  probs(frames) {
    const T = this.T, F = this.F, H = this.H, z = this.z, h0 = this.h0;
    for (let t = 0; t < T; t++) {
      const x = frames[t], prev = t > 0 ? frames[t - 1] : null;
      for (let i = 0; i < F; i++) { z[i] = x[i]; z[F + i] = prev ? x[i] - prev[i] : 0; }
      const o0 = t * H;
      for (let o = 0; o < H; o++) h0[o0 + o] = this.inpB[o];
      for (let i = 0; i < 2 * F; i++) {
        const v = z[i]; if (v === 0) continue;
        const wr = i * H;
        for (let o = 0; o < H; o++) h0[o0 + o] += v * this.inpW[wr + o];
      }
      for (let o = 0; o < H; o++) if (h0[o0 + o] < 0) h0[o0 + o] = 0;
    }
    let src = h0, dst = this.bufB;
    for (const c of this.convs) { this._conv(src, dst, c.W, c.B, c.d); [src, dst] = [dst, src]; }
    const pooled = new Float32Array(2 * H), h = src;
    for (let o = 0; o < H; o++) {
      let s = 0, m = -Infinity;
      for (let t = 0; t < T; t++) { const v = h[t * H + o]; s += v; if (v > m) m = v; }
      pooled[o] = s / T; pooled[H + o] = m;
    }
    const C = this.C, logits = new Float32Array(C);
    for (let c = 0; c < C; c++) logits[c] = this.outB[c];
    for (let i = 0; i < 2 * H; i++) { const v = pooled[i], wr = i * C; for (let c = 0; c < C; c++) logits[c] += v * this.outW[wr + c]; }
    const tmp = this.temperature;                 // calibrated: shown confidence matches real accuracy
    for (let c = 0; c < C; c++) logits[c] /= tmp;
    let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
    let sum = 0; for (let c = 0; c < C; c++) { logits[c] = Math.exp(logits[c] - mx); sum += logits[c]; }
    for (let c = 0; c < C; c++) logits[c] /= sum;
    return logits;
  }
}

/** Nearest-frame resampling of a timestamped buffer onto T evenly spaced times in [t0, t1]. */
export function windowFrom(buffer, t0, t1, T) {
  const out = new Array(T);
  let j = 0;
  for (let k = 0; k < T; k++) {
    const t = t0 + (t1 - t0) * k / (T - 1);
    while (j + 1 < buffer.length && Math.abs(buffer[j + 1].t - t) <= Math.abs(buffer[j].t - t)) j++;
    out[k] = buffer[j].f;
  }
  return out;
}
