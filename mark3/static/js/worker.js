// Runs the sign model off the main thread, so video, 3D and UI never stutter.
import { SignModel } from "./engine3.js";
let model = null;
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "load") {
    try { model = new SignModel(m.spec); self.postMessage({ type: "ready" }); }
    catch (err) { self.postMessage({ type: "error", message: String(err) }); }
  } else if (m.type === "run" && model) {
    const t0 = performance.now();
    const T = model.T, F = model.F, frames = new Array(T);
    for (let t = 0; t < T; t++) frames[t] = m.data.subarray(t * F, (t + 1) * F);
    const p = model.probs(frames);
    const top = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 3);
    self.postMessage({ type: "result", id: m.id, scale: m.scale, best: top[0], p: p[top[0]],
                       top: top.map((i) => [i, p[i]]), ms: performance.now() - t0 });
  }
};
