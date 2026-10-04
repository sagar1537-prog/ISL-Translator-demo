// Runs the sign model off the main thread, so video, 3D and UI never stutter.
import { SignModel } from "./engine2.js";
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
    let best = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    let second = best === 0 ? 1 : 0; for (let i = 0; i < p.length; i++) if (i !== best && p[i] > p[second]) second = i;
    self.postMessage({ type: "result", id: m.id, scale: m.scale, best, p: p[best], second, p2: p[second], ms: performance.now() - t0 });
  }
};
