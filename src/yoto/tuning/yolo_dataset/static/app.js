// Min/max pairs render as a single dual-handle "bracket" slider.
const PAIRS = [
  ["area", "area_min", "area_max", 0, 3000],
  ["angle", "angle_min", "angle_max", 30, 150],
  ["side", "side_min", "side_max", 0, 120],
];
const SINGLES = [
  ["ratio_min", 0, 1],
  ["dedup_px", 0, 100],
];
const ACCEPT = "#22e06b"; // green  = will be exported
const REJECT = "#2f9bff"; // blue   = filtered out

let state = {
  frames: [],
  cur: null,
  thr: {},
  data: null,
  overrides: {},
  img: null,
  scale: 1,
  dropAlpha: 1, // opacity of the blue "drop" quads (1 = opaque)
};

async function api(method, url, body) {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  return (await fetch(url, opt)).json();
}

function polyArea(p) {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    s += p[i][0] * p[j][1] - p[j][0] * p[i][1];
  }
  return Math.abs(s) / 2;
}
function angles(p) {
  const out = [];
  for (let i = 0; i < 4; i++) {
    const a = p[(i + 3) % 4],
      b = p[i],
      c = p[(i + 1) % 4];
    const v1 = [a[0] - b[0], a[1] - b[1]],
      v2 = [c[0] - b[0], c[1] - b[1]];
    const d =
      (v1[0] * v2[0] + v1[1] * v2[1]) /
      (Math.hypot(...v1) * Math.hypot(...v2) + 1e-12);
    out.push((Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI);
  }
  return out;
}
function sides(p) {
  const out = [];
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    out.push(Math.hypot(p[j][0] - p[i][0], p[j][1] - p[i][1]));
  }
  return out;
}
function passes(corners, thr) {
  const a = polyArea(corners);
  if (a < thr.area_min || a > thr.area_max) return false;
  if (!angles(corners).every((x) => x >= thr.angle_min && x <= thr.angle_max))
    return false;
  const s = sides(corners);
  if (Math.min(...s) < thr.side_min || Math.max(...s) > thr.side_max)
    return false;
  if (Math.min(...s) / Math.max(...s) < thr.ratio_min) return false;
  return true;
}
function isOn(q) {
  const ov = state.overrides[q.quad_idx];
  return ov === "include" || (ov !== "exclude" && passes(q.corners, state.thr));
}
function accepted(q) {
  return q.is_valid || isOn(q);
}

// Fit the canvas to the available window area (recomputed on resize).
function fit() {
  const d = state.data;
  if (!d) return;
  const cv = document.getElementById("canvas");
  const main = document.getElementById("main");
  const availW = Math.max(200, main.clientWidth - 24);
  const availH = Math.max(200, window.innerHeight - 70);
  state.scale = Math.min(availW / d.width, availH / d.height);
  cv.width = Math.round(d.width * state.scale);
  cv.height = Math.round(d.height * state.scale);
}

// Redraw only the overlay onto the already-loaded (cached) image — no reload,
// so the image itself never flickers.
function draw() {
  const d = state.data;
  if (!d || !state.img) return;
  const cv = document.getElementById("canvas");
  const ctx = cv.getContext("2d");
  ctx.drawImage(state.img, 0, 0, cv.width, cv.height);
  const s = state.scale;
  for (const q of d.quads) {
    const acc = accepted(q);
    ctx.globalAlpha = acc ? 1 : state.dropAlpha;
    ctx.strokeStyle = acc ? ACCEPT : REJECT;
    ctx.lineWidth = acc ? 3 : 2;
    ctx.beginPath();
    q.corners.forEach((p, i) => {
      const x = p[0] * s,
        y = p[1] * s;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

document.getElementById("canvas").addEventListener("click", (e) => {
  const cv = document.getElementById("canvas");
  const rect = cv.getBoundingClientRect();
  const s = state.scale;
  const mx = (e.clientX - rect.left) / s,
    my = (e.clientY - rect.top) / s;
  let best = null,
    bestDist = 1e9;
  for (const q of state.data.quads) {
    if (q.is_valid) continue;
    const cx = q.corners.reduce((a, p) => a + p[0], 0) / 4;
    const cy = q.corners.reduce((a, p) => a + p[1], 0) / 4;
    const dist = Math.hypot(cx - mx, cy - my);
    if (dist < bestDist) {
      bestDist = dist;
      best = q;
    }
  }
  if (!best || bestDist > 60) return;
  state.overrides[best.quad_idx] = isOn(best) ? "exclude" : "include";
  draw();
});

function enhanceOn() {
  return document.getElementById("enhance-chk").checked;
}

async function loadFrame(exp, frame) {
  state.cur = { exp, frame };
  state.data = await api("GET", `/api/frame/${exp}/${frame}?enhance=${enhanceOn()}`);
  state.overrides = state.data.overrides || {};
  document.getElementById("frame-title").textContent =
    `${exp} · frame ${frame} · ${state.data.status}`;
  const img = new Image();
  img.onload = () => {
    state.img = img;
    fit();
    draw();
  };
  img.src = state.data.image_url;
}

async function saveDecision(status) {
  await api(
    "PUT",
    `/api/frame/${state.cur.exp}/${state.cur.frame}/decision`,
    { status, overrides: state.overrides, enhance: enhanceOn() },
  );
  const idx = state.frames.findIndex(
    (f) => f.exp === state.cur.exp && f.frame === state.cur.frame,
  );
  if (idx >= 0) state.frames[idx].status = status;
  renderList();
  // Auto-advance to the next frame.
  const next = state.frames[idx + 1];
  if (next) loadFrame(next.exp, next.frame);
}

function renderList() {
  const ul = document.getElementById("frame-list");
  ul.innerHTML = "";
  for (const f of state.frames) {
    const li = document.createElement("li");
    li.textContent = `${f.exp} f${f.frame} (${f.count}) [${f.status}]`;
    li.className = f.status;
    li.onclick = () => loadFrame(f.exp, f.frame);
    ul.appendChild(li);
  }
}

function persistThr(obj) {
  api("PUT", "/api/thresholds", obj);
}

function makeBracket(box, name, kMin, kMax, lo, hi) {
  const step = (hi - lo) / 500;
  const wrap = document.createElement("div");
  wrap.className = "ctrl";
  const head = document.createElement("label");
  const rd = document.createElement("span");
  rd.className = "rd";
  head.textContent = name + " ";
  head.appendChild(rd);
  const dual = document.createElement("div");
  dual.className = "dual";
  const track = document.createElement("div");
  track.className = "track";
  const fill = document.createElement("div");
  fill.className = "fill";
  const smin = document.createElement("input");
  const smax = document.createElement("input");
  [smin, smax].forEach((s) => {
    s.type = "range";
    s.min = lo;
    s.max = hi;
    s.step = step;
  });
  smin.value = state.thr[kMin] ?? lo;
  smax.value = state.thr[kMax] ?? hi;

  function upd() {
    const a = +smin.value,
      b = +smax.value;
    state.thr[kMin] = a;
    state.thr[kMax] = b;
    const p0 = ((a - lo) / (hi - lo)) * 100,
      p1 = ((b - lo) / (hi - lo)) * 100;
    fill.style.left = p0 + "%";
    fill.style.width = p1 - p0 + "%";
    rd.textContent = `${a.toFixed(1)} – ${b.toFixed(1)}`;
    draw();
  }
  smin.oninput = () => {
    if (+smin.value > +smax.value) smin.value = smax.value;
    upd();
  };
  smax.oninput = () => {
    if (+smax.value < +smin.value) smax.value = smin.value;
    upd();
  };
  smin.onchange = smax.onchange = () =>
    persistThr({ [kMin]: +smin.value, [kMax]: +smax.value });

  dual.append(track, fill, smin, smax);
  wrap.append(head, dual);
  box.append(wrap);
  upd();
}

function makeSingle(box, key, lo, hi) {
  const step = (hi - lo) / 500;
  const wrap = document.createElement("div");
  wrap.className = "ctrl";
  const head = document.createElement("label");
  const rd = document.createElement("span");
  rd.className = "rd";
  head.textContent = key + " ";
  head.appendChild(rd);
  const s = document.createElement("input");
  s.type = "range";
  s.min = lo;
  s.max = hi;
  s.step = step;
  s.value = state.thr[key] ?? lo;
  function upd() {
    state.thr[key] = +s.value;
    rd.textContent = (+s.value).toFixed(2);
    draw();
  }
  s.oninput = upd;
  s.onchange = () => persistThr({ [key]: +s.value });
  wrap.append(head, s);
  box.append(wrap);
  upd();
}

function renderControls() {
  const box = document.getElementById("sliders");
  box.innerHTML = "";
  for (const [name, kMin, kMax, lo, hi] of PAIRS)
    makeBracket(box, name, kMin, kMax, lo, hi);
  for (const [key, lo, hi] of SINGLES) makeSingle(box, key, lo, hi);
}

document.getElementById("enhance-chk").onchange = () => {
  if (state.cur) loadFrame(state.cur.exp, state.cur.frame);
};
const dropInput = document.getElementById("drop-opacity");
const dropRd = document.getElementById("drop-opacity-rd");
function updateDrop() {
  const transparency = +dropInput.value; // 0 = opaque, 1 = invisible
  state.dropAlpha = 1 - transparency;
  dropRd.textContent = transparency.toFixed(2);
  draw();
}
dropInput.oninput = updateDrop;
updateDrop();
document.getElementById("reset-thr-btn").onclick = async () => {
  state.thr = await api("GET", "/api/thresholds/default");
  persistThr(state.thr);
  renderControls();
  draw();
};
document.getElementById("accept-btn").onclick = () => saveDecision("accepted");
document.getElementById("skip-btn").onclick = () => saveDecision("skipped");
document.getElementById("accept-all-btn").onclick = async () => {
  const res = await api("POST", "/api/accept-all");
  for (const f of state.frames) f.status = "accepted";
  renderList();
  document.getElementById("export-msg").textContent =
    `Accepted all ${res.accepted} frames.`;
};
document.addEventListener("keydown", (e) => {
  if (e.key === "a") saveDecision("accepted");
  if (e.key === "s") saveDecision("skipped");
});
window.addEventListener("resize", () => {
  fit();
  draw();
});
document.getElementById("export-btn").onclick = async () => {
  const formats = [...document.querySelectorAll(".fmt:checked")].map(
    (c) => c.value,
  );
  const val_fraction = +document.getElementById("val-frac").value;
  const res = await api("POST", "/api/export", { formats, val_fraction });
  document.getElementById("export-msg").textContent =
    `Wrote ${res.written} frames.`;
};

(async function init() {
  state.frames = await api("GET", "/api/frames");
  state.thr = await api("GET", "/api/thresholds");
  renderList();
  renderControls();
  if (state.frames.length) loadFrame(state.frames[0].exp, state.frames[0].frame);
})();
