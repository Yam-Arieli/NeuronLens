/* NeuronLens – main.js
 * Pure vanilla JS, no external dependencies required.
 * Data is inlined as __NETWORK__ and __ACTIVATIONS__ by the Python renderer.
 */

"use strict";

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_EDGE_ALPHA = 0.55;
const DIM_ALPHA      = 0.2;   // softened dim for non-hovered neurons

const COLOR_A   = { r: 88,  g: 166, b: 255 };  // #58a6ff  (blue)
const COLOR_B   = { r: 255, g: 100, b: 100 };  // #ff6464  (red)
const COLOR_SIG = { r: 255, g: 200, b: 80  };  // warm amber (hover signal)

const FADE_IN_MS  = 100;   // ms to blend into hover state
const FADE_OUT_MS = 300;   // ms to blend back to normal

// ── State ─────────────────────────────────────────────────────────────────────

let network     = null;
let activations = null;
let canvasEl, ctx;
let mode         = "single";   // "single" | "compare"
let filterGroupA = "default";
let filterGroupB = "default";
let layout       = null;

// Hover animation state — replaces the old hoveredNeuron boolean
const hoverState = {
  neuron:    null,   // { layerIdx, neuronIdx } — persists during fade-out
  alpha:     0,      // current blend [0 = normal, 1 = full hover]
  target:    0,      // animation target (0 or 1)
  animFrom:  0,      // alpha when current animation started
  animStart: null,   // DOMHighResTimeStamp when animation started
  rafId:     null,   // requestAnimationFrame handle
};

// ── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  canvasEl    = document.getElementById("main-canvas");
  ctx         = canvasEl.getContext("2d");
  network     = __NETWORK__;
  activations = __ACTIVATIONS__;

  buildUI();
  buildLayout();
  render();

  canvasEl.addEventListener("mousemove",  onMouseMove);
  canvasEl.addEventListener("mouseleave", onMouseLeave);
  window.addEventListener("resize", () => { buildLayout(); render(); });
});

// ── UI ────────────────────────────────────────────────────────────────────────

function buildUI() {
  const groups      = Object.keys(activations.groups);
  const filterIndex = activations.filter_index || {};

  function specToLabel(spec) {
    if (spec.and) return spec.and.map(specToLabel).join(" & ");
    const opMap = { eq:"=", ne:"≠", lt:"<", le:"≤", gt:">", ge:"≥", in:"∈", not_in:"∉" };
    return `${spec.column} ${opMap[spec.op] || spec.op} ${JSON.stringify(spec.value)}`;
  }
  function groupLabel(key) {
    if (key === "default") return "All data";
    const spec = filterIndex[key];
    return spec ? specToLabel(spec) : key;
  }
  function populateSelect(id, onChange) {
    const sel = document.getElementById(id);
    groups.forEach(g => {
      const opt = document.createElement("option");
      opt.value = g; opt.textContent = groupLabel(g);
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => { onChange(sel.value); render(); });
    return sel;
  }

  populateSelect("filter-a", v => { filterGroupA = v; });
  const selB = populateSelect("filter-b", v => { filterGroupB = v; });
  selB.value   = groups[Math.min(1, groups.length - 1)];
  filterGroupB = selB.value;

  document.getElementById("btn-single").addEventListener("click", () => {
    mode = "single";
    document.getElementById("btn-single").classList.add("active");
    document.getElementById("btn-compare").classList.remove("active");
    document.getElementById("compare-row").style.display = "none";
    render();
  });
  document.getElementById("btn-compare").addEventListener("click", () => {
    mode = "compare";
    document.getElementById("btn-compare").classList.add("active");
    document.getElementById("btn-single").classList.remove("active");
    document.getElementById("compare-row").style.display = "flex";
    render();
  });
}

// ── Layout ────────────────────────────────────────────────────────────────────

function buildLayout() {
  if (!network) return;
  const layers = network.layers;
  const n      = layers.length;

  const container = canvasEl.parentElement;
  const availW    = Math.max(container.clientWidth,  400);
  const availH    = Math.max(container.clientHeight, 300);

  const maxUnits = Math.max(...layers.map(l => l.n_display_units));

  // ── Label area ──────────────────────────────────────────────────────────
  const labelFontSize = Math.max(Math.min(Math.round(availH * 0.018), 15), 10);
  const labelAreaH    = Math.ceil(labelFontSize * 2.8);  // room for 2 text lines

  // ── Block vertical padding ───────────────────────────────────────────────
  const blockPadV = Math.max(Math.round(availH * 0.03), 4);

  // ── Neuron sizing: fill 88 % of canvas height minus label + padding ──────
  const neuronSpaceH = availH * 0.88 - labelAreaH - 2 * blockPadV;
  const unitH        = neuronSpaceH / Math.max(maxUnits, 1);
  const neuronR      = Math.max(Math.floor(unitH * 0.42), 4);
  const neuronGap    = Math.max(unitH - neuronR * 2, 2);

  // ── Total content height for the tallest column ──────────────────────────
  const maxNeuronColH = maxUnits * (neuronR * 2 + neuronGap) - neuronGap;
  const totalContentH = labelAreaH + blockPadV + maxNeuronColH + blockPadV;

  // ── Vertical centering: shift so content sits in the middle ──────────────
  const topOffset  = Math.max((availH - totalContentH) / 2, 4);

  // ── Derived y coordinates (stored in layout for draw functions) ──────────
  const labelY1    = topOffset + labelFontSize * 1.0;
  const labelY2    = topOffset + labelFontSize * 2.2;
  const blockTopY  = topOffset + labelAreaH;
  const blockBotY  = blockTopY + blockPadV + maxNeuronColH + blockPadV;
  const neuronTopY = blockTopY + blockPadV;

  // ── Horizontal layout: spread layers evenly ──────────────────────────────
  const PAD_SIDE     = Math.round(availW * 0.03);
  const innerW       = availW - PAD_SIDE * 2 - neuronR * 2;
  const layerSpacing = n > 1 ? innerW / (n - 1) : 0;

  // ── Per-layer positions ──────────────────────────────────────────────────
  const layoutLayers = layers.map((layer, li) => {
    const cx        = PAD_SIDE + neuronR + li * layerSpacing;
    const nu        = layer.n_display_units;
    const layerColH = nu * (neuronR * 2 + neuronGap) - neuronGap;
    // Shorter columns are centered vertically within the max-height area
    const startY    = neuronTopY + (maxNeuronColH - layerColH) / 2;
    const neurons   = [];
    for (let k = 0; k < nu; k++)
      neurons.push({ x: cx, y: startY + k * (neuronR * 2 + neuronGap) + neuronR });
    return { cx, neurons };
  });

  const maxEdgeWidth = Math.max(neuronR * 0.35, 2);
  const blockPadH    = Math.max(neuronR * 0.7,  4);
  const canvasW      = availW;
  const canvasH      = availH;

  // Scale by devicePixelRatio so text/lines are sharp on retina displays
  const dpr = window.devicePixelRatio || 1;
  canvasEl.width        = Math.round(canvasW * dpr);
  canvasEl.height       = Math.round(canvasH * dpr);
  canvasEl.style.width  = canvasW + "px";
  canvasEl.style.height = canvasH + "px";

  layout = {
    layers: layoutLayers,
    canvasW, canvasH, dpr,
    neuronR, maxEdgeWidth, labelFontSize,
    labelY1, labelY2,
    blockTopY, blockBotY, blockPadH,
  };

  const sb = document.getElementById("status-bar");
  if (sb) sb.textContent = layers.map(l =>
    `${l.name}: ${l.n_neurons}${l.aggregated ? ` (${l.n_display_units} buckets)` : ""}`
  ).join("  ·  ");
}

// ── Activation helpers ────────────────────────────────────────────────────────

function getActivations(groupKey, layerIdx) {
  const grp = activations.groups[groupKey];
  return grp ? (grp[String(layerIdx)] || null) : null;
}

// Returns activations in DISPLAY ORDER (display unit k → its activation).
// raw[] is stored in original-neuron order; perm[display_pos] = original_idx.
function getDisplayActivations(groupKey, layerIdx) {
  const raw   = getActivations(groupKey, layerIdx);
  if (!raw) return null;
  const layer = network.layers[layerIdx];
  const perm  = layer.perm;

  if (!layer.aggregated) {
    return perm.map(origIdx => raw[origIdx] || 0);
  }

  // Aggregated: average over bucket of original neurons
  const bs      = layer.bucket_size;
  const nu      = layer.n_display_units;
  const display = new Array(nu).fill(0);
  for (let k = 0; k < nu; k++) {
    let sum = 0, cnt = 0;
    for (let b = 0; b < bs; b++) {
      const pos = k * bs + b;
      if (pos >= perm.length) break;
      sum += raw[perm[pos]] || 0;
      cnt++;
    }
    display[k] = cnt > 0 ? sum / cnt : 0;
  }
  return display;
}

function normalizeArray(arr) {
  if (!arr || arr.length === 0) return arr;
  const mx = Math.max(...arr, 1e-9);
  return arr.map(v => v / mx);
}

// ── Edge weight lookup ────────────────────────────────────────────────────────

// edges[l][display_i][display_j] = weight from display unit i (layer l)
//                                  to display unit j (layer l+1)
// For aggregated layers, average over the bucket of underlying neurons.
function getBucketWeight(fromLayerIdx, i_disp, j_disp) {
  const edges   = network.edges[fromLayerIdx];
  if (!edges) return 0;
  const fromLayer = network.layers[fromLayerIdx];
  const toLayer   = network.layers[fromLayerIdx + 1];
  const bs_in     = fromLayer.bucket_size;
  const bs_out    = toLayer.bucket_size;

  let sum = 0, cnt = 0;
  for (let bi = 0; bi < bs_in; bi++) {
    const ri = i_disp * bs_in + bi;
    if (ri >= fromLayer.n_neurons || ri >= edges.length) break;
    for (let bj = 0; bj < bs_out; bj++) {
      const rj = j_disp * bs_out + bj;
      if (rj >= toLayer.n_neurons) break;
      if (!edges[ri] || rj >= edges[ri].length) continue;
      sum += Math.abs(edges[ri][rj]);
      cnt++;
    }
  }
  return cnt > 0 ? sum / cnt : 0;
}

// ── Hover signal propagation ──────────────────────────────────────────────────

function computeHoverSignal(layerIdx, neuronIdx) {
  const n      = network.layers.length;
  const signal = new Array(n).fill(null);

  // Seed with the hovered neuron's actual activation
  const rawActs = getDisplayActivations(filterGroupA, layerIdx) || [];
  const seedVal = rawActs[neuronIdx] || 0;
  const seedLayer = new Array(network.layers[layerIdx].n_display_units).fill(0);
  seedLayer[neuronIdx] = seedVal;
  signal[layerIdx]     = seedLayer;

  // Forward pass
  for (let l = layerIdx; l < n - 1; l++) {
    const n_out = network.layers[l + 1].n_display_units;
    const n_in  = network.layers[l].n_display_units;
    const next  = new Array(n_out).fill(0);
    for (let i = 0; i < n_in; i++) {
      const src = signal[l][i];
      if (!src || src < 1e-15) continue;
      for (let j = 0; j < n_out; j++)
        next[j] += getBucketWeight(l, i, j) * src;
    }
    signal[l + 1] = next;
  }
  return signal;
}

// ── Colour helpers ────────────────────────────────────────────────────────────

function neuronColor(normA, normB) {
  if (mode === "compare" && normB !== null) {
    const total = normA + normB;
    if (total < 1e-9) return { r: 30, g: 35, b: 45, a: 0.5 };
    const tB = normB / total, tA = 1 - tB;
    const brightness = Math.min(total / 2, 1);
    return {
      r: Math.round(COLOR_A.r * tA + COLOR_B.r * tB),
      g: Math.round(COLOR_A.g * tA + COLOR_B.g * tB),
      b: Math.round(COLOR_A.b * tA + COLOR_B.b * tB),
      a: 0.2 + brightness * 0.8,
    };
  }
  const v = normA || 0;
  return {
    r: Math.round(30 + v * (COLOR_A.r - 30)),
    g: Math.round(35 + v * (COLOR_A.g - 35)),
    b: Math.round(45 + v * (COLOR_A.b - 45)),
    a: 0.3 + v * 0.7,
  };
}

function signalColor(normSig) {
  const v = normSig || 0;
  return {
    r: Math.round(30 + v * (COLOR_SIG.r - 30)),
    g: Math.round(35 + v * (COLOR_SIG.g - 35)),
    b: Math.round(45 + v * (COLOR_SIG.b - 45)),
    a: 0.25 + v * 0.75,
  };
}

// Linear interpolation between two colour objects
function lerpColor(a, b, t) {
  if (t <= 0) return a;
  if (t >= 1) return b;
  return {
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t),
    a: a.a + (b.a - a.a) * t,
  };
}

// ── Block backgrounds ─────────────────────────────────────────────────────────

function roundedRect(x, y, w, h, r) {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawBlockBackgrounds() {
  const n      = network.layers.length;
  const r      = layout.neuronR;
  const padH   = layout.blockPadH;
  const corner = Math.max(r * 0.5, 6);

  for (let l = 0; l < n; l++) {
    if (layout.layers[l].neurons.length === 0) continue;
    const cx = layout.layers[l].cx;
    const x  = cx - r - padH;
    const w  = (r + padH) * 2;

    ctx.beginPath();
    roundedRect(x, layout.blockTopY, w, layout.blockBotY - layout.blockTopY, corner);
    ctx.fillStyle   = "rgba(255,255,255,0.025)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth   = 1;
    ctx.stroke();
  }
}

// ── Hover animation ───────────────────────────────────────────────────────────

function startHoverAnim(target) {
  if (hoverState.rafId) cancelAnimationFrame(hoverState.rafId);
  hoverState.target    = target;
  hoverState.animFrom  = hoverState.alpha;
  hoverState.animStart = null;
  hoverState.rafId     = requestAnimationFrame(animStep);
}

function animStep(ts) {
  if (hoverState.animStart === null) hoverState.animStart = ts;
  const elapsed = ts - hoverState.animStart;
  const dur     = hoverState.target === 1 ? FADE_IN_MS : FADE_OUT_MS;
  const t       = Math.min(elapsed / dur, 1);
  hoverState.alpha = hoverState.animFrom + (hoverState.target - hoverState.animFrom) * t;
  render();
  if (t < 1) {
    hoverState.rafId = requestAnimationFrame(animStep);
  } else {
    hoverState.rafId = null;
    if (hoverState.target === 0) hoverState.neuron = null;  // fully faded out
  }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function render() {
  if (!network || !layout) return;
  const { canvasW, canvasH, dpr } = layout;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvasW, canvasH);

  const n     = network.layers.length;
  const ha    = hoverState.alpha;   // 0 = normal, 1 = full hover
  const hovL  = hoverState.neuron ? hoverState.neuron.layerIdx  : -1;
  const hovK  = hoverState.neuron ? hoverState.neuron.neuronIdx : -1;

  const actsA = network.layers.map((_, i) =>
    normalizeArray(getDisplayActivations(filterGroupA, i))
  );
  const actsB = mode === "compare"
    ? network.layers.map((_, i) => normalizeArray(getDisplayActivations(filterGroupB, i)))
    : null;

  // Precompute hover signal when any hover effect is visible
  let hoverSigNorm = null;
  if (hoverState.neuron && ha > 0) {
    const raw = computeHoverSignal(hovL, hovK);
    hoverSigNorm = raw.map(s => s ? normalizeArray(s) : null);
  }

  // Draw order: backgrounds → edges → neurons → labels
  drawBlockBackgrounds();
  for (let l = 0; l < n - 1; l++) drawEdges(l, actsA, actsB, hoverSigNorm, hovL, hovK, ha);
  for (let l = 0; l < n;     l++) drawLayer(l, actsA[l], actsB ? actsB[l] : null, hoverSigNorm, hovL, hovK, ha);

  ctx.font      = `${layout.labelFontSize}px -apple-system, monospace`;
  ctx.textAlign = "center";
  ctx.fillStyle = "#8b949e";
  for (let l = 0; l < n; l++) {
    const lx    = layout.layers[l].cx;
    const label = network.layers[l].block_label || network.layers[l].name;
    ctx.fillText(label,                              lx, layout.labelY1);
    ctx.fillText(`(${network.layers[l].n_neurons})`, lx, layout.labelY2);
  }
}

function drawLayer(layerIdx, normsA, normsB, hoverSigNorm, hovL, hovK, ha) {
  const layerLayout = layout.layers[layerIdx];
  const nu          = layerLayout.neurons.length;
  const r           = layout.neuronR;

  for (let k = 0; k < nu; k++) {
    const { x, y } = layerLayout.neurons[k];

    // ── Compute base (normal) colour ────────────────────────────────────────
    const normCol = neuronColor(normsA ? normsA[k] : 0, normsB ? normsB[k] : null);

    // ── Compute hover colour for this neuron ────────────────────────────────
    let hoverCol  = normCol;
    let isHovered = false;

    if (hoverSigNorm && ha > 0) {
      if (layerIdx < hovL) {
        hoverCol = { ...normCol, a: DIM_ALPHA };
      } else if (layerIdx === hovL) {
        if (k === hovK) {
          hoverCol  = normCol;   // hovered neuron stays at normal colour
          isHovered = true;
        } else {
          hoverCol = { ...normCol, a: DIM_ALPHA };
        }
      } else {
        // Downstream: colour by propagated signal
        const sig = hoverSigNorm[layerIdx] ? (hoverSigNorm[layerIdx][k] || 0) : 0;
        hoverCol  = signalColor(sig);
      }
    }

    // ── Blend normal → hover by ha ──────────────────────────────────────────
    const col   = lerpColor(normCol, hoverCol, ha);
    const alpha = col.a;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${alpha})`;
    ctx.fill();

    // Glow
    if (alpha > 0.4) {
      const glowVal = (hoverSigNorm && layerIdx > hovL)
        ? (hoverSigNorm[layerIdx] ? (hoverSigNorm[layerIdx][k] || 0) : 0)
        : (normsA ? normsA[k] : 0);
      if (glowVal > 0.3) {
        ctx.shadowColor = `rgb(${col.r},${col.g},${col.b})`;
        ctx.shadowBlur  = r * 0.4 + r * 0.4 * glowVal;
        ctx.fill();
        ctx.shadowBlur  = 0;
      }
    }

    // Border
    ctx.strokeStyle = (isHovered && ha > 0.5)
      ? `rgba(255,255,255,${0.6 + ha * 0.35})`
      : `rgba(${col.r},${col.g},${col.b},${Math.min(alpha + 0.2, 1)})`;
    ctx.lineWidth   = (isHovered && ha > 0.5) ? 2 : 1;
    ctx.stroke();
  }
}

function drawEdges(fromLayerIdx, actsA, actsB, hoverSigNorm, hovL, hovK, ha) {
  const toLayerIdx = fromLayerIdx + 1;
  const edges      = network.edges[fromLayerIdx];
  if (!edges) return;

  const fromLayer = network.layers[fromLayerIdx];
  const toLayer   = network.layers[toLayerIdx];
  const n_in      = fromLayer.n_display_units;
  const n_out     = toLayer.n_display_units;

  // Pre-compute max weight for normalisation
  let maxW = 0;
  for (let i = 0; i < n_in; i++)
    for (let j = 0; j < n_out; j++)
      maxW = Math.max(maxW, getBucketWeight(fromLayerIdx, i, j));
  if (maxW < 1e-12) return;

  for (let i = 0; i < n_in; i++) {
    const { x: x1, y: y1 } = layout.layers[fromLayerIdx].neurons[i];

    for (let j = 0; j < n_out; j++) {
      const { x: x2, y: y2 } = layout.layers[toLayerIdx].neurons[j];
      const w     = getBucketWeight(fromLayerIdx, i, j);
      const normW = w / maxW;
      if (normW < 0.01) continue;

      // ── Normal state ──────────────────────────────────────────────────────
      const srcA      = actsA[fromLayerIdx] ? actsA[fromLayerIdx][i] : 0;
      const srcB      = actsB && actsB[fromLayerIdx] ? actsB[fromLayerIdx][i] : null;
      const normAlpha = normW * MAX_EDGE_ALPHA;
      const normCol   = neuronColor(srcA, srcB);

      // ── Hover state ───────────────────────────────────────────────────────
      let hoverAlpha = normAlpha;
      let hoverCol   = normCol;

      if (hoverSigNorm && ha > 0) {
        if (fromLayerIdx < hovL) {
          // Before hovered layer: dim
          hoverAlpha = normW * MAX_EDGE_ALPHA * DIM_ALPHA;
          hoverCol   = neuronColor(0, null);

        } else if (fromLayerIdx === hovL) {
          // From hovered layer: only the hovered neuron's edges remain
          hoverAlpha = (i === hovK) ? normAlpha : 0;
          if (i === hovK) {
            hoverCol = neuronColor(actsA[fromLayerIdx] ? actsA[fromLayerIdx][hovK] : 0, null);
          }

        } else {
          // Downstream: scale by propagated signal
          const sigNorm = hoverSigNorm[fromLayerIdx];
          const sigVal  = sigNorm ? (sigNorm[i] || 0) : 0;
          hoverAlpha    = normW * sigVal * MAX_EDGE_ALPHA;
          hoverCol      = signalColor(sigVal);
        }
      }

      // ── Blend normal → hover ──────────────────────────────────────────────
      const alpha = normAlpha + (hoverAlpha - normAlpha) * ha;
      const col   = lerpColor(normCol, hoverCol, ha);

      if (alpha < 0.005) continue;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${alpha})`;
      ctx.lineWidth   = normW * layout.maxEdgeWidth;
      ctx.stroke();
    }
  }
}

// ── Hover interaction ─────────────────────────────────────────────────────────

function onMouseMove(e) {
  const rect = canvasEl.getBoundingClientRect();
  const mx   = e.clientX - rect.left;
  const my   = e.clientY - rect.top;

  let found = null;
  outer: for (let l = 0; l < layout.layers.length; l++) {
    for (let k = 0; k < layout.layers[l].neurons.length; k++) {
      const { x, y } = layout.layers[l].neurons[k];
      const dx = mx - x, dy = my - y;
      if (dx * dx + dy * dy <= (layout.neuronR + 3) ** 2) {
        found = { layerIdx: l, neuronIdx: k };
        break outer;
      }
    }
  }

  // Same neuron as already shown — just nudge the tooltip
  if (found && hoverState.neuron &&
      found.layerIdx  === hoverState.neuron.layerIdx &&
      found.neuronIdx === hoverState.neuron.neuronIdx) {
    updateTooltip(e.clientX, e.clientY, found);
    return;
  }

  if (found) {
    hoverState.neuron = found;
    startHoverAnim(1);
    updateTooltip(e.clientX, e.clientY, found);
  } else {
    startHoverAnim(0);
    hideTooltip();
  }
}

function onMouseLeave() {
  hideTooltip();
  startHoverAnim(0);
}

function updateTooltip(cx, cy, { layerIdx, neuronIdx }) {
  const tt    = document.getElementById("tooltip");
  const layer = network.layers[layerIdx];
  const dispA = getDisplayActivations(filterGroupA, layerIdx) || [];
  const nA    = dispA[neuronIdx] || 0;
  const nB    = mode === "compare"
    ? ((getDisplayActivations(filterGroupB, layerIdx) || [])[neuronIdx] || 0)
    : null;

  let html = `<div class="tt-title">${layer.block_label || layer.name} — unit ${neuronIdx}</div>`;
  if (layer.aggregated) {
    const s = neuronIdx * layer.bucket_size;
    const e = Math.min(s + layer.bucket_size - 1, layer.n_neurons - 1);
    html += `<div class="tt-row"><span class="tt-label">Neurons</span>
      <span class="tt-val">${s}–${e}</span></div>`;
  }
  html += `<div class="tt-row"><span class="tt-label">Activation (A)</span>
    <span class="tt-val">${nA.toFixed(4)}</span></div>`;
  if (nB !== null) {
    html += `<div class="tt-row"><span class="tt-label">Activation (B)</span>
      <span class="tt-val">${nB.toFixed(4)}</span></div>`;
  }

  tt.innerHTML     = html;
  tt.style.display = "block";
  const pad = 12;
  let tx = cx + pad, ty = cy + pad;
  if (tx + 280 > window.innerWidth)  tx = cx - 280 - pad;
  if (ty + 130 > window.innerHeight) ty = cy - 130 - pad;
  tt.style.left = tx + "px";
  tt.style.top  = ty + "px";
}

function hideTooltip() {
  document.getElementById("tooltip").style.display = "none";
}
