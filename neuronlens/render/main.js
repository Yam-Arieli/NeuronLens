/* NeuronLens – main.js
 * Pure vanilla JS, no external dependencies required.
 * Data is inlined as __NETWORK__ and __ACTIVATIONS__ by the Python renderer.
 */

"use strict";

// ── Constants ────────────────────────────────────────────────────────────────

const NEURON_R = 7;
const NEURON_GAP = 4;
const LAYER_WIDTH = 120;
const LAYER_PAD_TOP = 60;
const CANVAS_PAD_LEFT = 60;
const CANVAS_PAD_RIGHT = 60;
const MAX_EDGE_ALPHA = 0.55;
const MAX_EDGE_WIDTH = 3.5;
const DIM_ALPHA = 0.07;

const COLOR_A = { r: 88,  g: 166, b: 255 };  // #58a6ff  (blue)
const COLOR_B = { r: 255, g: 100, b: 100 };  // #ff6464  (red)
// Downstream hover signal colour (warm amber)
const COLOR_SIG = { r: 255, g: 200, b: 80 };

// ── State ─────────────────────────────────────────────────────────────────────

let network    = null;
let activations = null;
let canvasEl, ctx;
let hoveredNeuron = null;  // { layerIdx, neuronIdx }  — display-unit indices
let mode = "single";       // "single" | "compare"
let filterGroupA = "default";
let filterGroupB = "default";
let layout = null;

// ── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  canvasEl = document.getElementById("main-canvas");
  ctx = canvasEl.getContext("2d");

  network    = __NETWORK__;
  activations = __ACTIVATIONS__;

  buildUI();
  buildLayout();
  render();

  canvasEl.addEventListener("mousemove", onMouseMove);
  canvasEl.addEventListener("mouseleave", onMouseLeave);
  window.addEventListener("resize", () => { buildLayout(); render(); });
});

// ── UI ────────────────────────────────────────────────────────────────────────

function buildUI() {
  const groups = Object.keys(activations.groups);
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
  selB.value = groups[Math.min(1, groups.length - 1)];
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
  const n = layers.length;

  // Fill the entire canvas-container
  const container = canvasEl.parentElement;
  const availW = Math.max(container.clientWidth,  400);
  const availH = Math.max(container.clientHeight, 300);

  // Vertical sizing: compute neuronR so neurons fill available height
  const PAD_TOP   = Math.round(availH * 0.07);
  const maxUnits  = Math.max(...layers.map(l => l.n_display_units));
  const unitH     = (availH - PAD_TOP * 2) / Math.max(maxUnits, 1);
  const neuronR   = Math.max(Math.floor(unitH * 0.42), 4);
  const neuronGap = Math.max(unitH - neuronR * 2, 2);

  // Horizontal sizing: spread layers evenly across available width
  const PAD_SIDE     = Math.round(availW * 0.03);
  const innerW       = availW - PAD_SIDE * 2 - neuronR * 2;
  const layerSpacing = n > 1 ? innerW / (n - 1) : 0;

  const canvasW = availW;
  const canvasH = availH;

  // Scale by devicePixelRatio so canvas text/lines are sharp on retina displays
  const dpr = window.devicePixelRatio || 1;
  canvasEl.width  = Math.round(canvasW * dpr);
  canvasEl.height = Math.round(canvasH * dpr);
  canvasEl.style.width  = canvasW + "px";
  canvasEl.style.height = canvasH + "px";

  const layoutLayers = layers.map((layer, li) => {
    const cx = PAD_SIDE + neuronR + li * layerSpacing;
    const nu = layer.n_display_units;
    const totalH = nu * (neuronR * 2 + neuronGap) - neuronGap;
    const startY = (canvasH - totalH) / 2;
    const neurons = [];
    for (let k = 0; k < nu; k++)
      neurons.push({ x: cx, y: startY + k * (neuronR * 2 + neuronGap) + neuronR });
    return { cx, neurons };
  });

  const labelFontSize = Math.max(Math.min(Math.round(neuronR * 0.75), 15), 10);
  const maxEdgeWidth  = Math.max(neuronR * 0.35, 2);

  layout = { layers: layoutLayers, canvasW, canvasH, dpr, neuronR, maxEdgeWidth, labelFontSize, PAD_TOP };

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

// Returns activations in DISPLAY ORDER (i.e. display unit k → its activation).
// raw[] is stored in original-neuron order; perm[display_pos] = original_idx.
function getDisplayActivations(groupKey, layerIdx) {
  const raw = getActivations(groupKey, layerIdx);
  if (!raw) return null;
  const layer = network.layers[layerIdx];
  const perm  = layer.perm;

  if (!layer.aggregated) {
    // Reorder: display position k shows original neuron perm[k]
    return perm.map(origIdx => raw[origIdx] || 0);
  }

  // Aggregated: average bucket of original neurons
  const bs = layer.bucket_size;
  const nu = layer.n_display_units;
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

// edges[fromLayerIdx] is stored in display order:
//   edges[l][display_i][display_j] = weight from display unit i (layer l)
//                                    to display unit j (layer l+1)
// For aggregated layers, average over the bucket of underlying neurons.
function getBucketWeight(fromLayerIdx, i_disp, j_disp) {
  const edges = network.edges[fromLayerIdx];
  if (!edges) return 0;
  const fromLayer = network.layers[fromLayerIdx];
  const toLayer   = network.layers[fromLayerIdx + 1];
  const bs_in  = fromLayer.bucket_size;
  const bs_out = toLayer.bucket_size;

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

// Propagate activation forward from the hovered neuron.
// Returns an array of length n_layers:
//   signal[l]  = null for layers before hoveredLayer
//   signal[hoveredLayer] = sparse array with only neuronIdx non-zero
//   signal[l > hoveredLayer] = propagated values for all display units
function computeHoverSignal(layerIdx, neuronIdx) {
  const n = network.layers.length;
  const signal = new Array(n).fill(null);

  // Seed: hovered neuron's actual (raw, un-normalised) activation
  const rawActs = getDisplayActivations(filterGroupA, layerIdx) || [];
  const seedVal = rawActs[neuronIdx] || 0;

  const seedLayer = new Array(network.layers[layerIdx].n_display_units).fill(0);
  seedLayer[neuronIdx] = seedVal;
  signal[layerIdx] = seedLayer;

  // Forward pass
  for (let l = layerIdx; l < n - 1; l++) {
    const n_out = network.layers[l + 1].n_display_units;
    const n_in  = network.layers[l].n_display_units;
    const next  = new Array(n_out).fill(0);
    for (let i = 0; i < n_in; i++) {
      const src = signal[l][i];
      if (!src || src < 1e-15) continue;
      for (let j = 0; j < n_out; j++) {
        next[j] += getBucketWeight(l, i, j) * src;
      }
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

// ── Rendering ─────────────────────────────────────────────────────────────────

function render() {
  if (!network || !layout) return;
  const { canvasW, canvasH, dpr } = layout;
  // Re-apply DPR transform on every render (resets any stale state)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvasW, canvasH);

  const n = network.layers.length;
  const actsA = network.layers.map((_, i) =>
    normalizeArray(getDisplayActivations(filterGroupA, i))
  );
  const actsB = mode === "compare"
    ? network.layers.map((_, i) => normalizeArray(getDisplayActivations(filterGroupB, i)))
    : null;

  // Compute hover signal (normalised per layer) when a neuron is hovered
  let hoverSigNorm = null;
  if (hoveredNeuron) {
    const raw = computeHoverSignal(hoveredNeuron.layerIdx, hoveredNeuron.neuronIdx);
    hoverSigNorm = raw.map(s => s ? normalizeArray(s) : null);
  }

  // Edges first (drawn behind neurons)
  for (let l = 0; l < n - 1; l++) drawEdges(l, actsA, actsB, hoverSigNorm);

  // Neurons on top
  for (let l = 0; l < n; l++) drawLayer(l, actsA[l], actsB ? actsB[l] : null, hoverSigNorm);

  // Layer labels
  ctx.font = `${layout.labelFontSize}px -apple-system, monospace`;
  ctx.textAlign = "center";
  ctx.fillStyle = "#8b949e";
  for (let l = 0; l < n; l++) {
    const lx = layout.layers[l].cx;
    ctx.fillText(network.layers[l].name, lx, layout.PAD_TOP * 0.38);
    ctx.fillText(`(${network.layers[l].n_neurons})`, lx, layout.PAD_TOP * 0.72);
  }
}

function drawLayer(layerIdx, normsA, normsB, hoverSigNorm) {
  const layerLayout = layout.layers[layerIdx];
  const nu = layerLayout.neurons.length;
  const hovL = hoveredNeuron ? hoveredNeuron.layerIdx : -1;
  const hovK = hoveredNeuron ? hoveredNeuron.neuronIdx : -1;

  for (let k = 0; k < nu; k++) {
    const { x, y } = layerLayout.neurons[k];
    let col, alpha, isHovered = false;

    if (hoverSigNorm) {
      if (layerIdx < hovL) {
        // Pre-hover layers: dim everything
        col = neuronColor(0, null);
        alpha = DIM_ALPHA;
      } else if (layerIdx === hovL) {
        if (k === hovK) {
          // The hovered neuron itself: normal colour + highlight border
          col = neuronColor(normsA ? normsA[k] : 0, normsB ? normsB[k] : null);
          alpha = col.a;
          isHovered = true;
        } else {
          col = neuronColor(0, null);
          alpha = DIM_ALPHA;
        }
      } else {
        // Downstream layers: colour by propagated signal
        const sig = hoverSigNorm[layerIdx] ? (hoverSigNorm[layerIdx][k] || 0) : 0;
        col = signalColor(sig);
        alpha = col.a;
      }
    } else {
      col = neuronColor(normsA ? normsA[k] : 0, normsB ? normsB[k] : null);
      alpha = col.a;
    }

    const r = layout.neuronR;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${col.r},${col.g},${col.b},${alpha})`;
    ctx.fill();

    // Glow
    if (alpha > 0.4) {
      const glowVal = hoverSigNorm && layerIdx > hovL
        ? (hoverSigNorm[layerIdx] ? (hoverSigNorm[layerIdx][k] || 0) : 0)
        : (normsA ? normsA[k] : 0);
      if (glowVal > 0.3) {
        ctx.shadowColor = `rgb(${col.r},${col.g},${col.b})`;
        ctx.shadowBlur = r * 0.4 + r * 0.4 * glowVal;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Border
    ctx.strokeStyle = isHovered
      ? `rgba(255,255,255,0.95)`
      : `rgba(${col.r},${col.g},${col.b},${Math.min(alpha + 0.2, 1)})`;
    ctx.lineWidth = isHovered ? 2 : 1;
    ctx.stroke();
  }
}

function drawEdges(fromLayerIdx, actsA, actsB, hoverSigNorm) {
  const toLayerIdx = fromLayerIdx + 1;
  const edges = network.edges[fromLayerIdx];
  if (!edges) return;

  const fromLayer = network.layers[fromLayerIdx];
  const toLayer   = network.layers[toLayerIdx];
  const n_in  = fromLayer.n_display_units;
  const n_out = toLayer.n_display_units;

  const hovL = hoveredNeuron ? hoveredNeuron.layerIdx : -1;
  const hovK = hoveredNeuron ? hoveredNeuron.neuronIdx : -1;

  // Pre-compute max weight for this layer pair (normalises thickness/alpha)
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

      let alpha, col;

      if (!hoverSigNorm) {
        // ── Normal (no hover) ──────────────────────────────────────────────
        alpha = normW * MAX_EDGE_ALPHA;
        const srcA = actsA[fromLayerIdx] ? actsA[fromLayerIdx][i] : 0;
        const srcB = actsB && actsB[fromLayerIdx] ? actsB[fromLayerIdx][i] : null;
        col = neuronColor(srcA, srcB);

      } else if (fromLayerIdx < hovL) {
        // ── Before hovered layer: dim ──────────────────────────────────────
        alpha = normW * MAX_EDGE_ALPHA * DIM_ALPHA;
        col = neuronColor(0, null);

      } else if (fromLayerIdx === hovL) {
        // ── From hovered layer: only show edges from the hovered neuron ────
        if (i !== hovK) continue;
        alpha = normW * MAX_EDGE_ALPHA;
        const srcA = actsA[fromLayerIdx] ? actsA[fromLayerIdx][hovK] : 0;
        col = neuronColor(srcA, null);

      } else {
        // ── Downstream layers: scale by propagated signal ──────────────────
        const sigNorm = hoverSigNorm[fromLayerIdx];
        const sigVal  = sigNorm ? (sigNorm[i] || 0) : 0;
        if (sigVal < 0.01) continue;
        alpha = normW * sigVal * MAX_EDGE_ALPHA;
        col = signalColor(sigVal);
      }

      if (alpha < 0.005) continue;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${alpha})`;
      ctx.lineWidth = normW * layout.maxEdgeWidth;
      ctx.stroke();
    }
  }
}

// ── Hover interaction ─────────────────────────────────────────────────────────

function onMouseMove(e) {
  const rect = canvasEl.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

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

  if (found && hoveredNeuron &&
      found.layerIdx === hoveredNeuron.layerIdx &&
      found.neuronIdx === hoveredNeuron.neuronIdx) {
    updateTooltip(e.clientX, e.clientY, found);
    return;
  }

  hoveredNeuron = found;
  render();
  if (found) updateTooltip(e.clientX, e.clientY, found);
  else hideTooltip();
}

function onMouseLeave() {
  hoveredNeuron = null;
  hideTooltip();
  render();
}

function updateTooltip(cx, cy, { layerIdx, neuronIdx }) {
  const tt    = document.getElementById("tooltip");
  const layer = network.layers[layerIdx];
  const dispA = getDisplayActivations(filterGroupA, layerIdx) || [];
  const nA    = dispA[neuronIdx] || 0;
  const nB    = mode === "compare"
    ? ((getDisplayActivations(filterGroupB, layerIdx) || [])[neuronIdx] || 0)
    : null;

  let html = `<div class="tt-title">${layer.name} — unit ${neuronIdx}</div>`;
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

  tt.innerHTML = html;
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
