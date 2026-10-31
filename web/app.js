'use strict';

// ---------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------
const STORAGE_KEY = 'neutron-sim-state-v1';
const FATES = ['reflected', 'transmitted', 'absorbed'];
const FATE_COLORS = { reflected: '#e8871e', transmitted: '#7f7f7f', absorbed: '#2a6fdb' };
const PALETTE = ['#2a6fdb', '#e8871e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf'];
const MATERIAL_FIELDS = ['density', 'molar_mass', 'absorption_cross_section', 'scattering_cross_section'];
const DEFAULT_SWEEP = ['Water', 'Lead', 'Graphite', 'Beryllium'];
const PLOT_CONFIG = { responsive: true, displaylogo: false };

const state = {
  materials: [],      // {id, name, density, molar_mass, absorption_cross_section, scattering_cross_section}
  layers: [],         // {materialId, thickness}
  sweepSelected: [],  // material ids
  nextId: 1,
};
let defaultMaterials = [];
let lastStackRequest = null;
let lastSweepRequest = null;
let sweepData = null;   // {materials: [{name, color, diffusion_length, data}], thickness range}

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const child of [].concat(children)) {
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

function debounce(fn, delay) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined) return '∞';
  if (Number.isNaN(value)) return '–';
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(digits - 1);
  return Number(value.toPrecision(digits)).toString();
}

function percent(value, digits = 2) {
  return `${(100 * value).toFixed(digits)} %`;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function materialById(id) {
  return state.materials.find((m) => m.id === id);
}

function materialPayload(material) {
  const payload = { name: material.name };
  for (const field of MATERIAL_FIELDS) payload[field] = parseFloat(material[field]);
  return payload;
}

function readSettings() {
  return {
    n_neutrons: parseInt($('n-neutrons').value, 10),
    n_trials: parseInt($('n-trials').value, 10),
    seed: parseInt($('seed').value, 10),
  };
}

// ---------------------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------------------
function showTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
  document.querySelectorAll('.panel').forEach((panel) => panel.classList.toggle('active', panel.id === `panel-${name}`));
  // Plotly cannot size plots while they are hidden, so resize them once visible
  document.querySelectorAll(`#panel-${name} .js-plotly-plot`).forEach((plot) => Plotly.Plots.resize(plot));
}

// ---------------------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------------------
function renderMaterials() {
  const tbody = $('materials-table').querySelector('tbody');
  tbody.replaceChildren();
  for (const material of state.materials) {
    const nameInput = el('input', { type: 'text', value: material.name });
    nameInput.addEventListener('input', () => {
      material.name = nameInput.value;
      save();
      renderLayers();
      renderChips();
      updateDerived();
    });
    const cells = [el('td', {}, nameInput)];
    for (const field of MATERIAL_FIELDS) {
      const input = el('input', { type: 'number', min: '0', step: 'any', value: material[field] });
      input.addEventListener('input', () => {
        material[field] = input.value;
        save();
        updateDerived();
      });
      cells.push(el('td', {}, input));
    }
    const remove = el('button', { className: 'icon', title: 'Delete material', textContent: '✕' });
    remove.addEventListener('click', () => {
      state.materials = state.materials.filter((m) => m !== material);
      state.sweepSelected = state.sweepSelected.filter((id) => id !== material.id);
      save();
      renderAll();
    });
    cells.push(el('td', {}, remove));
    tbody.append(el('tr', {}, cells));
  }
}

const updateDerived = debounce(async () => {
  const tbody = $('derived-table').querySelector('tbody');
  const materials = state.materials.slice();
  let results;
  try {
    ({ results } = await api('/api/derive', { materials: materials.map(materialPayload) }));
  } catch (error) {
    tbody.replaceChildren(el('tr', {}, el('td', { className: 'error-cell', colSpan: 9 }, error.message)));
    return;
  }
  const rows = materials.map((material, i) => {
    const result = results[i];
    const nameCell = el('td', {}, material.name || '(unnamed)');
    if (result.error) {
      return el('tr', {}, [nameCell, el('td', { className: 'error-cell', colSpan: 8 }, result.error)]);
    }
    const d = result.derived;
    const values = [d.sigma_absorption, d.sigma_scattering, d.sigma_total, d.absorption_mean_free_path,
      d.scattering_mean_free_path, d.mean_free_path, d.absorption_probability, d.diffusion_length];
    return el('tr', {}, [nameCell, ...values.map((v) => el('td', {}, fmt(v)))]);
  });
  tbody.replaceChildren(...rows);
  // Highlight the rows of the editable table that are invalid
  $('materials-table').querySelectorAll('tbody tr').forEach((row, i) => {
    row.querySelectorAll('input').forEach((input) => input.classList.toggle('invalid', Boolean(results[i]?.error)));
  });
}, 250);

function addMaterial() {
  state.materials.push({
    id: state.nextId++, name: `Custom ${state.materials.length + 1}`,
    density: 1, molar_mass: 10, absorption_cross_section: 1, scattering_cross_section: 10,
  });
  save();
  renderAll();
  const inputs = $('materials-table').querySelectorAll('tbody tr:last-child input');
  if (inputs.length) inputs[0].select();
}

function resetMaterials() {
  if (!confirm('Replace all materials with the default library? Your edits will be lost.')) return;
  setDefaults();
  save();
  renderAll();
}

function setDefaults() {
  state.nextId = 1;
  state.materials = defaultMaterials.map((m) => ({ id: state.nextId++, ...m }));
  const byName = (name) => state.materials.find((m) => m.name === name);
  state.layers = ['Lead', 'Water'].filter(byName).map((name) => ({ materialId: byName(name).id, thickness: 10 }));
  state.sweepSelected = DEFAULT_SWEEP.filter(byName).map((name) => byName(name).id);
}

// ---------------------------------------------------------------------------------------
// Layer stack
// ---------------------------------------------------------------------------------------
function renderLayers() {
  const tbody = $('layers-table').querySelector('tbody');
  tbody.replaceChildren();
  state.layers.forEach((layer, index) => {
    const select = el('select');
    if (!materialById(layer.materialId)) select.append(el('option', { value: '', textContent: '(choose a material)' }));
    for (const material of state.materials) {
      select.append(el('option', { value: material.id, textContent: material.name || '(unnamed)',
        selected: material.id === layer.materialId }));
    }
    select.addEventListener('change', () => {
      layer.materialId = parseInt(select.value, 10);
      save();
      renderLayers();
    });
    const thickness = el('input', { type: 'number', min: '0.001', step: 'any', value: layer.thickness });
    thickness.addEventListener('input', () => { layer.thickness = thickness.value; save(); });
    const remove = el('button', { className: 'icon', title: 'Remove layer', textContent: '✕' });
    remove.addEventListener('click', () => {
      state.layers.splice(index, 1);
      save();
      renderLayers();
      checkStale();
    });
    tbody.append(el('tr', {}, [el('td', {}, String(index + 1)), el('td', {}, select),
      el('td', {}, thickness), el('td', {}, remove)]));
  });
  $('run-stack').disabled = state.layers.length === 0;
}

function addLayer() {
  const last = state.layers[state.layers.length - 1];
  state.layers.push({ materialId: last ? last.materialId : state.materials[0]?.id, thickness: 5 });
  save();
  renderLayers();
  checkStale();
}

function stackRequest() {
  const layers = state.layers.map((layer, i) => {
    const material = materialById(layer.materialId);
    if (!material) throw new Error(`Choose a material for layer ${i + 1}.`);
    return { material: materialPayload(material), thickness: parseFloat(layer.thickness) };
  });
  return { layers, ...readSettings(), n_tracked: parseInt($('n-tracked').value, 10) || 0 };
}

async function runStack() {
  const status = $('stack-status');
  let request;
  try {
    request = stackRequest();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
    return;
  }
  $('run-stack').disabled = true;
  status.textContent = 'Simulating…';
  status.className = 'status';
  try {
    const result = await api('/api/simulate', request);
    lastStackRequest = JSON.stringify(request);
    status.textContent = '';
    renderStackResults(request, result);
    checkStale();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
  } finally {
    $('run-stack').disabled = state.layers.length === 0;
  }
}

function renderStackResults(request, result) {
  $('stack-results').classList.remove('hidden');
  for (const fate of FATES) {
    $(`res-${fate}`).textContent = percent(result.mean[fate]);
    $(`err-${fate}`).textContent = `± ${percent(result.sem[fate])} (standard error over ${result.n_trials} trials)`;
  }
  $('stack-summary').textContent = `${result.total_neutrons.toLocaleString()} neutrons in total, `
    + `${result.collisions.toLocaleString()} real collisions, `
    + `${result.fictitious_collisions.toLocaleString()} fictitious collisions.`;

  const margin = { t: 50, b: 50, l: 60, r: 20 };
  Plotly.react('plot-fates', [{
    type: 'pie', hole: 0.4, sort: false,
    labels: FATES.map((f) => f[0].toUpperCase() + f.slice(1)),
    values: FATES.map((f) => result.mean[f]),
    marker: { colors: FATES.map((f) => FATE_COLORS[f]) },
  }], { title: { text: 'Neutron fates' }, margin }, PLOT_CONFIG);

  const labels = request.layers.map((layer, i) => `${i + 1}. ${layer.material.name} (${layer.thickness} cm)`);
  Plotly.react('plot-layers', [{
    type: 'bar', x: labels, y: result.absorbed_per_layer.map((v) => 100 * v),
    marker: { color: FATE_COLORS.absorbed },
  }], { title: { text: 'Absorbed in each layer' }, yaxis: { title: { text: '% of all neutrons' } }, margin },
  PLOT_CONFIG);

  const { edges, counts } = result.absorption_histogram;
  const centres = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);
  Plotly.react('plot-depths', [{
    type: 'bar', x: centres, y: counts, width: edges[1] - edges[0], marker: { color: FATE_COLORS.absorbed },
  }], {
    title: { text: 'Where neutrons are absorbed' },
    xaxis: { title: { text: 'Depth x (cm)' } },
    yaxis: { title: { text: 'Neutrons' }, type: 'log' },
    shapes: result.layer_ends.slice(0, -1).map((x) => ({
      type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1, line: { dash: 'dash', color: 'black', width: 1.5 },
    })),
    bargap: 0, margin,
  }, PLOT_CONFIG);

  const pathsPlot = $('plot-paths');
  if (result.paths.length === 0) {
    Plotly.purge(pathsPlot);
    pathsPlot.classList.add('hidden');
    return;
  }
  pathsPlot.classList.remove('hidden');
  let extent = 1;
  for (const path of result.paths) {
    for (const [, y, z] of path) extent = Math.max(extent, Math.abs(y), Math.abs(z));
  }
  const planes = [0, ...result.layer_ends].map((x) => ({
    type: 'mesh3d', x: [x, x, x, x], y: [-extent, extent, extent, -extent], z: [-extent, -extent, extent, extent],
    i: [0, 0], j: [1, 2], k: [2, 3], opacity: 0.15, color: 'gray', hoverinfo: 'skip', showlegend: false,
  }));
  const tracks = result.paths.map((path, i) => ({
    type: 'scatter3d', mode: 'lines+markers', name: `Neutron ${i + 1}`,
    x: path.map((p) => p[0]), y: path.map((p) => p[1]), z: path.map((p) => p[2]),
    marker: { size: 2 }, line: { width: 3 },
  }));
  Plotly.react(pathsPlot, [...planes, ...tracks], {
    title: { text: 'Neutron trajectories (planes mark layer boundaries)' },
    scene: { xaxis: { title: { text: 'x (cm)' } }, yaxis: { title: { text: 'y (cm)' } },
      zaxis: { title: { text: 'z (cm)' } }, aspectmode: 'data' },
    margin: { t: 50, b: 10, l: 10, r: 10 },
  }, PLOT_CONFIG);
}

// ---------------------------------------------------------------------------------------
// Thickness sweep
// ---------------------------------------------------------------------------------------
function renderChips() {
  const container = $('sweep-materials');
  container.replaceChildren();
  for (const material of state.materials) {
    const checkbox = el('input', { type: 'checkbox', checked: state.sweepSelected.includes(material.id) });
    checkbox.addEventListener('change', () => {
      state.sweepSelected = checkbox.checked
        ? [...state.sweepSelected, material.id]
        : state.sweepSelected.filter((id) => id !== material.id);
      save();
      checkStale();
    });
    container.append(el('label', { className: 'chip' }, [checkbox, material.name || '(unnamed)']));
  }
}

function sweepRequest() {
  const materials = state.materials.filter((m) => state.sweepSelected.includes(m.id)).map(materialPayload);
  if (materials.length === 0) throw new Error('Select at least one material.');
  return {
    materials,
    thickness_min: parseFloat($('sweep-min').value),
    thickness_max: parseFloat($('sweep-max').value),
    n_points: parseInt($('sweep-points').value, 10),
    ...readSettings(),
  };
}

async function runSweep() {
  const status = $('sweep-status');
  const progress = $('sweep-progress');
  const bar = progress.querySelector('.bar');
  let request;
  try {
    request = sweepRequest();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
    return;
  }
  $('run-sweep').disabled = true;
  status.className = 'status';
  progress.classList.remove('hidden');
  try {
    const { results: derived } = await api('/api/derive', { materials: request.materials });
    const materials = [];
    for (const [i, material] of request.materials.entries()) {
      status.textContent = `Simulating ${material.name} (${i + 1} of ${request.materials.length})…`;
      bar.style.width = `${(100 * i) / request.materials.length}%`;
      const { materials: _, ...settings } = request;
      const data = await api('/api/sweep', { material, ...settings });
      materials.push({
        name: material.name, color: PALETTE[i % PALETTE.length], data,
        diffusion_length: derived[i].derived ? derived[i].derived.diffusion_length : null,
      });
    }
    sweepData = { materials };
    lastSweepRequest = JSON.stringify(request);
    $('fit-min').value = request.thickness_min;
    $('fit-max').value = request.thickness_max;
    status.textContent = '';
    $('sweep-results').classList.remove('hidden');
    renderSweep();
    await updateFits();
    checkStale();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status error';
  } finally {
    $('run-sweep').disabled = false;
    progress.classList.add('hidden');
    bar.style.width = '0';
  }
}

function renderSweep() {
  const traces = [];
  sweepData.materials.forEach(({ name, color, data }) => {
    FATES.forEach((fate, i) => {
      traces.push({
        type: 'scatter', mode: 'lines+markers', name, legendgroup: name, showlegend: i === 0,
        x: data.thickness, y: data[fate], error_y: { type: 'data', array: data[`${fate}_sem`] },
        line: { color }, marker: { size: 5 }, xaxis: `x${i + 1}`, yaxis: `y${i + 1}`,
      });
    });
  });
  const layout = {
    title: { text: 'Neutron fates against thickness' },
    grid: { rows: 1, columns: 3, pattern: 'independent' },
    annotations: FATES.map((fate, i) => ({
      text: fate[0].toUpperCase() + fate.slice(1), showarrow: false, xref: `x${i + 1} domain`,
      yref: `y${i + 1} domain`, x: 0.5, y: 1.08, font: { size: 14 },
    })),
    margin: { t: 80, b: 50, l: 60, r: 20 },
  };
  FATES.forEach((_, i) => {
    layout[`xaxis${i + 1}`] = { title: { text: 'Thickness (cm)' } };
    layout[`yaxis${i + 1}`] = { range: [0, 1], title: i === 0 ? { text: 'Fraction of neutrons' } : undefined };
  });
  Plotly.react('plot-sweep', traces, layout, PLOT_CONFIG);
}

async function updateFits() {
  if (!sweepData) return;
  const fitMin = parseFloat($('fit-min').value);
  const fitMax = parseFloat($('fit-max').value);
  const fits = await Promise.all(sweepData.materials.map(({ data }) => api('/api/fit', {
    thickness: data.thickness, transmitted: data.transmitted, transmitted_sem: data.transmitted_sem,
    fit_min: Number.isFinite(fitMin) ? fitMin : -1e9, fit_max: Number.isFinite(fitMax) ? fitMax : 1e9,
  })));

  const traces = [];
  const rows = [];
  sweepData.materials.forEach(({ name, color, data, diffusion_length: diffusion }, i) => {
    traces.push({
      type: 'scatter', mode: 'markers', name, legendgroup: name, x: data.thickness, y: data.transmitted,
      error_y: { type: 'data', array: data.transmitted_sem }, marker: { color, size: 7 },
    });
    const { fit } = fits[i];
    if (fit) {
      const lo = Math.max(fitMin, data.thickness[0]);
      const hi = Math.min(fitMax, data.thickness[data.thickness.length - 1]);
      const x = Array.from({ length: 50 }, (_, k) => lo + ((hi - lo) * k) / 49);
      traces.push({
        type: 'scatter', mode: 'lines', name: `${name} fit`, legendgroup: name, showlegend: false,
        x, y: x.map((t) => Math.exp(fit.intercept + fit.slope * t)), line: { color },
      });
    }
    const cells = fit
      ? [fmt(fit.attenuation_length), fmt(fit.attenuation_length_error), fmt(fit.reduced_chi_squared ?? NaN),
        String(fit.n_points)]
      : ['no fit', '–', '–', '–'];
    rows.push(el('tr', {}, [name, ...cells, fmt(diffusion)].map((text) => el('td', {}, text))));
  });
  $('fit-table').querySelector('tbody').replaceChildren(...rows);
  Plotly.react('plot-fit', traces, {
    title: { text: 'Transmitted fraction (log scale) with exponential fits' },
    xaxis: { title: { text: 'Thickness (cm)' } },
    yaxis: { title: { text: 'Transmitted fraction' }, type: 'log', exponentformat: 'power' },
    margin: { t: 50, b: 50, l: 70, r: 20 },
  }, PLOT_CONFIG);
}

function downloadCsv() {
  const header = ['material', 'thickness', ...FATES.flatMap((f) => [f, `${f}_sem`])];
  const lines = [header.join(',')];
  for (const { name, data } of sweepData.materials) {
    data.thickness.forEach((t, i) => {
      const quoted = `"${name.replace(/"/g, '""')}"`;
      lines.push([quoted, t, ...FATES.flatMap((f) => [data[f][i], data[`${f}_sem`][i]])].join(','));
    });
  }
  const url = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  el('a', { href: url, download: 'thickness_sweep.csv' }).click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------------------
// Stale-result notices and start-up
// ---------------------------------------------------------------------------------------
function currentRequest(build) {
  try { return JSON.stringify(build()); } catch { return null; }
}

function checkStale() {
  $('stack-stale').classList.toggle('hidden', !lastStackRequest || currentRequest(stackRequest) === lastStackRequest);
  $('sweep-stale').classList.toggle('hidden', !lastSweepRequest || currentRequest(sweepRequest) === lastSweepRequest);
}

function renderAll() {
  renderMaterials();
  renderLayers();
  renderChips();
  updateDerived();
  checkStale();
}

async function init() {
  document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => showTab(tab.dataset.tab)));
  $('add-material').addEventListener('click', addMaterial);
  $('reset-materials').addEventListener('click', resetMaterials);
  $('add-layer').addEventListener('click', addLayer);
  $('run-stack').addEventListener('click', runStack);
  $('run-sweep').addEventListener('click', runSweep);
  $('download-csv').addEventListener('click', downloadCsv);
  const refit = debounce(() => updateFits().catch((error) => alert(error.message)), 300);
  $('fit-min').addEventListener('input', refit);
  $('fit-max').addEventListener('input', refit);
  document.addEventListener('input', checkStale);
  document.addEventListener('change', checkStale);

  try {
    ({ materials: defaultMaterials } = await api('/api/materials'));
  } catch (error) {
    document.querySelector('header').append(el('div', { className: 'notice' },
      `Could not reach the simulation server (${error.message}). Start it with "python server.py".`));
    return;
  }
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    Object.assign(state, JSON.parse(saved));
  } else {
    setDefaults();
  }
  renderAll();
}

init();
