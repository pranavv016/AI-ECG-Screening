/* ==========================================================================
   AI-Assisted ECG Screening System — Part 3: ECG visualization

   Scope:
   - Reuse the Part 2 CSV loading/validation flow.
   - Visualize the raw ECG with Plotly.js.
   - Show sample count, duration, and sampling rate when available.
   - Update the graph automatically for every valid upload.
   - Show the demo ECG on the graph.
   - No preprocessing, feature extraction, signal-quality scoring, or ML.
   ========================================================================== */

(() => {
  'use strict';

  const MIN_SAMPLES = 10;
  const DEMO_PATH = '../data/sample_ecg.csv';
  const API_BASE = '/api';
  const LEAD_NAMES = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
  const SUPPORTED_EXTENSIONS = ['csv', 'mat', 'edf', 'txt'];
  const GRID_MINOR = 10;
  const GRID_MAJOR = 50;
  const PRELOADED_DATASETS = {
    'mitbih-normal': {
      name: 'MIT-BIH-derived benchmark — sample beat',
      url: 'https://raw.githubusercontent.com/mdespinoza/demo-repo-applicaiton/main/datasets/ecg_data/mitbih_train_sample.csv',
      kind: 'mitbih'
    },
    'ptb-normal': {
      name: 'PTB Diagnostic Database — sample beat',
      url: 'https://raw.githubusercontent.com/mdespinoza/demo-repo-applicaiton/main/datasets/ecg_data/ptbdb_normal_sample.csv',
      kind: 'ptb'
    }
  };

  const els = {
    fileInput: document.getElementById('ecgFileInput'),
    demoBtn: document.getElementById('demoBtn'),
    datasetSelect: document.getElementById('datasetSelect'),
    loadDatasetBtn: document.getElementById('loadDatasetBtn'),
    leadSelect: document.getElementById('leadSelect'),
    clearBtn: document.getElementById('clearBtn'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    dropzone: document.getElementById('dropzone'),
    fileMeta: document.getElementById('fileMeta'),
    signalFilename: document.getElementById('signalFilename'),
    signalSamples: document.getElementById('signalSamples'),
    signalDuration: document.getElementById('signalDuration'),
    signalSamplingRate: document.getElementById('signalSamplingRate'),
    signalMessage: document.getElementById('signalMessage'),
    waveformStage: document.getElementById('waveformStage'),
    waveformPlaceholder: document.getElementById('waveformPlaceholder'),
    ecgPlot: document.getElementById('ecgPlot'),
    ecgGridCanvas: document.getElementById('ecgGridCanvas'),
    ecgMarkerCanvas: document.getElementById('ecgMarkerCanvas'),
    leadTag: document.getElementById('leadTag'),
    pipelineStatus: document.getElementById('pipelineStatus'),
    pipelineStatusText: document.getElementById('pipelineStatusText'),
    processedWaveformStage: document.getElementById('processedWaveformStage'),
    processedEcgPlot: document.getElementById('processedEcgPlot'),
    preprocessTag: document.getElementById('preprocessTag'),
    preprocessingMessage: document.getElementById('preprocessingMessage'),
    featureHeartRate: document.getElementById('featureHeartRate'),
    featureRPeaks: document.getElementById('featureRPeaks'),
    featureMeanRR: document.getElementById('featureMeanRR'),
    featureMeanAmplitude: document.getElementById('featureMeanAmplitude'),
    featureStd: document.getElementById('featureStd'),
    featureMin: document.getElementById('featureMin'),
    featureMax: document.getElementById('featureMax'),
    featureRange: document.getElementById('featureRange'),
    featureRms: document.getElementById('featureRms'),
    featureQuality: document.getElementById('featureQuality'),
    featureStatusMessage: document.getElementById('featureStatusMessage'),
    resultClassification: document.getElementById('resultClassification'),
    resultPattern: document.getElementById('resultPattern'),
    resultConfidence: document.getElementById('resultConfidence'),
    mlStatusBadge: document.getElementById('mlStatusBadge'),
    mlResultNote: document.getElementById('mlResultNote')
  };

  let currentSignal = null;
  let fiducialMarkers = [];
  let hoveredFiducial = null;

  init();

  function init() {
    bindEvents();

    if (typeof Plotly === 'undefined') {
      showError('Plotly.js could not be loaded. Check the internet connection and reload the page.');
      return;
    }

    // Part 3 requirement: demo ECG is visible on the graph by default.
    loadDemoSignal();
  }

  function bindEvents() {
    els.demoBtn.addEventListener('click', loadDemoSignal);
    els.loadDatasetBtn.addEventListener('click', loadSelectedDataset);
    els.leadSelect.addEventListener('change', () => selectLead(Number(els.leadSelect.value)));
    els.clearBtn.addEventListener('click', clearSignal);
    els.analyzeBtn.addEventListener('click', analyzeCurrentSignal);

    els.fileInput.addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) handleFile(file);
    });

    els.dropzone.addEventListener('click', () => els.fileInput.click());

    els.dropzone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        els.fileInput.click();
      }
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      els.dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        els.dropzone.classList.add('is-dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      els.dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        els.dropzone.classList.remove('is-dragover');
      });
    });

    els.dropzone.addEventListener('drop', (event) => {
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    window.addEventListener('resize', () => {
      resizeEcgCanvases();
      drawEcgGrid();
      if (currentSignal && typeof Plotly !== 'undefined') {
        Plotly.Plots.resize(els.ecgPlot);
        window.requestAnimationFrame(() => drawFiducials());
      }
    });

    resizeEcgCanvases();
    drawEcgGrid();
  }

  async function loadSelectedDataset() {
    const key = els.datasetSelect.value;
    if (key === 'demo') {
      await loadDemoSignal();
      return;
    }
    const dataset = PRELOADED_DATASETS[key];
    if (!dataset) return;
    els.loadDatasetBtn.disabled = true;
    els.loadDatasetBtn.textContent = 'Loading…';
    try {
      const response = await fetch(dataset.url, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Sample dataset could not be loaded (HTTP ${response.status}).`);
      const text = await response.text();
      const parsed = parseBenchmarkDataset(text, dataset.kind);
      loadSignal(parsed.samples, {
        filename: dataset.name,
        time: null,
        sampleRate: 125,
        duration: (parsed.samples.length - 1) / 125,
        isDemo: true,
        leads: [{ name: 'Lead I', samples: parsed.samples, time: null, sampleRate: 125 }]
      });
      els.fileMeta.textContent = `DEMO CLINICAL DATASET — ${dataset.name}`;
      els.signalMessage.textContent = `DEMO CLINICAL DATASET — ${dataset.name}. Benchmark sample only; not a live patient record.`;
      els.signalMessage.classList.add('is-demo');
    } catch (error) {
      showError(`${error.message || 'The selected dataset could not be loaded.'} Internet access is required for remote benchmark samples.`);
    } finally {
      els.loadDatasetBtn.disabled = false;
      els.loadDatasetBtn.textContent = 'Load selected dataset';
    }
  }

  function parseBenchmarkDataset(text, kind) {
    const rows = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(Boolean);
    if (!rows.length) throw new Error('The benchmark dataset is empty.');
    const values = rows[0].split(',').map(Number);
    if (kind === 'mitbih') {
      if (values.length !== 188) throw new Error('Unexpected MIT-BIH-derived sample format.');
      values.pop(); // final column is the benchmark class label
    }
    validateSampleCount(values);
    return { samples: values };
  }

  function handleFile(file) {
    if (!file) return;

    const extension = file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      showError('Unsupported ECG file type. Please use .csv, .mat, .edf, or .txt.');
      return;
    }

    if (file.size === 0) {
      showError('The selected ECG file is empty.');
      return;
    }

    els.fileMeta.textContent = `Reading ${file.name}…`;

    if (extension === 'csv' || extension === 'txt') {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = extension === 'csv'
            ? parseEcgCsv(String(reader.result))
            : parseEcgText(String(reader.result));
          loadSignal(parsed.samples, {
            filename: file.name,
            time: parsed.time,
            sampleRate: parsed.sampleRate,
            duration: parsed.duration,
            isDemo: false,
            leads: [{ name: parsed.leadName || 'Lead I', samples: parsed.samples, time: parsed.time, sampleRate: parsed.sampleRate }]
          });
        } catch (error) {
          showError(error.message || 'The ECG text file could not be loaded.');
        }
      };
      reader.onerror = () => showError('The ECG file could not be read.');
      reader.readAsText(file);
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      try {
        if (extension === 'mat') {
          const parsed = parseMatFile(reader.result);
          loadSignal(parsed.samples, { ...parsed, filename: file.name, isDemo: false });
        } else {
          const parsed = parseEdfFile(reader.result);
          loadSignal(parsed.samples, { ...parsed, filename: file.name, isDemo: false });
        }
      } catch (error) {
        showError(error.message || `The .${extension.toUpperCase()} ECG file could not be loaded.`);
      }
    };
    reader.onerror = () => showError('The binary ECG file could not be read.');
    reader.readAsArrayBuffer(file);
  }

  function parseEcgText(text) {
    const normalized = text.replace(/^\uFEFF/, '');
    const lines = normalized.split(/\r?\n/);
    const rows = lines
      .map((line, index) => ({ line: line.trim(), lineNumber: index + 1 }))
      .filter(({ line }) => line && !line.startsWith('#'));

    if (!rows.length) throw new Error('The TXT file is empty — no ECG data was found.');

    const first = rows[0].line.split(/[\s,;\t]+/).map(v => v.trim().toLowerCase());
    const hasHeader = first.some(v => ['time', 'amplitude', 'ecg', 'signal', 'lead'].includes(v));
    const dataRows = hasHeader ? rows.slice(1) : rows;
    if (!dataRows.length) throw new Error('The TXT file contains a header but no ECG samples.');

    const firstParts = dataRows[0].line.split(/[\s,;\t]+/).filter(Boolean);
    const time = firstParts.length >= 2 ? [] : null;
    const samples = [];

    for (const row of dataRows) {
      const parts = row.line.split(/[\s,;\t]+/).filter(Boolean);
      if (parts.length < 1 || parts.length > 2) {
        throw new Error(`Line ${row.lineNumber}: expected one amplitude value or time + amplitude.`);
      }
      if (parts.some(v => !Number.isFinite(Number(v)))) {
        throw new Error(`Line ${row.lineNumber}: ECG values must be valid numbers.`);
      }
      if (time) { time.push(Number(parts[0])); samples.push(Number(parts[1])); }
      else samples.push(Number(parts[0]));
    }

    validateSampleCount(samples);
    const meta = deriveTimeMetadata(time);
    return { samples, time, ...meta };
  }

  function validateSampleCount(samples) {
    if (!samples.length) throw new Error('No ECG samples were found.');
    if (samples.length < MIN_SAMPLES) throw new Error(`Insufficient ECG data. At least ${MIN_SAMPLES} samples are required.`);
    if (!samples.every(Number.isFinite)) throw new Error('ECG contains invalid numeric values.');
  }

  function deriveTimeMetadata(time) {
    if (!time) return { sampleRate: null, duration: null };
    for (let i = 1; i < time.length; i += 1) {
      if (!(time[i] > time[i - 1])) throw new Error('Time values must be strictly increasing.');
    }
    const duration = time[time.length - 1] - time[0];
    if (!(duration > 0)) throw new Error('The time column must span a positive duration.');
    const intervals = time.slice(1).map((v, i) => v - time[i]);
    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const maxRelativeDeviation = Math.max(...intervals.map(v => Math.abs(v - mean) / mean));
    const sampleRate = mean > 0 && maxRelativeDeviation <= 0.001 ? 1 / mean : null;
    return { sampleRate, duration };
  }

  function parseMatFile(buffer) {
    if (!window.mat4js || typeof window.mat4js.read !== 'function') {
      throw new Error('MAT support could not be initialized. Reload the page with internet access.');
    }
    const parsed = window.mat4js.read(buffer);
    const candidates = [];
    for (const [name, value] of Object.entries(parsed.data || {})) {
      const numeric = flattenNumericArray(value);
      if (numeric.length >= MIN_SAMPLES) candidates.push({ name, samples: numeric });
    }
    if (!candidates.length) throw new Error('No numeric ECG vector with enough samples was found in this MAT file. MATLAB v7.3/HDF5 MAT files are not supported by the browser reader.');
    const leads = candidates.map((c, i) => ({ name: normalizeLeadName(c.name, i), samples: c.samples, time: null, sampleRate: null }));
    const first = leads[0];
    return { samples: first.samples, time: null, sampleRate: null, duration: null, leads };
  }

  function flattenNumericArray(value) {
    const out = [];
    const walk = (v) => {
      if (typeof v === 'number' && Number.isFinite(v)) { out.push(v); return; }
      if (Array.isArray(v)) { v.forEach(walk); return; }
      if (ArrayBuffer.isView(v)) { Array.from(v).forEach(walk); return; }
      if (v && typeof v === 'object' && 'r' in v && Number.isFinite(Number(v.r))) out.push(Number(v.r));
    };
    walk(value);
    return out;
  }

  function parseEdfFile(buffer) {
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 256) throw new Error('The EDF file is too small to contain a valid header.');
    const ascii = (start, length) => new TextDecoder('ascii').decode(bytes.slice(start, start + length)).trim();
    const number = (start, length, fallback = NaN) => { const n = Number(ascii(start, length)); return Number.isFinite(n) ? n : fallback; };
    const ns = Math.trunc(number(252, 4));
    const headerBytes = Math.trunc(number(184, 8));
    const recordDuration = number(244, 8);
    const records = Math.trunc(number(236, 8));
    if (!Number.isFinite(ns) || ns < 1 || !Number.isFinite(headerBytes) || headerBytes < 256 || !Number.isFinite(recordDuration) || recordDuration <= 0) {
      throw new Error('Invalid or unsupported EDF header.');
    }
    if (bytes.length < headerBytes) throw new Error('EDF header is incomplete.');

    const labels = Array.from({ length: ns }, (_, i) => ascii(256 + i * 16, 16));
    const units = Array.from({ length: ns }, (_, i) => ascii(256 + ns * 96 + i * 80, 8));
    const physicalMin = Array.from({ length: ns }, (_, i) => number(256 + ns * 104 + i * 8, 8));
    const physicalMax = Array.from({ length: ns }, (_, i) => number(256 + ns * 112 + i * 8, 8));
    const digitalMin = Array.from({ length: ns }, (_, i) => number(256 + ns * 120 + i * 8, 8));
    const digitalMax = Array.from({ length: ns }, (_, i) => number(256 + ns * 128 + i * 8, 8));
    const samplesPerRecord = Array.from({ length: ns }, (_, i) => Math.trunc(number(256 + ns * 216 + i * 8, 8)));
    const recordSize = samplesPerRecord.reduce((a, b) => a + b, 0);
    const availableRecords = Math.floor((bytes.length - headerBytes) / (recordSize * 2));
    const recordCount = records > 0 ? Math.min(records, availableRecords) : availableRecords;
    if (recordCount < 1 || recordSize < 1) throw new Error('EDF contains no readable data records.');

    const leads = [];
    let byteOffset = headerBytes;
    for (let channel = 0; channel < ns; channel += 1) {
      const label = labels[channel] || `Channel ${channel + 1}`;
      if (/annotation/i.test(label)) continue;
      const count = samplesPerRecord[channel];
      if (!Number.isFinite(count) || count < 1) continue;
      const values = new Float64Array(recordCount * count);
      let outIndex = 0;
      let channelOffset = 0;
      for (let c = 0; c < channel; c += 1) channelOffset += samplesPerRecord[c];
      for (let r = 0; r < recordCount; r += 1) {
        const start = headerBytes + (r * recordSize + channelOffset) * 2;
        for (let j = 0; j < count; j += 1) {
          const digital = new DataView(buffer).getInt16(start + j * 2, true);
          const denom = digitalMax[channel] - digitalMin[channel];
          const physical = denom ? physicalMin[channel] + (digital - digitalMin[channel]) * (physicalMax[channel] - physicalMin[channel]) / denom : digital;
          values[outIndex++] = physical;
        }
      }
      validateSampleCount(Array.from(values.slice(0, Math.min(values.length, MIN_SAMPLES + 1))));
      const fs = count / recordDuration;
      const time = Array.from({ length: values.length }, (_, i) => i / fs);
      leads.push({ name: normalizeLeadName(label, channel), samples: Array.from(values), time, sampleRate: fs, duration: time[time.length - 1] || 0, unit: units[channel] });
    }
    if (!leads.length) throw new Error('No readable ECG signal channel was found in the EDF file.');
    const first = leads[0];
    return { ...first, leads };
  }

  function normalizeLeadName(name, index) {
    const trimmed = String(name || '').trim();
    return trimmed || LEAD_NAMES[index] || `Channel ${index + 1}`;
  }

  /*
   * Supported CSV formats from Part 2:
   *
   * Format A:
   *   time,amplitude
   *   0.00,0.12
   *
   * Format B:
   *   amplitude
   *   0.12
   */
  function parseEcgCsv(text) {
    const normalized = text.replace(/^\uFEFF/, '');
    const lines = normalized.split(/\r?\n/);

    while (lines.length && lines[lines.length - 1].trim() === '') {
      lines.pop();
    }

    const meaningful = lines
      .map((line, index) => ({
        line: line.trim(),
        lineNumber: index + 1
      }))
      .filter(({ line }) => line !== '' && !line.startsWith('#'));

    if (meaningful.length === 0) {
      throw new Error('The CSV file is empty — no ECG data was found.');
    }

    const header = meaningful[0].line
      .split(',')
      .map((value) => value.trim().toLowerCase());

    const isTimeAmplitude =
      header.length === 2 &&
      header[0] === 'time' &&
      header[1] === 'amplitude';

    const isAmplitudeOnly =
      header.length === 1 &&
      header[0] === 'amplitude';

    if (!isTimeAmplitude && !isAmplitudeOnly) {
      throw new Error(
        'Unsupported CSV format. Use either "time,amplitude" or a single "amplitude" column.'
      );
    }

    const dataEntries = [];
    let headerFound = false;

    for (const entry of lines.map((line, index) => ({
      line: line.trim(),
      lineNumber: index + 1
    }))) {
      if (!headerFound) {
        if (entry.line === '' || entry.line.startsWith('#')) continue;
        headerFound = true;
        continue;
      }

      // Comments are allowed, but blank data lines are intentionally validated.
      if (entry.line.startsWith('#')) continue;
      dataEntries.push(entry);
    }

    const samples = [];
    const time = isTimeAmplitude ? [] : null;

    for (const { line, lineNumber } of dataEntries) {
      const parts = line.split(',').map((value) => value.trim());

      if (isTimeAmplitude) {
        if (parts.length !== 2) {
          throw new Error(`Line ${lineNumber}: expected time and amplitude values.`);
        }

        if (parts[0] === '' || parts[1] === '') {
          throw new Error(`Line ${lineNumber}: missing time or ECG amplitude value.`);
        }

        const t = Number(parts[0]);
        const amplitude = Number(parts[1]);

        if (!Number.isFinite(t) || !Number.isFinite(amplitude)) {
          throw new Error(`Line ${lineNumber}: time and amplitude must be valid numbers.`);
        }

        time.push(t);
        samples.push(amplitude);
      } else {
        if (parts.length !== 1) {
          throw new Error(`Line ${lineNumber}: expected one amplitude value.`);
        }

        if (parts[0] === '') {
          throw new Error(`Line ${lineNumber}: missing ECG amplitude value.`);
        }

        const amplitude = Number(parts[0]);

        if (!Number.isFinite(amplitude)) {
          throw new Error(`Line ${lineNumber}: ECG amplitude must be a valid number.`);
        }

        samples.push(amplitude);
      }
    }

    if (samples.length === 0) {
      throw new Error('No ECG samples were found after the header.');
    }

    if (samples.length < MIN_SAMPLES) {
      throw new Error(`Insufficient ECG data. At least ${MIN_SAMPLES} samples are required.`);
    }

    let sampleRate = null;
    let duration = null;

    if (time) {
      for (let i = 1; i < time.length; i += 1) {
        if (!(time[i] > time[i - 1])) {
          throw new Error('Time values must be strictly increasing.');
        }
      }

      duration = time[time.length - 1] - time[0];

      if (!(duration > 0)) {
        throw new Error('The time column must span a positive duration.');
      }

      const intervals = [];

      for (let i = 1; i < time.length; i += 1) {
        intervals.push(time[i] - time[i - 1]);
      }

      const meanInterval =
        intervals.reduce((sum, value) => sum + value, 0) / intervals.length;

      const maxRelativeDeviation = Math.max(
        ...intervals.map(
          (interval) => Math.abs(interval - meanInterval) / meanInterval
        )
      );

      // Report a sampling rate only when timestamps are effectively uniform.
      if (
        meanInterval > 0 &&
        Number.isFinite(meanInterval) &&
        maxRelativeDeviation <= 0.001
      ) {
        sampleRate = 1 / meanInterval;
      }
    }

    return { samples, time, sampleRate, duration };
  }

  async function loadDemoSignal() {
    try {
      const response = await fetch(DEMO_PATH, { cache: 'no-store' });

      if (!response.ok) {
        throw new Error(`Demo ECG file could not be loaded (HTTP ${response.status}).`);
      }

      const text = await response.text();
      const parsed = parseEcgCsv(text);

      loadSignal(parsed.samples, {
        filename: 'sample_ecg.csv',
        time: parsed.time,
        sampleRate: parsed.sampleRate,
        duration: parsed.duration,
        isDemo: true
      });
    } catch (error) {
      showError(
        `${error.message} If you opened index.html directly, run the project with a local web server.`
      );
    }
  }

  function loadSignal(samples, metadata) {
    const leads = (metadata.leads && metadata.leads.length ? metadata.leads : [{
      name: metadata.leadName || 'Lead I',
      samples,
      time: metadata.time || null,
      sampleRate: metadata.sampleRate ?? inferSampleRate(metadata.time),
      duration: metadata.duration ?? inferDuration(metadata.time)
    }]).map((lead, index) => ({
      name: normalizeLeadName(lead.name, index),
      samples: lead.samples.slice(),
      time: lead.time ? lead.time.slice() : null,
      sampleRate: lead.sampleRate ?? inferSampleRate(lead.time),
      duration: lead.duration ?? inferDuration(lead.time)
    }));

    currentSignal = {
      samples: samples.slice(),
      time: metadata.time ? metadata.time.slice() : null,
      sampleRate: metadata.sampleRate ?? inferSampleRate(metadata.time),
      duration: metadata.duration ?? inferDuration(metadata.time),
      filename: metadata.filename || 'ECG signal',
      isDemo: Boolean(metadata.isDemo),
      leads,
      selectedLeadIndex: 0
    };

    els.clearBtn.disabled = false;
    els.waveformStage.classList.add('has-data');

    updateSignalInfo(currentSignal);
    renderRawEcg(currentSignal);
    resetPart4Display();
    resetMlDisplay();

    populateLeadSelector(currentSignal.leads);
    els.leadTag.textContent = currentSignal.isDemo
      ? `Lead: ${currentSignal.leads[0].name} · demo`
      : `Lead: ${currentSignal.leads[0].name}`;

    els.fileMeta.textContent = currentSignal.isDemo
      ? 'DEMO ECG — SIMULATED/EDUCATIONAL DATA'
      : `${currentSignal.filename} — ${currentSignal.samples.length} samples`;

    els.signalMessage.classList.remove('is-error');
    els.signalMessage.classList.toggle('is-demo', currentSignal.isDemo);
    els.signalMessage.textContent = currentSignal.isDemo
      ? 'DEMO ECG — SIMULATED/EDUCATIONAL DATA. Not a real patient signal.'
      : 'ECG loaded successfully. Raw signal only — no preprocessing or prediction is performed.';

    setStatus(
      true,
      currentSignal.isDemo
        ? 'Demo ECG displayed'
        : 'ECG displayed — raw signal'
    );
  }


  function populateLeadSelector(leads) {
    els.leadSelect.innerHTML = '';
    leads.forEach((lead, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = lead.name;
      els.leadSelect.appendChild(option);
    });
    els.leadSelect.value = '0';
    els.leadSelect.disabled = leads.length <= 1;
  }

  function selectLead(index) {
    if (!currentSignal || !currentSignal.leads[index]) return;
    const lead = currentSignal.leads[index];
    currentSignal.selectedLeadIndex = index;
    currentSignal.samples = lead.samples.slice();
    currentSignal.time = lead.time ? lead.time.slice() : null;
    currentSignal.sampleRate = lead.sampleRate ?? inferSampleRate(lead.time);
    currentSignal.duration = lead.duration ?? inferDuration(lead.time);
    fiducialMarkers = [];
    updateSignalInfo(currentSignal);
    renderRawEcg(currentSignal);
    resetPart4Display();
    resetMlDisplay();
    els.leadTag.textContent = `Lead: ${lead.name}${currentSignal.isDemo ? ' · demo' : ''}`;
    els.signalMessage.classList.toggle('is-demo', currentSignal.isDemo);
    els.signalMessage.textContent = currentSignal.isDemo
      ? 'DEMO ECG — SIMULATED/EDUCATIONAL DATA. Not a real patient signal.'
      : 'ECG lead selected successfully. Raw signal only until analysis is run.';
  }

  async function analyzeCurrentSignal() {
    if (!currentSignal) {
      showError('Load an ECG signal first.');
      return;
    }

    els.analyzeBtn.disabled = true;
    els.analyzeBtn.textContent = 'Analyzing…';
    resetPart4Display();
    resetMlDisplay();

    try {
      // Step 1: send the uploaded ECG through the FastAPI validation layer.
      const uploadResponse = await fetch(`${API_BASE}/upload-ecg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          samples: currentSignal.samples,
          time: currentSignal.time,
          sampling_rate_hz: currentSignal.sampleRate,
          filename: currentSignal.filename
        })
      });
      const uploadData = await uploadResponse.json();
      if (!uploadResponse.ok) {
        throw new Error(uploadData.detail || `Upload validation failed (HTTP ${uploadResponse.status}).`);
      }

      // Step 2: Python preprocessing + feature extraction.
      await analyzeOnBackend(currentSignal);

      // Step 3: only the exact training representation can enter the ML model.
      await predictOnBackend(currentSignal);

      setStatus(true, 'Analysis complete');
    } catch (error) {
      els.mlStatusBadge.textContent = 'Analysis error';
      els.mlStatusBadge.className = 'badge badge--pending';
      els.mlResultNote.textContent = error.message || 'The analysis could not be completed.';
      setStatus(false, 'Analysis failed');
    } finally {
      els.analyzeBtn.disabled = false;
      els.analyzeBtn.textContent = 'Analyze ECG';
    }
  }

  async function predictOnBackend(signalData) {
    if (signalData.samples.length !== 187) {
      els.mlStatusBadge.textContent = '187 samples required';
      els.mlStatusBadge.className = 'badge badge--pending';
      els.mlResultNote.textContent =
        'The trained model accepts one 187-sample heartbeat segment, matching the training dataset. This ECG was processed and displayed, but it was not silently truncated or given a model prediction.';
      return;
    }

    els.mlStatusBadge.textContent = 'Running trained model…';
    els.mlStatusBadge.className = 'badge badge--pending';
    els.mlResultNote.textContent = 'Sending the 187-sample heartbeat to the saved Python ML model…';

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        samples: signalData.samples,
        time: signalData.time,
        sampling_rate_hz: signalData.sampleRate,
        filename: signalData.filename
      })
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || `Prediction failed (HTTP ${response.status}).`);
    }

    els.resultClassification.textContent = payload.classification || '—';
    els.resultPattern.textContent = 'Benchmark class: ' + (payload.classification || '—');
    els.resultConfidence.textContent =
      payload.confidence !== undefined
        ? `${(payload.confidence * 100).toFixed(1)}% model probability`
        : 'Not provided by model';

    els.mlStatusBadge.textContent = 'Model active';
    els.mlStatusBadge.className = 'badge badge--success';
    els.mlResultNote.textContent =
      'This is an educational/research screening result and is not a medical diagnosis.';

    // The predict endpoint also returns the exact processed beat used for ML.
    if (Array.isArray(payload.processed_ecg)) {
      renderModelProcessedEcg(payload.processed_ecg, signalData.sampleRate || 125);
    }
  }

  function resetMlDisplay() {
    els.resultClassification.textContent = '—';
    els.resultPattern.textContent = '—';
    els.resultConfidence.textContent = '—';
    els.mlStatusBadge.textContent = 'Ready';
    els.mlStatusBadge.className = 'badge badge--pending';
    els.mlResultNote.textContent =
      'Click “Analyze ECG” to send the signal through FastAPI, preprocessing, feature extraction, and the trained ML model.';
  }

  function renderModelProcessedEcg(processed, fs) {
    if (!Array.isArray(processed) || processed.length === 0 || typeof Plotly === 'undefined') return;
    const x = processed.map((_, i) => i / fs);
    Plotly.react(els.processedEcgPlot, [{
      x, y: processed, type: 'scattergl', mode: 'lines',
      name: 'ML input after training-matched preprocessing',
      line: { color: '#0E7C86', width: 1.6, simplify: false },
      hovertemplate: '<b>Time</b>: %{x:.4f} s<br><b>ML input</b>: %{y:.4f}<extra></extra>'
    }], {
      autosize: true,
      margin: { l: 70, r: 24, t: 18, b: 58 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(255,255,255,0)',
      showlegend: false,
      xaxis: { title: { text: 'Time (s)' }, showgrid: true, automargin: true },
      yaxis: { title: { text: 'Standardized amplitude' }, showgrid: true, automargin: true }
    }, { responsive: true, displaylogo: false });
  }

  function inferSampleRate(time) {
    if (!time || time.length < 2) return null;

    const intervals = [];

    for (let i = 1; i < time.length; i += 1) {
      const interval = time[i] - time[i - 1];
      if (!(interval > 0)) return null;
      intervals.push(interval);
    }

    const mean =
      intervals.reduce((sum, value) => sum + value, 0) / intervals.length;

    return mean > 0 ? 1 / mean : null;
  }

  function inferDuration(time) {
    if (!time || time.length < 2) return null;

    const duration = time[time.length - 1] - time[0];
    return duration > 0 ? duration : null;
  }

  function updateSignalInfo(signal) {
    els.signalFilename.textContent = signal.filename || '—';
    els.signalSamples.textContent = String(signal.samples.length);

    els.signalDuration.textContent =
      signal.duration != null
        ? `${signal.duration.toFixed(3)} s`
        : 'Not available';

    els.signalSamplingRate.textContent =
      signal.sampleRate != null
        ? `${signal.sampleRate.toFixed(3)} Hz`
        : 'Not available';
  }

  function renderRawEcg(signal) {
    if (typeof Plotly === 'undefined') {
      showError('Plotly.js is not available, so the ECG cannot be visualized.');
      return;
    }

    const x = signal.time
      ? signal.time
      : signal.samples.map((_, index) => index);

    const xTitle = signal.time
      ? 'Time (s)'
      : 'Time (sample index; sampling rate unavailable)';

    const customData = signal.samples.map((_, index) => index);

    const trace = {
      x,
      y: signal.samples,
      customdata: customData,
      type: 'scattergl',
      mode: 'lines',
      name: signal.isDemo ? 'Demo ECG' : signal.filename,
      line: {
        color: '#0E7C86',
        width: 1.6,
        simplify: false
      },
      hovertemplate:
        '<b>Time</b>: %{x:.4f}<br>' +
        '<b>Amplitude</b>: %{y:.6g}<br>' +
        '<b>Sample</b>: %{customdata}<extra></extra>'
    };

    const layout = {
      autosize: true,
      margin: { l: 70, r: 24, t: 18, b: 58 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(255,255,255,0)',
      hovermode: 'x unified',
      showlegend: false,
      dragmode: 'zoom',
      xaxis: {
        title: { text: xTitle },
        showgrid: false,
        zeroline: false,
        fixedrange: false,
        automargin: true,
        rangeslider: { visible: false }
      },
      yaxis: {
        title: { text: 'ECG amplitude' },
        showgrid: false,
        zeroline: true,
        zerolinecolor: 'rgba(14, 124, 134, 0.25)',
        fixedrange: false,
        automargin: true
      }
    };

    const config = {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'raw_ecg_signal',
        scale: 2
      }
    };

    Plotly.react(els.ecgPlot, [trace], layout, config)
      .then(() => {
        Plotly.Plots.resize(els.ecgPlot);
        if (!els.ecgPlot.__fiducialHandlers && typeof els.ecgPlot.on === 'function') {
          els.ecgPlot.on('plotly_hover', handlePlotlyHover);
          els.ecgPlot.on('plotly_unhover', () => { hoveredFiducial = null; drawFiducials(); });
          els.ecgPlot.__fiducialHandlers = true;
        }
        resizeEcgCanvases();
        drawEcgGrid();
        drawFiducials();
      })
      .catch((error) => {
        showError(`ECG visualization failed: ${error.message || error}`);
      });
  }

  function resizeEcgCanvases() {
    const stage = els.waveformStage;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 3));
    [els.ecgGridCanvas, els.ecgMarkerCanvas].forEach(canvas => {
      if (!canvas) return;
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    });
  }

  function drawEcgGrid() {
    const canvas = els.ecgGridCanvas;
    if (!canvas) return;
    resizeEcgCanvases();
    const ctx = canvas.getContext('2d');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-surface-sunken').trim() || '#f5f8f9';
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += GRID_MINOR) {
      ctx.strokeStyle = 'rgba(210, 85, 105, 0.08)';
      ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, height); ctx.stroke();
    }
    for (let y = 0; y <= height; y += GRID_MINOR) {
      ctx.strokeStyle = 'rgba(210, 85, 105, 0.08)';
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(width, y + 0.5); ctx.stroke();
    }
    for (let x = 0; x <= width; x += GRID_MAJOR) {
      ctx.strokeStyle = 'rgba(210, 85, 105, 0.16)';
      ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, height); ctx.stroke();
    }
    for (let y = 0; y <= height; y += GRID_MAJOR) {
      ctx.strokeStyle = 'rgba(210, 85, 105, 0.16)';
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(width, y + 0.5); ctx.stroke();
    }
  }

  function estimateFiducials(features) {
    const peaks = Array.isArray(features?.r_peak_indices) ? features.r_peak_indices : [];
    const fs = Number(currentSignal?.sampleRate);
    if (!currentSignal || !peaks.length || !(fs > 0)) return [];
    const result = [];
    const offsets = [
      ['P-start', -0.25],
      ['P-peak', -0.18],
      ['QRS-onset', -0.06],
      ['R-peak', 0],
      ['T-end', 0.28]
    ];
    peaks.forEach((r, beat) => {
      offsets.forEach(([label, seconds]) => {
        const index = Math.round(r + seconds * fs);
        if (index >= 0 && index < currentSignal.samples.length) {
          result.push({ label, index, beat, estimated: label !== 'R-peak' });
        }
      });
    });
    return result;
  }

  function drawFiducials() {
    const canvas = els.ecgMarkerCanvas;
    if (!canvas) return;
    resizeEcgCanvases();
    const ctx = canvas.getContext('2d');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    if (!fiducialMarkers.length || !currentSignal || !els.ecgPlot || !els.ecgPlot._fullLayout) return;
    const xaxis = els.ecgPlot._fullLayout.xaxis;
    const yaxis = els.ecgPlot._fullLayout.yaxis;
    if (!xaxis || !yaxis || typeof xaxis.l2p !== 'function' || typeof yaxis.l2p !== 'function') return;
    const xValues = currentSignal.time || currentSignal.samples.map((_, i) => i);
    const xOffset = xaxis._offset || 0;
    const yOffset = yaxis._offset || 0;
    const labelColors = { 'P-start': '#7C3AED', 'P-peak': '#7C3AED', 'QRS-onset': '#D97706', 'R-peak': '#B3261E', 'T-end': '#2563EB' };

    fiducialMarkers.forEach((marker) => {
      const x = xOffset + xaxis.l2p(xValues[marker.index]);
      const y = yOffset + yaxis.l2p(currentSignal.samples[marker.index]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > width || y < 0 || y > height) return;
      const color = labelColors[marker.label] || '#0F2A3D';
      const active = hoveredFiducial === marker;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.beginPath(); ctx.moveTo(x, Math.max(0, y - 18)); ctx.lineTo(x, Math.min(height, y + 18)); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, active ? 5 : 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.font = '10px IBM Plex Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${marker.label}${marker.estimated ? ' ~' : ''}`, x, Math.max(12, y - 23));
    });
  }

  function handlePlotlyHover(event) {
    if (!fiducialMarkers.length || !currentSignal || !event?.points?.length) return;
    const point = event.points[0];
    const index = Number(point.pointIndex);
    hoveredFiducial = fiducialMarkers.find(marker => marker.index === index) || null;
    drawFiducials();
  }


  async function analyzeOnBackend(signalData) {
    setStatus(true, 'Processing ECG…');
    els.preprocessTag.textContent = 'Processing…';
    els.preprocessingMessage.textContent = 'Sending the ECG to the Python preprocessing pipeline…';

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          samples: signalData.samples,
          time: signalData.time,
          sampling_rate_hz: signalData.sampleRate,
          filename: signalData.filename
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Backend returned HTTP ${response.status}.`);
      }

      renderProcessedEcg(data, signalData);
      renderFeatures(data.features);
      setStatus(true, 'ECG processed — features extracted');
    } catch (error) {
      els.preprocessTag.textContent = 'Processing failed';
      els.preprocessingMessage.textContent =
        `Backend processing failed: ${error.message || error}. Start the FastAPI backend and reload.`;
      els.preprocessingMessage.classList.add('is-error');
      setStatus(false, 'Backend processing failed');
      throw error;
    }
  }

  function resetPart4Display() {
    els.processedWaveformStage.classList.remove('has-data');
    els.preprocessTag.textContent = 'Waiting for ECG';
    els.preprocessingMessage.classList.remove('is-error');
    els.preprocessingMessage.textContent = 'No preprocessing has been performed yet.';
    if (typeof Plotly !== 'undefined' && els.processedEcgPlot) {
      Plotly.purge(els.processedEcgPlot);
    }
    [
      els.featureHeartRate, els.featureRPeaks, els.featureMeanRR,
      els.featureMeanAmplitude, els.featureStd, els.featureMin,
      els.featureMax, els.featureRange, els.featureRms, els.featureQuality
    ].forEach((el) => { if (el) el.textContent = '—'; });
    els.featureStatusMessage.textContent =
      'Features are unavailable until an ECG is successfully processed.';
  }

  function formatFeature(value, digits = 4) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      return '—';
    }
    return Number(value).toFixed(digits).replace(/\.?0+$/, '');
  }

  function setMetric(el, value, digits = 4) {
    if (!el) return;
    const unit = el.querySelector('.metric-card__unit');
    const unitText = unit ? unit.outerHTML : '';
    el.innerHTML = `${formatFeature(value, digits)} ${unitText}`.trim();
  }

  function renderProcessedEcg(data, originalSignal) {
    const processed = data.processed_signal;
    if (!Array.isArray(processed) || processed.length === 0) {
      throw new Error('Backend returned no processed ECG samples.');
    }

    const x = originalSignal.time
      ? originalSignal.time
      : processed.map((_, index) => index / data.sampling_rate_hz);

    const trace = {
      x,
      y: processed,
      type: 'scattergl',
      mode: 'lines',
      name: 'Processed ECG',
      line: { color: '#0E7C86', width: 1.6, simplify: false },
      hovertemplate:
        '<b>Time</b>: %{x:.4f} s<br>' +
        '<b>Processed amplitude</b>: %{y:.4f}<extra></extra>'
    };

    const layout = {
      autosize: true,
      margin: { l: 70, r: 24, t: 18, b: 58 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(255,255,255,0)',
      hovermode: 'x unified',
      showlegend: false,
      dragmode: 'zoom',
      xaxis: { title: { text: 'Time (s)' }, showgrid: true, gridcolor: 'rgba(210,85,105,0.18)', automargin: true },
      yaxis: { title: { text: 'Normalized amplitude' }, showgrid: true, gridcolor: 'rgba(210,85,105,0.18)', zeroline: true, automargin: true }
    };

    Plotly.react(els.processedEcgPlot, [trace], layout, {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    }).then(() => Plotly.Plots.resize(els.processedEcgPlot));

    els.processedWaveformStage.classList.add('has-data');
    els.preprocessTag.textContent =
      `${data.preprocessing_info.band_pass.low_hz}–${data.preprocessing_info.band_pass.high_hz} Hz`;
    els.preprocessingMessage.classList.remove('is-error');
    els.preprocessingMessage.textContent =
      `Processed at ${formatFeature(data.sampling_rate_hz, 2)} Hz using zero-phase filtering. ` +
      `Baseline wander reduction: ${data.preprocessing_info.baseline_wander_reduction}.`;
  }

  function renderFeatures(features) {
    setMetric(els.featureHeartRate, features.heart_rate_bpm, 2);
    setMetric(els.featureRPeaks, features.r_peak_count, 0);
    setMetric(els.featureMeanRR, features.mean_rr_interval_s, 3);
    setMetric(els.featureMeanAmplitude, features.mean_amplitude, 4);
    setMetric(els.featureStd, features.standard_deviation, 4);
    setMetric(els.featureMin, features.minimum_amplitude, 4);
    setMetric(els.featureMax, features.maximum_amplitude, 4);
    setMetric(els.featureRange, features.signal_range, 4);
    setMetric(els.featureRms, features.rms, 4);

    const quality = features.signal_quality;
    els.featureQuality.textContent =
      quality && quality.status ? quality.status : '—';

    const rrStatus = features.feature_status?.rr_intervals || 'unknown';
    const hrStatus = features.feature_status?.heart_rate || 'unknown';
    els.featureStatusMessage.textContent =
      `R-peaks: ${features.feature_status?.r_peaks || 'unknown'}. ` +
      `RR intervals: ${rrStatus}. Heart rate: ${hrStatus}. ` +
      `Signal quality is a non-clinical heuristic, not a diagnosis.`;

    // Fiducials are overlaid only after feature extraction. R-peak positions
    // come from the backend; the surrounding P/QRS/T positions are explicitly
    // marked as estimated visual aids rather than clinical measurements.
    fiducialMarkers = estimateFiducials(features);
    els.ecgMarkerCanvas.classList.toggle('is-interactive', fiducialMarkers.length > 0);
    window.requestAnimationFrame(() => drawFiducials());
  }

  function clearSignal() {
    currentSignal = null;
    fiducialMarkers = [];
    hoveredFiducial = null;
    els.ecgMarkerCanvas.classList.remove('is-interactive');
    els.fileInput.value = '';
    els.fileMeta.textContent = 'No file selected';
    els.leadTag.textContent = 'Lead: —';
    els.clearBtn.disabled = true;
    els.waveformStage.classList.remove('has-data');

    if (typeof Plotly !== 'undefined' && els.ecgPlot) {
      Plotly.purge(els.ecgPlot);
    }

    els.signalFilename.textContent = '—';
    els.signalSamples.textContent = '—';
    els.signalDuration.textContent = '—';
    els.signalSamplingRate.textContent = '—';
    els.signalMessage.textContent = 'Load a CSV ECG signal to populate these fields.';
    els.signalMessage.classList.remove('is-error', 'is-demo');
    resetPart4Display();

    setStatus(false, 'No signal loaded');
  }

  function showError(message) {
    currentSignal = null;
    fiducialMarkers = [];
    hoveredFiducial = null;
    els.ecgMarkerCanvas.classList.remove('is-interactive');
    els.waveformStage.classList.remove('has-data');

    els.fileMeta.textContent = message;
    els.signalMessage.textContent = message;
    els.signalMessage.classList.remove('is-demo');
    els.signalMessage.classList.add('is-error');

    setStatus(false, 'ECG not loaded');

    if (typeof Plotly !== 'undefined' && els.ecgPlot) {
      Plotly.purge(els.ecgPlot);
    }
  }

  function setStatus(active, text) {
    els.pipelineStatus.classList.toggle('is-active', Boolean(active));
    els.pipelineStatusText.textContent = text;
  }
})();
