// WebGWAYA Interactive Console JavaScript

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initGpuLifecycle();
  initSystemPoller();
  initScenarios();
  initAstAuditor();
  initStudio();
  initBenchmarkDashboard();
  initRag();
  initLora();
});

// ─────────────────────────────────────────────────────────────────────────────
// Tab Navigation
// ─────────────────────────────────────────────────────────────────────────────
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }

      if (targetId === "tab-benchmark") {
        fetchBenchmarkResults();
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU Lifecycle & VRAM Allocator
// ─────────────────────────────────────────────────────────────────────────────
let isGpuRunning = true;
let isTogglingGpu = false;

function initGpuLifecycle() {
  const btnToggle = document.getElementById("btn-gpu-toggle");
  const btnBannerStart = document.getElementById("btn-banner-start-gpu");
  const btnLoraToggle = document.getElementById("btn-lora-gpu-toggle");

  if (btnToggle) {
    btnToggle.addEventListener("click", () => handleGpuToggle());
  }
  if (btnBannerStart) {
    btnBannerStart.addEventListener("click", () => handleGpuToggle(true));
  }
  if (btnLoraToggle) {
    btnLoraToggle.addEventListener("click", () => handleGpuToggle());
  }

  // Immediate status check
  checkGpuStatus();
}

async function checkGpuStatus() {
  try {
    const res = await fetch("/api/gpu/status");
    const data = await res.json();
    updateGpuUiState(data.is_running, data);
  } catch (err) {
    console.warn("Could not query GPU status:", err);
  }
}

function updateGpuUiState(running, data) {
  isGpuRunning = !!running;

  const btnToggle = document.getElementById("btn-gpu-toggle");
  const btnLabel = document.getElementById("gpu-btn-label");
  const btnAction = document.getElementById("gpu-btn-action");
  const gpuDot = document.getElementById("gpu-dot");
  const gpuStatusText = document.getElementById("gpu-status-text");
  const ollamaStatusText = document.getElementById("ollama-status-text");
  const studioBanner = document.getElementById("studio-gpu-alert");
  const loraStatusText = document.getElementById("lora-gpu-status-text");
  const loraDot = document.getElementById("lora-gpu-dot");
  const btnLoraToggle = document.getElementById("btn-lora-gpu-toggle");

  if (isGpuRunning) {
    // ACTIVE STATE
    if (btnToggle) {
      btnToggle.className = "btn-gpu-toggle btn-gpu-active";
      btnToggle.disabled = false;
    }
    if (btnLabel) btnLabel.textContent = "GPU ACTIVE";
    if (btnAction) btnAction.textContent = "(Stop to Free VRAM)";
    if (gpuDot) gpuDot.className = "dot dot-green";

    if (data && data.gpu && data.gpu.available) {
      const memUsed = Math.round(data.gpu.memory_used_mb);
      const memTotal = Math.round(data.gpu.memory_total_mb);
      const temp = Math.round(data.gpu.temp_c);
      const pwr = Math.round(data.gpu.power_w);
      if (gpuStatusText) gpuStatusText.textContent = `${data.gpu.name} | ${memUsed}/${memTotal}MB | ${temp}°C | ${pwr}W`;
    } else if (gpuStatusText) {
      gpuStatusText.textContent = "RTX 2070 8GB | Active";
    }

    if (ollamaStatusText) {
      const count = data && data.models_count !== undefined ? data.models_count : 4;
      ollamaStatusText.textContent = `Ollama CUDA | ${count} Models`;
    }

    if (studioBanner) studioBanner.style.display = "none";

    if (loraStatusText) {
      loraStatusText.textContent = "RTX 2070 Active (Ready for Training)";
      loraStatusText.style.color = "var(--accent-green)";
    }
    if (loraDot) loraDot.className = "dot dot-green";
    if (btnLoraToggle) btnLoraToggle.textContent = "Stop GPU (Free VRAM)";

  } else {
    // STOPPED STATE
    if (btnToggle) {
      btnToggle.className = "btn-gpu-toggle btn-gpu-stopped";
      btnToggle.disabled = false;
    }
    if (btnLabel) btnLabel.textContent = "▶ START GPU";
    if (btnAction) btnAction.textContent = "(VRAM Released)";
    if (gpuDot) gpuDot.className = "dot dot-amber";

    if (gpuStatusText) {
      gpuStatusText.textContent = "GPU STOPPED (VRAM Released for Other Projects)";
    }
    if (ollamaStatusText) {
      ollamaStatusText.textContent = "Ollama Offloaded | 0 MB VRAM";
    }

    if (studioBanner) studioBanner.style.display = "flex";

    if (loraStatusText) {
      loraStatusText.textContent = "GPU STOPPED (VRAM Freed for External Workloads)";
      loraStatusText.style.color = "var(--accent-amber)";
    }
    if (loraDot) loraDot.className = "dot dot-amber";
    if (btnLoraToggle) btnLoraToggle.textContent = "Start GPU Engine";
  }
}

async function handleGpuToggle(forceStart = false) {
  if (isTogglingGpu) return;
  isTogglingGpu = true;

  const btnToggle = document.getElementById("btn-gpu-toggle");
  const btnLabel = document.getElementById("gpu-btn-label");
  const btnAction = document.getElementById("gpu-btn-action");

  const shouldStart = forceStart || !isGpuRunning;

  if (btnToggle) {
    btnToggle.disabled = true;
    if (btnLabel) btnLabel.textContent = shouldStart ? "STARTING GPU..." : "STOPPING GPU...";
    if (btnAction) btnAction.textContent = shouldStart ? "(Warming up CUDA)" : "(Releasing VRAM)";
  }

  try {
    const endpoint = shouldStart ? "/api/gpu/start" : "/api/gpu/stop";
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    updateGpuUiState(data.is_running, data);
  } catch (err) {
    console.error("GPU toggle error:", err);
    alert(`Failed to ${shouldStart ? "start" : "stop"} GPU: ` + err.message);
  } finally {
    isTogglingGpu = false;
    await checkGpuStatus();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// System Telemetry Poller
// ─────────────────────────────────────────────────────────────────────────────
function initSystemPoller() {
  async function pollStatus() {
    try {
      const res = await fetch("/api/system/status");
      const data = await res.json();

      if (data.gpu_service_running !== undefined && !isTogglingGpu) {
        updateGpuUiState(data.gpu_service_running, {
          gpu: data.gpu,
          models_count: data.ollama_models ? data.ollama_models.length : 0,
        });
      }
    } catch (err) {
      console.warn("Telemetry polling error:", err);
    }
  }

  pollStatus();
  setInterval(pollStatus, 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenarios Loader
// ─────────────────────────────────────────────────────────────────────────────
let loadedScenarios = [];

async function initScenarios() {
  const selector = document.getElementById("scenario-selector");
  const btnLoad = document.getElementById("btn-load-scenario");

  try {
    const res = await fetch("/api/scenarios");
    loadedScenarios = await res.json();

    loadedScenarios.forEach((sc, idx) => {
      const opt = document.createElement("option");
      opt.value = sc.id;
      opt.textContent = `[${sc.difficulty}] ${sc.title}`;
      selector.appendChild(opt);
    });

    // Auto-select first scenario
    if (loadedScenarios.length > 0) {
      selector.value = loadedScenarios[0].id;
      loadScenarioToStudio(loadedScenarios[0]);
    }
  } catch (err) {
    console.error("Failed to load scenarios:", err);
  }

  btnLoad.addEventListener("click", () => {
    const selectedId = selector.value;
    const sc = loadedScenarios.find((s) => s.id === selectedId);
    if (sc) {
      loadScenarioToStudio(sc);
    }
  });
}

function loadScenarioToStudio(sc) {
  document.getElementById("studio-goal").value = sc.goal;
  document.getElementById("studio-test-spec").value = sc.test_spec;
  document.getElementById("studio-code-output").value = sc.starter_code || "";
}

// ─────────────────────────────────────────────────────────────────────────────
// AST ZeroStubAudit Probe
// ─────────────────────────────────────────────────────────────────────────────
function initAstAuditor() {
  const btn = document.getElementById("btn-run-ast-audit");
  const input = document.getElementById("ast-test-input");
  const resultDiv = document.getElementById("ast-audit-result");

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Auditing AST...";
    try {
      const res = await fetch("/api/gwaya/ast-audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: input.value }),
      });
      const data = await res.json();

      if (data.is_clean) {
        resultDiv.innerHTML = `
          <span class="badge badge-green">AST SHIELD: PASSED</span>
          <span style="color: var(--text-muted); margin-left: 0.5rem;">CC: ${data.cyclomatic_complexity} | Energy: ${data.energy}</span>
        `;
      } else {
        resultDiv.innerHTML = `
          <span class="badge badge-red">AST SHIELD: REJECTED (E=1e6)</span>
          <span style="color: var(--accent-rose); margin-left: 0.5rem;">${data.violations.join("; ")}</span>
        `;
      }
    } catch (err) {
      resultDiv.textContent = `Audit error: ${err.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Run AST Audit";
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Live AI Coding Studio
// ─────────────────────────────────────────────────────────────────────────────
function initStudio() {
  const btnGen = document.getElementById("btn-generate-code");
  const btnEval = document.getElementById("btn-eval-only");
  const goalInput = document.getElementById("studio-goal");
  const testSpecInput = document.getElementById("studio-test-spec");
  const codeOutput = document.getElementById("studio-code-output");
  const modelSelect = document.getElementById("studio-model-select");
  const armSelect = document.getElementById("studio-arm-select");
  const tokCounter = document.getElementById("tok-counter");

  btnGen.addEventListener("click", async () => {
    const goal = goalInput.value.trim();
    if (!goal) {
      alert("Please enter a task goal or select a scenario.");
      return;
    }

    if (!isGpuRunning) {
      alert("GPU Engine is currently STOPPED to free VRAM for another project. Click 'Start GPU' in the header or the alert banner to activate inference.");
      tokCounter.className = "badge badge-amber";
      tokCounter.textContent = "GPU Stopped";
      return;
    }

    btnGen.disabled = true;
    btnGen.innerHTML = `<span>⏳ Synthesizing on RTX GPU...</span>`;
    tokCounter.className = "badge badge-amber";
    tokCounter.textContent = "Running...";

    try {
      const res = await fetch("/api/gwaya/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: modelSelect.value,
          goal: goal,
          test_spec: testSpecInput.value.trim(),
          mode: armSelect.value,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || res.statusText);
      }

      const data = await res.json();
      codeOutput.value = data.code || "";

      tokCounter.className = "badge badge-purple";
      tokCounter.textContent = `${data.tokens_per_s || 0} tok/s | ${data.latency_ms || 0}ms`;

      if (data.evaluation) {
        updateEvaluatorUI(data.evaluation);
      }
    } catch (err) {
      alert("Generation failed: " + err.message);
      tokCounter.className = "badge badge-red";
      tokCounter.textContent = "Error";
    } finally {
      btnGen.disabled = false;
      btnGen.innerHTML = `<span>🚀 Synthesize & Verify with GWAYA</span>`;
    }
  });

  btnEval.addEventListener("click", async () => {
    const code = codeOutput.value.trim();
    if (!code) {
      alert("No code in output area to evaluate.");
      return;
    }

    btnEval.disabled = true;
    btnEval.textContent = "Evaluating...";

    try {
      const res = await fetch("/api/gwaya/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: code,
          test_spec: testSpecInput.value.trim(),
          goal: goalInput.value.trim(),
        }),
      });
      const data = await res.json();
      updateEvaluatorUI(data);
    } catch (err) {
      alert("Evaluation failed: " + err.message);
    } finally {
      btnEval.disabled = false;
      btnEval.textContent = "🛡️ Laya Evaluator Only";
    }
  });
}

function updateEvaluatorUI(ev) {
  const badge = document.getElementById("laya-verdict-badge");
  const testsVal = document.getElementById("eval-tests");
  const ccVal = document.getElementById("eval-cc");
  const energyVal = document.getElementById("eval-energy");
  const latencyVal = document.getElementById("eval-latency");
  const feedback = document.getElementById("eval-feedback");

  testsVal.textContent = `${ev.tests_passed || 0}/${ev.tests_total || 0}`;
  ccVal.textContent = ev.cyclomatic_complexity || 1;
  energyVal.textContent = ev.energy >= 1000000 ? "10^6 (Pain)" : (ev.energy || 0);
  latencyVal.textContent = ev.duration_ms || 0;

  if (ev.verified) {
    badge.className = "badge badge-green";
    badge.textContent = "VERIFIED_TESTS (100%)";
    feedback.style.color = "var(--accent-green)";
    feedback.textContent = `All assertions satisfied in isolated sandbox with nonce verification. Zero stubs detected.`;
  } else if (ev.level === "well_formed") {
    badge.className = "badge badge-amber";
    badge.textContent = "WELL_FORMED";
    feedback.style.color = "var(--accent-amber)";
    feedback.textContent = `Code parses and has no stubs, but test verification failed or was incomplete.`;
  } else {
    badge.className = "badge badge-red";
    badge.textContent = "UNVERIFIED";
    feedback.style.color = "var(--accent-rose)";
    const err = ev.stub_violations?.length
      ? `Stub violations: ${ev.stub_violations.join(", ")}`
      : ev.error_message || "Verification gate rejected candidate.";
    feedback.textContent = err;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark & Multi-Model Gains Dashboard
// ─────────────────────────────────────────────────────────────────────────────
let chartPass = null;
let chartDeltas = null;

function initBenchmarkDashboard() {
  document.getElementById("btn-refresh-benchmark").addEventListener("click", fetchBenchmarkResults);
  fetchBenchmarkResults();
}

async function fetchBenchmarkResults() {
  try {
    const res = await fetch("/api/benchmark/results");
    const data = await res.json();
    if (!data.available || !data.models) return;

    renderBenchmarkCharts(data.models);
    renderBenchmarkSummaryCards(data.models);
    renderBenchmarkTasksTable(data.models);
  } catch (err) {
    console.error("Benchmark data fetch error:", err);
  }
}

function renderBenchmarkCharts(modelsData) {
  const modelKeys = Object.keys(modelsData);
  if (modelKeys.length === 0) return;

  const labels = modelKeys.map((k) => k.replace("qwen2.5-coder:", "Qwen-"));
  const a0Pass = modelKeys.map((k) => modelsData[k].pass_at_1?.A0_raw || 0);
  const a2Pass = modelKeys.map((k) => modelsData[k].pass_at_1?.A2_best_of_3 || 0);
  const a3Pass = modelKeys.map((k) => modelsData[k].pass_at_1?.A3_gwaya_v3 || 0);
  const deltas = modelKeys.map((k) => modelsData[k].deltas_percentage_points?.A3_vs_A0 || 0);
  const tokSpeed = modelKeys.map((k) => modelsData[k].performance?.tokens_per_s || 0);

  // Chart 1: Pass@1
  const ctxPass = document.getElementById("chart-pass-rates").getContext("2d");
  if (chartPass) chartPass.destroy();

  chartPass = new Chart(ctxPass, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: "A0: Raw Baseline", data: a0Pass, backgroundColor: "rgba(148, 163, 184, 0.4)", borderColor: "#94a3b8", borderWidth: 1 },
        { label: "A2: Best-of-3", data: a2Pass, backgroundColor: "rgba(56, 189, 248, 0.5)", borderColor: "#38bdf8", borderWidth: 1 },
        { label: "A3: GWAYA v3 Repair", data: a3Pass, backgroundColor: "rgba(16, 185, 129, 0.7)", borderColor: "#10b981", borderWidth: 1 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        y: { min: 0, max: 100, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
      },
    },
  });

  // Chart 2: Deltas & Tok/s
  const ctxDeltas = document.getElementById("chart-deltas").getContext("2d");
  if (chartDeltas) chartDeltas.destroy();

  chartDeltas = new Chart(ctxDeltas, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: "Δ(A3 - A0) Gain (pts)", data: deltas, backgroundColor: "rgba(139, 92, 246, 0.7)", borderColor: "#8b5cf6", borderWidth: 1 },
        { label: "Throughput (tok/s)", data: tokSpeed, backgroundColor: "rgba(245, 158, 11, 0.6)", borderColor: "#f59e0b", borderWidth: 1 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
      },
    },
  });
}

function renderBenchmarkSummaryCards(modelsData) {
  const container = document.getElementById("benchmark-summary-cards");
  container.innerHTML = "";

  Object.values(modelsData).forEach((m) => {
    const card = document.createElement("div");
    card.className = "glass-card";
    card.style.marginBottom = "0";

    const a0 = m.pass_at_1?.A0_raw || 0;
    const a3 = m.pass_at_1?.A3_gwaya_v3 || 0;
    const delta = m.deltas_percentage_points?.A3_vs_A0 || 0;
    const repaired = m.self_repair_efficiency?.repaired_by_a3_count || 0;
    const totalFailed = m.self_repair_efficiency?.a0_failed_count || 0;

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <h4 style="font-size: 0.95rem; font-weight: 700;">${m.model}</h4>
        <span class="badge badge-green">+${delta} pts</span>
      </div>
      <div style="font-size: 1.6rem; font-weight: 800; color: var(--accent-cyan);">${a3}%</div>
      <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.75rem;">A0: ${a0}% &rarr; A3: ${a3}%</div>
      <div style="font-size: 0.8rem; color: var(--text-dim);">
        <div>Cured Failures: <strong>${repaired}/${totalFailed}</strong></div>
        <div>Throughput: <strong>${m.performance?.tokens_per_s || 0} tok/s</strong></div>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderBenchmarkTasksTable(modelsData) {
  const tbody = document.querySelector("#benchmark-tasks-table tbody");
  // Pick the most recent or largest model rows
  const modelKeys = Object.keys(modelsData);
  if (modelKeys.length === 0) return;

  const currentModel = modelsData[modelKeys[modelKeys.length - 1]];
  if (!currentModel.rows || currentModel.rows.length === 0) return;

  tbody.innerHTML = "";
  currentModel.rows.forEach((r) => {
    const tr = document.createElement("tr");

    const a0Badge = r.A0?.solved ? `<span class="badge badge-green">PASS</span>` : `<span class="badge badge-red">FAIL</span>`;
    const a2Badge = r.A2?.solved ? `<span class="badge badge-green">PASS</span>` : `<span class="badge badge-red">FAIL</span>`;
    const a3Badge = r.A3?.solved ? `<span class="badge badge-green">PASS</span>` : `<span class="badge badge-red">FAIL</span>`;
    const rounds = r.A3?.repair_attempts || 1;
    const energy = r.A3?.energy || "-";

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 600;">${r.id}</td>
      <td><span class="badge badge-purple">${r.category}</span></td>
      <td>${a0Badge}</td>
      <td>${a2Badge}</td>
      <td>${a3Badge}</td>
      <td style="font-family: var(--font-mono);">${rounds}</td>
      <td style="font-family: var(--font-mono);">${energy}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Private RAG Vault (ChromaDB)
// ─────────────────────────────────────────────────────────────────────────────
function initRag() {
  const btnIndex = document.getElementById("btn-index-folder");
  const folderInput = document.getElementById("rag-folder-path");
  const statusDiv = document.getElementById("rag-index-status");

  const btnQuery = document.getElementById("btn-query-rag");
  const queryInput = document.getElementById("rag-query-input");
  const resultsContainer = document.getElementById("rag-results-container");

  btnIndex.addEventListener("click", async () => {
    const folder = folderInput.value.trim();
    if (!folder) return;

    btnIndex.disabled = true;
    btnIndex.textContent = "Indexing into ChromaDB...";
    statusDiv.textContent = "Scanning directory and building embeddings...";

    try {
      const res = await fetch("/api/rag/index-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: folder }),
      });
      const data = await res.json();
      statusDiv.innerHTML = `
        <span class="badge badge-green">INDEXED</span>
        <span>Indexed ${data.indexed_chunks} code chunks. Total Vault Size: ${data.total_collection_size} chunks.</span>
      `;
    } catch (err) {
      statusDiv.innerHTML = `<span class="badge badge-red">Error: ${err.message}</span>`;
    } finally {
      btnIndex.disabled = false;
      btnIndex.textContent = "Index Folder in Chroma";
    }
  });

  btnQuery.addEventListener("click", async () => {
    const q = queryInput.value.trim();
    if (!q) return;

    btnQuery.disabled = true;
    btnQuery.textContent = "Searching...";

    try {
      const res = await fetch("/api/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, n_results: 3 }),
      });
      const data = await res.json();

      if (!data.results || data.results.length === 0) {
        resultsContainer.innerHTML = `<p style="color: var(--text-dim);">No matching chunks found in vault.</p>`;
      } else {
        resultsContainer.innerHTML = data.results
          .map(
            (item) => `
            <div style="background: #060910; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                <span style="font-weight: 600; color: var(--accent-cyan); font-size: 0.85rem;">${item.file} (${item.lines})</span>
                <span class="badge badge-purple">Similarity: ${(item.similarity * 100).toFixed(1)}%</span>
              </div>
              <pre class="code-block" style="max-height: 140px; margin: 0; padding: 0.5rem; font-size: 0.8rem;">${escapeHtml(item.code)}</pre>
            </div>
          `
          )
          .join("");
      }
    } catch (err) {
      resultsContainer.innerHTML = `<span class="badge badge-red">Search failed: ${err.message}</span>`;
    } finally {
      btnQuery.disabled = false;
      btnQuery.textContent = "Retrieve Relevant Code";
    }
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ─────────────────────────────────────────────────────────────────────────────
// LoRA Admin Panel
// ─────────────────────────────────────────────────────────────────────────────
function initLora() {
  const btnGenConfig = document.getElementById("btn-generate-lora-config");
  const datasetInfo = document.getElementById("lora-dataset-info");
  const configOutput = document.getElementById("lora-config-output");

  async function loadDatasets() {
    try {
      const res = await fetch("/api/lora/datasets");
      const datasets = await res.json();
      datasetInfo.innerHTML = datasets
        .map(
          (d) => `
          <div style="margin-bottom: 0.35rem;">
            <strong>${d.name}:</strong> ${d.samples} available instruction-correction pairs (${d.recommended_for}).
          </div>
        `
        )
        .join("");
    } catch (err) {
      console.warn("Dataset loading err:", err);
    }
  }

  loadDatasets();

  btnGenConfig.addEventListener("click", async () => {
    btnGenConfig.disabled = true;
    btnGenConfig.textContent = "Building LoRA Spec...";

    const payload = {
      base_model: document.getElementById("lora-base-model").value,
      r: parseInt(document.getElementById("lora-rank").value, 10),
      lora_alpha: parseInt(document.getElementById("lora-alpha").value, 10),
      learning_rate: parseFloat(document.getElementById("lora-lr").value),
      quantization: document.getElementById("lora-quant").value,
      dataset_name: document.getElementById("lora-dataset-select").value,
    };

    try {
      const res = await fetch("/api/lora/generate-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      configOutput.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      configOutput.textContent = `Failed to generate LoRA config: ${err.message}`;
    } finally {
      btnGenConfig.disabled = false;
      btnGenConfig.textContent = "⚡ Generate LoRA Config & Training Script";
    }
  });
}
