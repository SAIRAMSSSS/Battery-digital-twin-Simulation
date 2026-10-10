"""
Interactive Web Dashboard & Real-Time Telemetry Server for Lithium-Ion Battery Digital Twin.
Serves on http://localhost:8000 using Python's built-in ThreadingHTTPServer.
"""

import sys
import os
import json
import time
import threading
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from config import BatteryChemistry, get_default_config
from ocv_model import OCVModel
from battery_digital_twin import BatteryDigitalTwin, SyntheticBatterySimulator


# Global Simulation Engine State
class SimulationEngine:
    def __init__(self, chemistry=BatteryChemistry.NMC, cycle_count=350, initial_soc=0.90):
        self.lock = threading.Lock()
        self.chemistry = chemistry
        self.cycle_count = cycle_count
        self.initial_soc = initial_soc
        self.user_current = None
        self.reset()

    def reset(self):
        with self.lock:
            self.config = get_default_config(self.chemistry)
            self.simulator = SyntheticBatterySimulator(
                params=self.config,
                chemistry=self.chemistry,
                initial_soc=self.initial_soc,
                cycle_count=self.cycle_count,
                seed=int(time.time()) % 100000,
            )
            # Intentionally initialize EKF with prior offset to showcase convergence
            self.twin = BatteryDigitalTwin(
                params=self.config,
                chemistry=self.chemistry,
                initial_soc=self.initial_soc - 0.20,
                initial_cycle_count=self.cycle_count,
            )
            self.step_idx = 0
            self.running = True
            self.speed = 1.0  # steps per second
            self.override_mode = None  # cruise, accel, regen, rest, fault_temp, fault_v
            self.user_current = None
            self.history = []

    def configure(self, soc=None, soh=None, cycle_count=None, chemistry=None, temp=None, current=None, r0_mohm=None, reset_history=True):
        with self.lock:
            if chemistry and chemistry != self.chemistry.value:
                self.chemistry = BatteryChemistry.LFP if chemistry == "LFP" else BatteryChemistry.NMC
                self.config = get_default_config(self.chemistry)
                self.simulator.ocv_model = OCVModel(self.chemistry)
                self.twin.ocv_model = OCVModel(self.chemistry)
                self.twin.params = self.config
                self.simulator.params = self.config

            if soc is not None:
                soc_val = max(0.01, min(1.0, float(soc) / 100.0))
                self.initial_soc = soc_val
                self.simulator.true_soc = soc_val
                self.twin.ekf.x[0] = soc_val
                self.twin.ekf.soc_coulomb_counting = soc_val

            if soh is not None:
                soh_val = max(60.0, min(100.0, float(soh)))
                self.simulator.true_soh_pct = soh_val
                self.simulator.true_capacity_ah = self.config.nominal_capacity_ah * (soh_val / 100.0)
                self.twin.soh_estimator.q_current = self.config.nominal_capacity_ah * (soh_val / 100.0)
                if cycle_count is None:
                    loss = max(0.01, 100.0 - soh_val)
                    cycle_est = int((loss / self.config.aging_alpha) ** (1.0 / self.config.aging_beta))
                    self.cycle_count = cycle_est
                    self.simulator.cycle_count = cycle_est
                    self.twin.cycle_count = cycle_est
                    self.twin.soh_estimator.cycle_count = cycle_est

            if cycle_count is not None:
                self.cycle_count = int(cycle_count)
                self.simulator.cycle_count = int(cycle_count)
                self.twin.cycle_count = int(cycle_count)
                self.twin.soh_estimator.cycle_count = int(cycle_count)

            if temp is not None:
                self.simulator.true_temp = float(temp)

            if r0_mohm is not None:
                r0_val = float(r0_mohm) / 1000.0
                self.simulator.true_r0_base = r0_val
                self.twin.soh_estimator.rls_r0.r0_est = r0_val

            if current is not None:
                if str(current).lower() == "auto":
                    self.user_current = None
                else:
                    self.user_current = float(current)

            if reset_history:
                self.step_idx = 0
                self.history = []
                self.twin.history = []

    def set_override(self, mode):
        with self.lock:
            self.override_mode = mode

    def step(self):
        with self.lock:
            self.step_idx += 1
            dt = 1.0

            # Determine load current
            if self.override_mode == "fault_temp":
                current = 6.0
                self.simulator.true_temp = 58.0  # Force critical temperature
            elif self.override_mode == "fault_v":
                current = 7.5
                self.simulator.true_soc = 0.02  # Force critical low SOC
            elif self.user_current is not None:
                current = self.user_current + float(np_noise(0.02))
            elif self.override_mode == "cruise":
                current = 2.0
            elif self.override_mode == "accel":
                current = 5.5
            elif self.override_mode == "regen":
                current = -1.8
            elif self.override_mode == "rest":
                current = 0.0
            else:
                # Default dynamic stress test pattern
                phase = (self.step_idx // 30) % 4
                if phase == 0:
                    current = 2.0 + float(np_noise(0.15))
                elif phase == 1:
                    current = 4.5 + float(np_noise(0.20))
                elif phase == 2:
                    current = -1.5 + float(np_noise(0.15))
                else:
                    current = 0.0

            # Single ground truth step
            eta = self.config.coulombic_efficiency if current < 0 else 1.0
            coulomb_cap = self.simulator.true_capacity_ah * 3600.0
            self.simulator.true_soc -= (eta * current * dt) / coulomb_cap
            self.simulator.true_soc = max(0.01, min(1.0, self.simulator.true_soc))

            tau1 = self.config.tau1
            exp_d = 0.960789
            self.simulator.true_v1 = exp_d * self.simulator.true_v1 + self.config.r1_nominal * (1.0 - exp_d) * current

            # Lumped thermal
            r0_t = self.config.get_temperature_adjusted_r0(self.simulator.true_temp)
            q_gen = (current ** 2) * r0_t + (self.simulator.true_v1 ** 2) / self.config.r1_nominal
            q_cool = self.config.heat_transfer_ha * (self.simulator.true_temp - self.config.ambient_temp_c)
            self.simulator.true_temp += (dt / self.config.thermal_mass_cp) * (q_gen - q_cool)

            r0_eff = r0_t + (self.simulator.true_r0_base - self.config.r0_nominal)
            ocv_true = self.simulator.ocv_model.ocv(self.simulator.true_soc)
            v_true = ocv_true - self.simulator.true_v1 - current * r0_eff

            # Sensor noise
            import random
            v_meas = v_true + random.gauss(0, 0.005)
            i_meas = current + random.gauss(0, 0.025)
            t_meas = self.simulator.true_temp + random.gauss(0, 0.20)

            # Digital Twin real-time update
            telemetry = self.twin.update(
                voltage=v_meas,
                current=i_meas,
                temperature=t_meas,
                dt=dt,
                cycle_count=self.cycle_count,
            )

            # Payload
            payload = {
                **telemetry,
                "step": self.step_idx,
                "true_soc": round(self.simulator.true_soc * 100.0, 3),
                "true_voltage": round(v_true, 4),
                "r0_est_mohm": round(self.twin.history[-1]["r0_est_ohm"] * 1000.0, 2),
                "v_model": self.twin.history[-1]["v_model"],
                "active_mode": self.override_mode or ("USER_LOAD" if self.user_current is not None else "DYNAMIC_CYCLE"),
                "cycle_count": self.cycle_count,
            }

            self.history.append(payload)
            if len(self.history) > 200:
                self.history.pop(0)

            return payload


def np_noise(std):
    import random
    return random.gauss(0, std)


engine = SimulationEngine()


# Background simulation ticker thread
def simulation_ticker():
    while True:
        if engine.running:
            try:
                engine.step()
            except Exception as e:
                print(f"[Engine Error] {e}")
        interval = 1.0 / max(0.1, engine.speed)
        time.sleep(interval)


# Web Request Handler
class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with engine.lock:
                current_payload = engine.history[-1] if engine.history else {}
            self.wfile.write(json.dumps(current_payload).encode("utf-8"))

        elif path == "/api/history":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with engine.lock:
                hist = list(engine.history)
            self.wfile.write(json.dumps(hist).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if path == "/api/control":
            action = data.get("action")
            if action == "play":
                engine.running = True
            elif action == "pause":
                engine.running = False
            elif action == "step":
                engine.running = False
                engine.step()
            elif action == "speed":
                engine.speed = float(data.get("value", 1.0))
            elif action == "mode":
                val = data.get("value")
                engine.set_override(val if val != "auto" else None)
            elif action == "configure":
                engine.configure(
                    soc=data.get("soc"),
                    soh=data.get("soh"),
                    cycle_count=data.get("cycle_count"),
                    chemistry=data.get("chemistry"),
                    temp=data.get("temp"),
                    current=data.get("current"),
                    r0_mohm=data.get("r0_mohm"),
                    reset_history=data.get("reset_history", True),
                )
            elif action == "reset":
                chem = BatteryChemistry.LFP if data.get("chemistry") == "LFP" else BatteryChemistry.NMC
                engine.chemistry = chem
                engine.reset()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "running": engine.running}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lithium-Ion Battery Digital Twin — Interactive Cockpit</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --input-bg: #1e293b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #38bdf8;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 1.2rem; min-height: 100vh; }
    
    header {
      display: flex; justify-content: space-between; align-items: center;
      padding-bottom: 1rem; border-bottom: 1px solid var(--card-border); margin-bottom: 1.2rem;
    }
    .brand { display: flex; align-items: center; gap: 0.8rem; }
    .pulse-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

    /* Configuration Panel */
    .config-panel {
      background: linear-gradient(180deg, #131c2e 0%, var(--card-bg) 100%);
      border: 1px solid #2563eb44; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .config-header {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;
    }
    .config-title { font-size: 1rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.5rem; }
    .config-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;
    }
    .form-group { display: flex; flex-direction: column; gap: 0.35rem; }
    .form-label { font-size: 0.8rem; font-weight: 600; color: var(--text-muted); display: flex; justify-content: space-between; }
    .form-label span.val-badge { color: var(--accent); font-weight: 700; }
    .form-input {
      background: var(--input-bg); color: #fff; border: 1px solid #334155; padding: 0.5rem 0.75rem; border-radius: 6px;
      font-size: 0.9rem; outline: none; transition: 0.15s;
    }
    .form-input:focus { border-color: var(--accent); }
    input[type="range"] { accent-color: var(--accent); cursor: pointer; }
    .config-actions {
      display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; justify-content: flex-end; padding-top: 0.8rem; border-top: 1px solid #1f2937;
    }
    
    /* Metrics Row */
    .metrics-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.2rem;
    }
    .metric-card {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; padding: 1rem;
      position: relative; overflow: hidden;
    }
    .metric-title { font-size: 0.78rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; margin-bottom: 0.4rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .metric-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem; }
    .badge {
      display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em;
    }
    .badge-NORMAL { background: rgba(16, 185, 129, 0.15); color: var(--green); border: 1px solid var(--green); }
    .badge-WARNING { background: rgba(245, 158, 11, 0.15); color: var(--amber); border: 1px solid var(--amber); }
    .badge-CRITICAL { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); animation: blink 1s infinite; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

    /* Control Bar */
    .control-panel {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; padding: 0.9rem 1.2rem;
      display: flex; flex-wrap: wrap; gap: 0.8rem; align-items: center; justify-content: space-between; margin-bottom: 1.2rem;
    }
    .btn-group { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    button {
      background: #1e293b; color: #fff; border: 1px solid #334155; padding: 0.5rem 0.9rem; border-radius: 6px;
      font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: 0.15s;
    }
    button:hover { background: #334155; }
    button.primary { background: #0284c7; border-color: #0284c7; }
    button.primary:hover { background: #0369a1; }
    button.success { background: #059669; border-color: #10b981; }
    button.success:hover { background: #047857; }
    button.danger { background: #b91c1c; border-color: #ef4444; }
    button.danger:hover { background: #991b1b; }
    select {
      background: var(--input-bg); color: #fff; border: 1px solid #334155; padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 0.85rem;
    }

    /* Charts Grid */
    .charts-grid {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.2rem; margin-bottom: 1.5rem;
    }
    @media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }
    .chart-card {
      background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; padding: 1rem;
      height: 310px; display: flex; flex-direction: column;
    }
    .chart-header { display: flex; justify-content: space-between; margin-bottom: 0.6rem; font-size: 0.85rem; font-weight: 600; color: var(--text); }
    .chart-wrapper { flex: 1; position: relative; width: 100%; height: 100%; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="pulse-dot"></div>
      <div>
        <h1 style="font-size: 1.25rem; font-weight: 800;">Lithium-Ion Battery Digital Twin</h1>
        <p style="font-size: 0.8rem; color: var(--text-muted);">Extended Kalman Filter (EKF) • Recursive Least Squares R0 • TinyML RUL</p>
      </div>
    </div>
  </header>

  <!-- USER INPUT & CONFIGURATION PANEL -->
  <div class="config-panel">
    <div class="config-header">
      <div class="config-title">
        <span>⚙️ Battery State & Parameter Configuration</span>
      </div>
      <span style="font-size: 0.78rem; color: var(--text-muted);">Configure cell health, initial charge, and test the virtual twin in real-time</span>
    </div>

    <div class="config-grid">
      <!-- 1. State of Charge (SOC) -->
      <div class="form-group">
        <label class="form-label">
          <span>State of Charge (SOC)</span>
          <span class="val-badge" id="lbl-input-soc">90%</span>
        </label>
        <input type="range" id="input-soc" min="5" max="100" step="1" value="90" oninput="updateInputLabels()">
        <input type="number" class="form-input" id="num-input-soc" min="1" max="100" value="90" onchange="syncSlider('input-soc', this.value)">
      </div>

      <!-- 2. State of Health (SOH) -->
      <div class="form-group">
        <label class="form-label">
          <span>State of Health (SOH)</span>
          <span class="val-badge" id="lbl-input-soh">92%</span>
        </label>
        <input type="range" id="input-soh" min="65" max="100" step="0.5" value="92" oninput="updateInputLabels()">
        <input type="number" class="form-input" id="num-input-soh" min="60" max="100" step="0.5" value="92" onchange="syncSlider('input-soh', this.value)">
      </div>

      <!-- 3. Battery Chemistry -->
      <div class="form-group">
        <label class="form-label">Cell Chemistry</label>
        <select class="form-input" id="input-chem">
          <option value="NMC" selected>NMC (3.0V - 4.2V, 2.5Ah)</option>
          <option value="LFP">LFP (2.5V - 3.65V, 2.5Ah)</option>
        </select>
      </div>

      <!-- 4. Cycle Count -->
      <div class="form-group">
        <label class="form-label">Cycle Count (Cycles)</label>
        <input type="number" class="form-input" id="input-cycles" min="0" max="2500" value="350">
      </div>

      <!-- 5. Operating Current Load -->
      <div class="form-group">
        <label class="form-label">
          <span>Current Load</span>
          <span class="val-badge" id="lbl-input-curr">Dynamic</span>
        </label>
        <select class="form-input" id="input-current" onchange="updateCurrentLabel()">
          <option value="auto" selected>Auto: Dynamic Profile (DST/WLTP)</option>
          <option value="0.0">0.0 A (Idle Open-Circuit Rest)</option>
          <option value="1.25">1.25 A (0.5C Moderate Discharge)</option>
          <option value="2.5">2.50 A (1.0C Nominal Discharge)</option>
          <option value="5.0">5.00 A (2.0C High Acceleration Pulse)</option>
          <option value="-1.25">-1.25 A (0.5C Regenerative Braking)</option>
          <option value="-2.5">-2.50 A (1.0C Fast Charge Pulse)</option>
        </select>
      </div>

      <!-- 6. Temperature -->
      <div class="form-group">
        <label class="form-label">Temperature (°C)</label>
        <input type="number" class="form-input" id="input-temp" min="-10" max="65" step="0.5" value="25.0">
      </div>

      <!-- 7. Ohmic Resistance R0 -->
      <div class="form-group">
        <label class="form-label">Internal Resistance R0 (mΩ)</label>
        <input type="number" class="form-input" id="input-r0" min="15" max="80" step="0.5" value="31.8">
      </div>
    </div>

    <div class="config-actions">
      <button class="primary" onclick="applyUserConfiguration(true)">
        🚀 Apply Parameters & Restart Simulation
      </button>
      <button class="success" onclick="applyUserConfiguration(false)">
        ⚡ Live Inject State (Keep History)
      </button>
      <button onclick="resetDefaults()">
        ↺ Reset to Defaults
      </button>
    </div>
  </div>

  <!-- Key Metrics Row -->
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-title">State of Charge (SOC)</div>
      <div class="metric-value" id="val-soc">-- %</div>
      <div class="metric-sub" id="sub-soc">True: -- % • EKF Deadbeat</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">State of Health (SOH)</div>
      <div class="metric-value" id="val-soh">-- %</div>
      <div class="metric-sub">Capacity Fade (Cycle <span id="val-cycle">350</span>)</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Remaining Useful Life (RUL)</div>
      <div class="metric-value" id="val-rul">--</div>
      <div class="metric-sub">Cycles remaining to 80% EOL</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Terminal Voltage</div>
      <div class="metric-value" id="val-v">-- V</div>
      <div class="metric-sub">Model: <span id="val-vmodel">--</span> V</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Load Current & Temp</div>
      <div class="metric-value" id="val-it">-- A</div>
      <div class="metric-sub">Core Temp: <span id="val-t">--</span> °C</div>
    </div>
    <div class="metric-card">
      <div class="metric-title">Safety Supervisor</div>
      <div style="margin-top: 0.3rem;"><span id="badge-status" class="badge badge-NORMAL">NORMAL</span></div>
      <div class="metric-sub">ISO 26262 / UL 1973</div>
    </div>
  </div>

  <!-- Interactive Control Panel -->
  <div class="control-panel">
    <div class="btn-group">
      <button id="btn-toggle" class="primary" onclick="togglePlay()">Pause</button>
      <button onclick="stepSim()">Step +1s</button>
      <button onclick="resetSim()">Restart Twin</button>
      <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.5rem;">Speed:</span>
      <input type="range" id="speed-slider" min="0.5" max="10" step="0.5" value="1" onchange="setSpeed(this.value)">
      <span id="speed-label" style="font-size: 0.8rem;">1.0x</span>
    </div>
    <div class="btn-group">
      <span style="font-size: 0.8rem; color: var(--text-muted);">Quick Stress Test:</span>
      <button class="danger" onclick="injectFault('fault_temp')">Thermal Overheat (58°C)</button>
      <button class="danger" onclick="injectFault('fault_v')">Deep Overdischarge (2% SOC)</button>
    </div>
  </div>

  <!-- 4 Real-Time Interactive Charts -->
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-header">
        <span>1. Dynamic Terminal Voltage & Current</span>
        <span style="color: var(--accent);">V(t) [V] & I(t) [A]</span>
      </div>
      <div class="chart-wrapper"><canvas id="chart-voltage"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <span>2. State of Charge: EKF Estimate vs True Ground Truth</span>
        <span style="color: var(--green);">SOC (%)</span>
      </div>
      <div class="chart-wrapper"><canvas id="chart-soc"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <span>3. SOH Capacity Fade & R0 Ohmic Resistance Tracking</span>
        <span style="color: var(--amber);">SOH (%) & R0 (mΩ)</span>
      </div>
      <div class="chart-wrapper"><canvas id="chart-soh"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-header">
        <span>4. Remaining Useful Life (RUL) Trajectory</span>
        <span style="color: var(--purple);">Cycles to 80% EOL</span>
      </div>
      <div class="chart-wrapper"><canvas id="chart-rul"></canvas></div>
    </div>
  </div>

  <script>
    let isPlaying = true;
    const maxPoints = 50;

    function updateInputLabels() {
      const soc = document.getElementById('input-soc').value;
      const soh = document.getElementById('input-soh').value;
      document.getElementById('lbl-input-soc').innerText = soc + '%';
      document.getElementById('lbl-input-soh').innerText = soh + '%';
      document.getElementById('num-input-soc').value = soc;
      document.getElementById('num-input-soh').value = soh;
    }

    function syncSlider(sliderId, val) {
      document.getElementById(sliderId).value = val;
      updateInputLabels();
    }

    function updateCurrentLabel() {
      const sel = document.getElementById('input-current');
      const text = sel.options[sel.selectedIndex].text;
      document.getElementById('lbl-input-curr').innerText = text.split(' ')[0];
    }

    function resetDefaults() {
      document.getElementById('input-soc').value = 90;
      document.getElementById('input-soh').value = 92;
      document.getElementById('input-chem').value = 'NMC';
      document.getElementById('input-cycles').value = 350;
      document.getElementById('input-current').value = 'auto';
      document.getElementById('input-temp').value = 25.0;
      document.getElementById('input-r0').value = 31.8;
      updateInputLabels();
      updateCurrentLabel();
      applyUserConfiguration(true);
    }

    async function applyUserConfiguration(resetHistory) {
      const payload = {
        action: 'configure',
        soc: parseFloat(document.getElementById('input-soc').value),
        soh: parseFloat(document.getElementById('input-soh').value),
        chemistry: document.getElementById('input-chem').value,
        cycle_count: parseInt(document.getElementById('input-cycles').value),
        current: document.getElementById('input-current').value,
        temp: parseFloat(document.getElementById('input-temp').value),
        r0_mohm: parseFloat(document.getElementById('input-r0').value),
        reset_history: resetHistory
      };

      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (resetHistory) {
        [chartV, chartSOC, chartSOH, chartRUL].forEach(c => {
          c.data.labels = [];
          c.data.datasets.forEach(ds => ds.data = []);
          c.update();
        });
      }
      fetchTelemetry();
    }

    // Common Chart Config
    const chartDefaults = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#64748b' } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#64748b' } }
      },
      plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 12 } } }
    };

    // Chart 1: Voltage
    const ctxV = document.getElementById('chart-voltage').getContext('2d');
    const chartV = new Chart(ctxV, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Measured Voltage (V)', data: [], borderColor: '#94a3b8', borderWidth: 1, pointRadius: 0 },
          { label: 'Model Estimated V_model (V)', data: [], borderColor: '#ef4444', borderDash: [4, 4], borderWidth: 1.8, pointRadius: 0 },
          { label: 'Load Current (A)', data: [], borderColor: '#38bdf8', borderWidth: 1.2, pointRadius: 0, yAxisID: 'y1' }
        ]
      },
      options: {
        ...chartDefaults,
        scales: {
          x: chartDefaults.scales.x,
          y: { ...chartDefaults.scales.y, title: { display: true, text: 'Voltage (V)', color: '#94a3b8' } },
          y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#38bdf8' }, title: { display: true, text: 'Current (A)', color: '#38bdf8' } }
        }
      }
    });

    // Chart 2: SOC
    const ctxSOC = document.getElementById('chart-soc').getContext('2d');
    const chartSOC = new Chart(ctxSOC, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'True SOC (%)', data: [], borderColor: '#10b981', borderWidth: 2, pointRadius: 0 },
          { label: 'EKF Estimated SOC (%)', data: [], borderColor: '#a855f7', borderDash: [4, 4], borderWidth: 2, pointRadius: 0 }
        ]
      },
      options: chartDefaults
    });

    // Chart 3: SOH & R0
    const ctxSOH = document.getElementById('chart-soh').getContext('2d');
    const chartSOH = new Chart(ctxSOH, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'SOH Capacity (%)', data: [], borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0 },
          { label: 'Tracked R0 (mΩ)', data: [], borderColor: '#f59e0b', borderWidth: 1.8, pointRadius: 0, yAxisID: 'y1' }
        ]
      },
      options: {
        ...chartDefaults,
        scales: {
          x: chartDefaults.scales.x,
          y: { ...chartDefaults.scales.y, title: { display: true, text: 'SOH (%)', color: '#3b82f6' } },
          y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#f59e0b' }, title: { display: true, text: 'R0 (mΩ)', color: '#f59e0b' } }
        }
      }
    });

    // Chart 4: RUL
    const ctxRUL = document.getElementById('chart-rul').getContext('2d');
    const chartRUL = new Chart(ctxRUL, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'RUL Predicted (Cycles)', data: [], borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(6, 182, 212, 0.08)' }
        ]
      },
      options: chartDefaults
    });

    // Polling Telemetry Stream
    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const d = await res.json();
        if (!d.timestamp && d.timestamp !== 0) return;

        // Update Metric Cards
        document.getElementById('val-soc').innerText = d.soc.toFixed(1) + ' %';
        document.getElementById('sub-soc').innerText = `True: ${d.true_soc.toFixed(1)} % • Error: ${(d.soc - d.true_soc).toFixed(2)}%`;
        document.getElementById('val-soh').innerText = d.soh.toFixed(1) + ' %';
        document.getElementById('val-cycle').innerText = d.cycle_count || 350;
        document.getElementById('val-rul').innerText = d.rul_cycles + ' cyc';
        document.getElementById('val-v').innerText = d.voltage.toFixed(3) + ' V';
        document.getElementById('val-vmodel').innerText = d.v_model.toFixed(3);
        document.getElementById('val-it').innerText = d.current.toFixed(2) + ' A';
        document.getElementById('val-t').innerText = d.temp.toFixed(1);

        const badge = document.getElementById('badge-status');
        badge.innerText = d.status;
        badge.className = 'badge badge-' + d.status;

        // Push to charts
        const timeLabel = d.timestamp.toFixed(0) + 's';
        
        function pushData(chart, values) {
          if (chart.data.labels.length > maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(ds => ds.data.shift());
          }
          chart.data.labels.push(timeLabel);
          values.forEach((val, idx) => chart.data.datasets[idx].data.push(val));
          chart.update();
        }

        pushData(chartV, [d.voltage, d.v_model, d.current]);
        pushData(chartSOC, [d.true_soc, d.soc]);
        pushData(chartSOH, [d.soh, d.r0_est_mohm]);
        pushData(chartRUL, [d.rul_cycles]);

      } catch (err) {
        console.warn('Telemetry fetch error:', err);
      }
    }

    setInterval(fetchTelemetry, 350);

    // Control APIs
    async function togglePlay() {
      isPlaying = !isPlaying;
      document.getElementById('btn-toggle').innerText = isPlaying ? 'Pause' : 'Resume';
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: isPlaying ? 'play' : 'pause' })
      });
    }

    async function stepSim() {
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'step' })
      });
      fetchTelemetry();
    }

    async function resetSim() {
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'reset' })
      });
      [chartV, chartSOC, chartSOH, chartRUL].forEach(c => {
        c.data.labels = [];
        c.data.datasets.forEach(ds => ds.data = []);
        c.update();
      });
    }

    async function setSpeed(v) {
      document.getElementById('speed-label').innerText = parseFloat(v).toFixed(1) + 'x';
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'speed', value: v })
      });
    }

    async function injectFault(faultType) {
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'mode', value: faultType })
      });
      setTimeout(async () => {
        await fetch('/api/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'mode', value: 'auto' })
        });
      }, 4000);
    }
  </script>
</body>
</html>
"""


def start_server(port=8000):
    server = ThreadedHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"================================================================================")
    print(f"  BATTERY DIGITAL TWIN REAL-TIME WEB DASHBOARD")
    print(f"  Live Local URL:    http://localhost:{port}")
    print(f"================================================================================")

    # Start simulation engine background ticker
    t = threading.Thread(target=simulation_ticker, daemon=True)
    t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
