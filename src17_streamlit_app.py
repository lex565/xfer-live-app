"""
CLIMATE-XFER  ·  Interactive Drought Intelligence Dashboard
Transferable Deep Learning for Seasonal Drought Forecasting
Across Southern Africa and Southeast Asia

Beihang University  ·  MSc Artificial Intelligence & Large Models  ·  2025
Authors: Tanaka Alex Mbendana · Fitrotur Rofiqoh · Munashe Innocent Mafuta

Run: streamlit run src17_streamlit_app.py
"""

import base64, random, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# BEIHANG LOGO  (embedded base64 so it works offline / on any host)
# ══════════════════════════════════════════════════════════════════════════════
_LOGO_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsK"
    "CwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQU"
    "FBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCABvAGoDASIAAhEB"
    "AxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9"
    "AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
    "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEB"
    "AQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
    "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6"
    "wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9U6KKKACi"
    "kJxXDeP/AIw6D4AuYNOlafVvEF0M2mhaVF9ovbj3CD7q+rsQo9auMJTfLFXZE5xguaTsjuc02SVI"
    "Yy8jqiDqzHAH4148unfFj4iES3uo2fw10l+llp6rfakV/wBuZh5cZ9kVsetedfF3w58O/hYtv/wk"
    "Cav8SPEs+JRpWp6zK88sO7DzLH93C9SMCumnh1OShzXfZa/jovxOSpiXCPPy2Xd6fhq/wPpGXxjo"
    "ED7ZNc02Nv7rXcYP86u2Or2OpjNne292P+mEqv8AyNeOfDf4dfCb4h6CdT0/4d2tla7tinU9L8oy"
    "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"
)
_LOGO_URI = f"data:image/jpeg;base64,{_LOGO_B64}"

# ══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
ROOT    = Path(__file__).parent
MODELS  = ROOT / "models"
REPORTS = ROOT / "reports"
ZARR    = str(ROOT / "data_processed" / "aligned_1deg.zarr")
DATES     = pd.date_range("2000-01-01", periods=276, freq="MS")
N_HIST    = 276        # Jan 2000 – Dec 2022 (training / validation)
N_FUTURE  = 40         # Jan 2023 – Apr 2026 (simulated)
N_TOTAL   = N_HIST + N_FUTURE
DATES_EXT = pd.date_range("2000-01-01", periods=N_TOTAL, freq="MS")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CLIMATE-XFER · Drought Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GRU MODEL  (self-contained — no import from src08)
# ══════════════════════════════════════════════════════════════════════════════
class GRURegressor(nn.Module):
    def __init__(self, n_features: int = 4, hidden: int = 32):
        super().__init__()
        self.gru  = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY DARK DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(5,14,31,0.55)",
    font=dict(color="#e0f4ff", family="Inter, system-ui, sans-serif"),
    margin=dict(l=12, r=12, t=46, b=12),
    legend=dict(bgcolor="rgba(5,14,31,0.88)", bordercolor="rgba(0,212,255,0.22)", borderwidth=1),
)
AX = dict(gridcolor="rgba(0,212,255,0.09)", linecolor="rgba(0,212,255,0.22)", tickcolor="rgba(0,212,255,0.3)")

# ══════════════════════════════════════════════════════════════════════════════
# RAIN DROPS  (Python-generated CSS animation — no JS required)
# ══════════════════════════════════════════════════════════════════════════════
def _rain_html() -> str:
    rng = random.Random(42)
    drops = ""
    for _ in range(70):
        left     = rng.uniform(0, 100)
        delay    = rng.uniform(0, 7)
        duration = rng.uniform(0.55, 2.4)
        height   = rng.randint(12, 48)
        opacity  = rng.uniform(0.15, 0.62)
        drops += (
            f'<div style="position:fixed;left:{left:.1f}%;top:-{height}px;'
            f'width:1.5px;height:{height}px;pointer-events:none;z-index:0;'
            f'background:linear-gradient(to bottom,transparent,rgba(0,212,255,{opacity:.2f}));'
            f'animation:rfall {duration:.2f}s {delay:.2f}s linear infinite;'
            f'border-radius:0 0 2px 2px;"></div>'
        )
    return drops

# ══════════════════════════════════════════════════════════════════════════════
# OCEAN WAVE  (pure SVG animate)
# ══════════════════════════════════════════════════════════════════════════════
def _wave_html() -> str:
    return """
<div style="width:100%;overflow:hidden;line-height:0;margin:2px 0">
<svg viewBox="0 0 1440 90" xmlns="http://www.w3.org/2000/svg" style="display:block">
  <defs>
    <linearGradient id="wg1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%"   style="stop-color:#00d4ff;stop-opacity:.45"/>
      <stop offset="100%" style="stop-color:#00ffc8;stop-opacity:.07"/>
    </linearGradient>
  </defs>
  <path fill="url(#wg1)">
    <animate attributeName="d" dur="6s" repeatCount="indefinite" values="
      M0,44 C240,74 480,14 720,44 C960,74 1200,14 1440,44 L1440,90 L0,90 Z;
      M0,44 C240,14 480,74 720,44 C960,14 1200,74 1440,44 L1440,90 L0,90 Z;
      M0,44 C240,74 480,14 720,44 C960,74 1200,14 1440,44 L1440,90 L0,90 Z"/>
  </path>
  <path fill="rgba(0,212,255,.11)">
    <animate attributeName="d" dur="9s" repeatCount="indefinite" values="
      M0,56 C360,26 720,86 1080,56 C1260,41 1380,61 1440,56 L1440,90 L0,90 Z;
      M0,56 C360,86 720,26 1080,56 C1260,71 1380,49 1440,56 L1440,90 L0,90 Z;
      M0,56 C360,26 720,86 1080,56 C1260,41 1380,61 1440,56 L1440,90 L0,90 Z"/>
  </path>
</svg>
</div>"""

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    drops = _rain_html()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,900&family=Space+Grotesk:wght@300;400;500;600&display=swap');
:root {{
  --bg0:#050e1f; --bg1:#0a1f3d;
  --cyan:#00d4ff; --teal:#00ffc8; --gold:#ffd700; --red:#ff4757;
  --txt:#e0f4ff;  --mute:#7eb8d4;
  --border:rgba(0,212,255,.18); --glow:0 0 22px rgba(0,212,255,.28);
  --font-head:'Fraunces',Georgia,serif;
  --font-body:'Space Grotesk',system-ui,sans-serif;
}}

/* ── Photographic dual-continent background: Africa (left) · SEA (right) ── */
body::before {{
  content:'';
  position:fixed; top:0; left:0;
  width:50vw; height:100vh; z-index:-999;
  background:url('https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?w=1280&q=80&fit=crop') center/cover no-repeat;
}}
body::after {{
  content:'';
  position:fixed; top:0; right:0;
  width:50vw; height:100vh; z-index:-999;
  background:url('https://images.unsplash.com/photo-1537953773345-d172ccf13cf4?w=1280&q=80&fit=crop') center/cover no-repeat;
}}

/* ── Dark cinematic overlay ── */
.stApp,
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(
    90deg,
    rgba(1,10,20,0.90) 0%,
    rgba(2,13,26,0.76) 46%,
    rgba(2,13,26,0.76) 54%,
    rgba(1,10,20,0.90) 100%
  ) !important;
  color: var(--txt) !important;
  font-family: var(--font-body) !important;
}}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg,#050e1f,#071a35) !important;
  border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--txt) !important; }}
[data-testid="stHeader"]  {{ background: transparent !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Rain keyframe ── */
@keyframes rfall {{
  0%   {{ transform:translateY(-60px); opacity:0; }}
  10%  {{ opacity:1; }}
  88%  {{ opacity:.55; }}
  100% {{ transform:translateY(112vh); opacity:0; }}
}}

/* ── Sidebar nav ── */
.stRadio > div {{ gap:6px !important; }}
.stRadio label {{
  background: rgba(0,212,255,.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  cursor: pointer !important;
  transition: all .2s !important;
  color: var(--txt) !important;
  font-size: .88rem !important;
}}
.stRadio label:hover {{
  background: rgba(0,212,255,.14) !important;
  border-color: var(--cyan) !important;
  box-shadow: var(--glow) !important;
}}

/* ── Glassmorphism panel ── */
.glass {{
  background: rgba(10,31,61,.72);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 26px 30px;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  margin-bottom: 18px;
  box-shadow: var(--glow);
}}

/* ── Metric card ── */
.mcard {{
  background: rgba(0,212,255,.06);
  border: 1px solid rgba(0,212,255,.2);
  border-radius: 14px;
  padding: 20px 14px;
  text-align: center;
  transition: transform .22s, box-shadow .22s;
  height: 100%;
}}
.mcard:hover {{ transform: translateY(-5px); box-shadow: 0 10px 36px rgba(0,212,255,.25); }}
.mval {{
  font-size: 1.9rem; font-weight: 800;
  background: linear-gradient(135deg, var(--cyan), var(--teal));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  line-height: 1;
}}
.mlbl {{ font-size:.72rem; color:var(--mute); margin-top:5px; text-transform:uppercase; letter-spacing:1.2px; }}
.msub {{ font-size:.65rem; color:rgba(126,184,212,.7); margin-top:3px; }}

/* ── Author cards ── */
.agrid {{
  display: flex;
  gap: 22px;
  justify-content: center;
  flex-wrap: wrap;
  margin: 34px 0 26px;
}}
.acard {{
  background: rgba(10,31,61,.93);
  border: 1px solid rgba(0,212,255,.26);
  border-radius: 20px;
  padding: 30px 22px;
  text-align: center;
  width: 248px;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: transform .3s, box-shadow .3s;
  position: relative;
  overflow: hidden;
}}
.acard::before {{
  content: '';
  position: absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, var(--cyan), var(--teal));
}}
.acard:hover {{ transform: translateY(-8px); box-shadow: 0 18px 50px rgba(0,212,255,.28); }}
.avatar {{
  width:76px; height:76px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:1.6rem; font-weight:700;
  margin:0 auto 14px;
  border:2px solid var(--cyan);
  box-shadow: 0 0 18px rgba(0,212,255,.38);
  color:#fff;
}}
.aname {{ font-size:.97rem; font-weight:700; color:var(--txt); margin-bottom:3px; }}
.aid   {{ font-size:.71rem; color:var(--cyan); font-family:monospace; letter-spacing:1px; margin-bottom:10px; }}
.auni  {{ font-size:.67rem; color:var(--mute); margin-bottom:11px; line-height:1.5; }}
.arole {{
  display:block;
  background: rgba(0,212,255,.09); border:1px solid rgba(0,212,255,.25);
  border-radius: 20px; padding:5px 10px;
  font-size:.67rem; color:var(--teal); line-height:1.55;
  width:100%; box-sizing:border-box;
}}

/* ── Hero title ── */
.hero-title {{
  font-size: 3.8rem; font-weight:900;
  font-family: var(--font-head) !important;
  background: linear-gradient(130deg, #00d4ff 0%, #00ffc8 55%, #ffffff 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  text-align:center; letter-spacing:-1px; line-height:1.1;
  animation: hpulse 3.5s ease-in-out infinite;
}}
@keyframes hpulse {{
  0%,100% {{ filter:brightness(1); }}
  50%      {{ filter:brightness(1.2); }}
}}
.hero-sub  {{ text-align:center; color:var(--mute); font-size:.93rem; margin-top:6px; letter-spacing:2.5px; text-transform:uppercase; }}
.hero-inst {{ text-align:center; color:rgba(0,212,255,.58); font-size:.78rem; margin-top:3px; letter-spacing:1px; }}

/* ── Section header ── */
.sh {{
  font-size:1.48rem; font-weight:700;
  background: linear-gradient(90deg, var(--cyan), var(--teal));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  border-bottom:1px solid var(--border); padding-bottom:10px; margin-bottom:18px;
}}

/* ── Severity badge ── */
.sbadge {{
  display:block; padding:12px 30px; border-radius:40px;
  font-size:1.25rem; font-weight:700; letter-spacing:1px; text-transform:uppercase;
  margin:12px 0; text-align:center; box-sizing:border-box;
}}

/* ── Streamlit tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  background: rgba(0,212,255,.04) !important;
  border-radius:10px !important;
  gap:4px !important;
}}
.stTabs [data-baseweb="tab"] {{ color:var(--mute) !important; border-radius:8px !important; }}
.stTabs [aria-selected="true"] {{
  background: rgba(0,212,255,.16) !important;
  color:var(--cyan) !important;
}}

/* ── Selectbox / slider ── */
.stSelectbox > div > div {{
  background: rgba(0,212,255,.05) !important;
  border-color: var(--border) !important;
  color: var(--txt) !important;
}}
[data-testid="stSlider"] {{ color: var(--txt) !important; }}

/* ── Divider ── */
hr {{ border-color: rgba(0,212,255,.15) !important; }}

/* ── Caption text ── */
.stCaption {{ color: var(--mute) !important; font-size:.8rem !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-track {{ background: var(--bg0); }}
::-webkit-scrollbar-thumb {{ background: var(--cyan); border-radius:3px; }}
</style>
{drops}
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_zarr_data() -> dict:
    """Load all spatial grids from Zarr into numpy arrays (cached)."""
    def _load(group: str, var: str):
        ds = xr.open_zarr(ZARR, group=group)
        return (
            ds[var].values.astype(np.float32),
            ds["lat"].values.astype(np.float32),
            ds["lon"].values.astype(np.float32),
        )

    sp_s, lt_s, ln_s = _load("spei_sadc",   "spei")
    sp_e, lt_e, ln_e = _load("spei_sea",    "spei")
    ch_s, _,    _    = _load("chirps_sadc", "precip")
    ch_e, _,    _    = _load("chirps_sea",  "precip")
    pc,   lt_p, ln_p = _load("sst_pac",    "sst_anom")
    ind,  lt_i, ln_i = _load("sst_ind",    "sst_anom")

    return dict(
        spei_sadc=sp_s, lat_sadc=lt_s, lon_sadc=ln_s,
        spei_sea=sp_e,  lat_sea=lt_e,  lon_sea=ln_e,
        chirps_sadc=ch_s, chirps_sea=ch_e,
        sst_pac=pc,  lat_pac=lt_p, lon_pac=ln_p,
        sst_ind=ind, lat_ind=lt_i, lon_ind=ln_i,
    )


@st.cache_resource
def load_series() -> dict:
    """Compute area-mean time series (276, 4) for SADC and SEA."""
    z = load_zarr_data()
    def mn(arr):
        return np.nan_to_num(np.nanmean(arr, axis=(1, 2)), nan=0.0).astype(np.float32)

    pac = mn(z["sst_pac"])
    ind = mn(z["sst_ind"])
    return {
        "sadc": np.stack([mn(z["spei_sadc"]), mn(z["chirps_sadc"]), pac, ind], axis=1),
        "sea":  np.stack([mn(z["spei_sea"]),  mn(z["chirps_sea"]),  pac, ind], axis=1),
    }


@st.cache_resource
def load_models() -> dict:
    dev = torch.device("cpu")

    def _load(path: Path):
        m = GRURegressor(n_features=4, hidden=32).to(dev)
        m.load_state_dict(torch.load(str(path), map_location=dev))
        m.eval()
        return m

    sadc = _load(MODELS / "sadc_mean_gru.pt")
    sea  = _load(MODELS / "sea_mean_gru_finetuned.pt")
    return {"sadc": sadc, "zero_shot": sadc, "fine_tuned": sea}


@st.cache_data
def load_csv_timeseries() -> dict:
    return {
        "sadc": pd.read_csv(REPORTS / "sadc_mean_val_timeseries.csv", parse_dates=["time"]),
        "sea_zs": pd.read_csv(REPORTS / "sea_zeroshot_timeseries.csv",  parse_dates=["time"]),
        "sea_ft": pd.read_csv(REPORTS / "sea_finetune_timeseries.csv",  parse_dates=["time"]),
    }


@st.cache_resource
def load_extended_series() -> dict:
    """Historical 276 months + GRU-rolled 40-month simulation → (316, 4) each."""
    hist = load_series()
    mdls = load_models()

    def _simulate(series: np.ndarray, model: nn.Module) -> np.ndarray:
        full = series.copy()
        rng  = np.random.default_rng(42)
        for i in range(N_FUTURE):
            cal   = DATES_EXT[N_HIST + i].month
            same  = [j for j in range(N_HIST) if DATES_EXT[j].month == cal]
            clim  = series[same].mean(axis=0)
            win   = full[len(full) - 12:]
            x     = torch.tensor(win[np.newaxis]).float()
            with torch.no_grad():
                spei_pred = float(model(x).item())
            noise = rng.normal(0, 0.04 * (i + 1) / N_FUTURE, size=4).astype(np.float32)
            row   = np.array([spei_pred, clim[1], clim[2], clim[3]], dtype=np.float32) + noise
            full  = np.vstack([full, row])
        return full  # (316, 4)

    return {
        "sadc": _simulate(hist["sadc"], mdls["sadc"]),
        "sea":  _simulate(hist["sea"],  mdls["fine_tuned"]),
    }


def build_extended_spatial(data3d: np.ndarray) -> np.ndarray:
    """Pad (276, lat, lon) → (316, lat, lon) using climatological monthly means."""
    clim = np.stack([
        data3d[[j for j in range(N_HIST) if DATES_EXT[j].month == m]].mean(axis=0)
        for m in range(1, 13)
    ])  # (12, lat, lon)
    future = np.stack([clim[DATES_EXT[N_HIST + i].month - 1] for i in range(N_FUTURE)])
    return np.concatenate([data3d, future], axis=0)


@st.cache_data
def load_training_history() -> dict:
    with open(REPORTS / "train_sadc_mean_history.json") as f:
        sadc = json.load(f)
    with open(REPORTS / "sea_finetune_history.json") as f:
        sea = json.load(f)
    return {"sadc": sadc, "sea": sea}

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(series: np.ndarray, model: nn.Module, t_target: int, history: int = 12) -> float | None:
    t = min(t_target - 1, len(series) - 1)
    if t < history - 1:
        return None
    start  = max(0, t - history + 1)
    window = series[start : t + 1, :]
    if len(window) < history:
        window = np.pad(window, ((history - len(window), 0), (0, 0)), mode="edge")
    x = torch.tensor(window).unsqueeze(0).float()
    with torch.no_grad():
        return model(x).item()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def forecast_confidence(t_idx: int):
    """Return (confidence 0-1, sigma_multiplier) for a month index."""
    if t_idx < N_HIST:
        return 0.85, 1.0
    months_ahead = t_idx - N_HIST + 1
    conf = max(0.42, 0.85 - months_ahead * 0.011)
    sigma_mult = 1.0 + months_ahead * 0.07
    return round(conf, 3), round(sigma_mult, 3)


def spei_severity(val: float):
    if val >= 0:
        return "Normal / Wet", "#00ffc8", "rgba(0,255,200,.14)"
    elif val >= -1.0:
        return "Mild Dry", "#ffd700", "rgba(255,215,0,.14)"
    elif val >= -1.5:
        return "Moderate Drought", "#ff9500", "rgba(255,149,0,.16)"
    elif val >= -2.0:
        return "Severe Drought", "#ff4757", "rgba(255,71,87,.18)"
    else:
        return "Extreme Drought", "#cc0000", "rgba(180,0,0,.24)"


def metric_card(val: str, label: str, sub: str = "") -> str:
    return (f'<div class="mcard">'
            f'<div class="mval">{val}</div>'
            f'<div class="mlbl">{label}</div>'
            + (f'<div class="msub">{sub}</div>' if sub else "")
            + '</div>')


def drought_band_traces(fig, df: pd.DataFrame, col: str = "true"):
    times = df["time"].tolist()
    vals  = df[col].tolist()
    for i in range(len(vals) - 1):
        clr = "rgba(255,71,87,.11)" if vals[i] < 0 else "rgba(0,255,200,.07)"
        fig.add_vrect(x0=times[i], x1=times[i + 1], fillcolor=clr, line_width=0, layer="below")
    return fig


def hline_drought(fig):
    specs = [
        (0,    "rgba(255,255,255,.18)", "dash",  ""),
        (-1.0, "rgba(255,215,0,.45)",   "dot",   "Mild"),
        (-1.5, "rgba(255,149,0,.45)",   "dot",   "Moderate"),
        (-2.0, "rgba(255,71,87,.5)",    "dot",   "Severe"),
    ]
    for y, col, dash, ann in specs:
        kw = dict(annotation_text=ann, annotation_font_color=col) if ann else {}
        fig.add_hline(y=y, line_color=col, line_dash=dash, line_width=1, **kw)
    return fig


def animated_heatmap(data3d, lats, lons, t_idx, d_labels, cscale, zmin, zmax, title) -> go.Figure:
    """Build Plotly animated heatmap with Play/Pause + time slider."""
    frames = [
        go.Frame(
            data=[go.Heatmap(z=data3d[ti], x=lons, y=lats,
                             colorscale=cscale, zmin=zmin, zmax=zmax, showscale=False)],
            name=dl,
            layout=go.Layout(title_text=f"{title}  —  {dl}"),
        )
        for ti, dl in zip(t_idx, d_labels)
    ]

    fig = go.Figure(
        data=[go.Heatmap(
            z=data3d[t_idx[0]], x=lons, y=lats,
            colorscale=cscale, zmin=zmin, zmax=zmax,
            colorbar=dict(
                thickness=14, tickfont=dict(color="#e0f4ff", size=10),
                bgcolor="rgba(5,14,31,.72)",
                outlinecolor="rgba(0,212,255,.3)", outlinewidth=1,
            ),
        )],
        frames=frames,
    )

    slider_steps = [
        dict(method="animate",
             args=[[dl], dict(mode="immediate",
                              frame=dict(duration=1500, redraw=True),
                              transition=dict(duration=0))],
             label=dl)
        for dl in d_labels
    ]

    fig.update_layout(
        **PL,
        height=430,
        title=dict(text=f"{title}  —  {d_labels[0]}", font=dict(size=14)),
        xaxis=dict(**AX, title="Longitude"),
        yaxis=dict(**AX, title="Latitude"),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.14, x=0.0, xanchor="left",
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=1500, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
            font=dict(color="#050e1f", size=12),
            bgcolor="#00d4ff", bordercolor="rgba(0,212,255,.5)",
        )],
        sliders=[dict(
            active=0, steps=slider_steps,
            x=0, y=-0.02, len=1.0,
            currentvalue=dict(font=dict(size=11, color="#00d4ff"),
                               prefix="Month: ", visible=True, xanchor="center"),
            pad=dict(b=10, t=40),
            tickcolor="rgba(0,212,255,.35)",
            font=dict(color="#7eb8d4", size=8),
            bgcolor="rgba(0,212,255,.07)",
            bordercolor="rgba(0,212,255,.22)",
            activebgcolor="rgba(0,212,255,.32)",
        )],
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HERO / HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_hero():
    # ── Beihang logo + title banner ───────────────────────────────────────────
    st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:center;gap:28px;
            padding:20px 0 6px;flex-wrap:wrap;">
  <img src="{_LOGO_URI}"
       style="height:88px;border-radius:50%;
              border:2px solid rgba(0,212,255,0.45);
              box-shadow:0 0 22px rgba(0,212,255,0.35);">
  <div style="text-align:left;">
    <div class="hero-title" style="text-align:left;font-size:3.2rem;">CLIMATE-XFER</div>
    <div class="hero-sub" style="text-align:left;">
      Transferable Deep Learning &nbsp;·&nbsp; Seasonal Drought Forecasting
    </div>
    <div class="hero-inst" style="text-align:left;">
      北京航空航天大学 &nbsp;·&nbsp; Beihang University
      &nbsp;·&nbsp; MSc AI &amp; Large Models &nbsp;·&nbsp; 2025
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(_wave_html(), unsafe_allow_html=True)

    # ── Embedded live geo scene (as visual header) ────────────────────────────
    st.markdown(
        '<div class="sh" style="margin-top:10px">🌍  Live Geographical Scene</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#7eb8d4;font-size:.82rem;margin-bottom:8px">'
        'Globe rotates automatically · Rain density = CHIRPS precipitation · '
        'Press <strong style="color:#00d4ff">▶ Play</strong> to animate 2000–2022'
        '</div>',
        unsafe_allow_html=True,
    )
    html_str = build_geo_html(MAPBOX_TOKEN)
    components.html(html_str, height=480, scrolling=False)

    st.markdown(_wave_html(), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Key results ──────────────────────────────────────────────────────────
    st.markdown('<div class="sh">Key Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("0.835", "SADC Pearson r",     "Mean-GRU Training"),
        ("0.882", "SEA Zero-Shot r",    "Direct Transfer"),
        ("0.903", "SEA Fine-Tuned r",   "10-Epoch Adaptation"),
        ("0.193", "SADC RMSE",          "vs Persist. 0.202"),
        ("11,777", "Parameters",        "Lightweight GRU"),
    ]
    for col, (v, lbl, sub) in zip([c1, c2, c3, c4, c5], cards):
        col.markdown(metric_card(v, lbl, sub), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Research Team ────────────────────────────────────────────────────────
    st.markdown('<div class="sh">Research Team</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="agrid">

  <div class="acard">
    <div class="avatar"
         style="background:linear-gradient(135deg,#004fa3,#00d4ff)">TM</div>
    <div class="aname">Tanaka Alex Mbendana</div>
    <div class="aid">LS2525233 &nbsp;·&nbsp; Corresponding Author</div>
    <div class="auni">MSc Space Technology (Remote Sensing)<br>Beihang University, Beijing, China</div>
    <div class="arole">
      Core GRU Architecture &amp; Training Pipeline<br>
      SADC-to-SEA Transfer Learning Framework<br>
      Primary Manuscript &amp; Experimental Design
    </div>
  </div>

  <div class="acard">
    <div class="avatar"
         style="background:linear-gradient(135deg,#005c3b,#00ffc8)">FR</div>
    <div class="aname">Fitrotur Rofiqoh</div>
    <div class="aid">LS2525220</div>
    <div class="auni">MSc Space Technology (Remote Sensing)<br>Beihang University, Beijing, China</div>
    <div class="arole">
      Data Acquisition &amp; Preprocessing Pipeline<br>
      CHIRPS &amp; ERA5 Harmonisation<br>
      Zarr Storage Architecture &amp; SPEI Spatial Analysis
    </div>
  </div>

  <div class="acard">
    <div class="avatar"
         style="background:linear-gradient(135deg,#3d0075,#a855f7)">MM</div>
    <div class="aname">Munashe Innocent Mafuta</div>
    <div class="aid">LS2557204</div>
    <div class="auni">MSc Mechanical Engineering<br>Beihang University, Beijing, China</div>
    <div class="arole">
      GRU Gate Equations &amp; Fine-Tune Loss Function<br>
      CNN-GRU Ablation Study<br>
      Metrics Computation &amp; Comparative Analysis
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    # ── Abstract ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sh">Abstract</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="glass" style="font-size:.9rem;line-height:1.85;color:#b8d8f0">
Drought forecasting is a critical challenge for climate adaptation in the Global South. This study presents
<strong style="color:#00d4ff">CLIMATE-XFER</strong>, a transferable deep learning framework that trains a
Gated Recurrent Unit (GRU) model on area-averaged monthly climate indicators over the Southern African
Development Community (SADC) region and evaluates its generalizability to Southeast Asia (SEA) via
zero-shot inference and lightweight fine-tuning. Four input features — the Standardised
Precipitation-Evapotranspiration Index (SPEI-1), CHIRPS precipitation anomalies, and Pacific and Indian Ocean
sea-surface temperature (SST) indices — are fed as 12-month history windows to forecast the subsequent
month's mean SPEI. On the SADC validation set (<em>n</em> = 52 months, 2018–2022), the Mean-GRU achieves
RMSE = 0.193 and Pearson <em>r</em> = 0.835, surpassing the persistence baseline (RMSE = 0.202).
Zero-shot transfer to SEA yields <em>r</em> = 0.882, while ten epochs of fine-tuning improve <em>r</em>
to <strong style="color:#00ffc8">0.903</strong>, demonstrating that temporal teleconnection patterns
learned in one region transfer meaningfully across continents.
</div>
""", unsafe_allow_html=True)

    # ── Pipeline diagram (text-based) ────────────────────────────────────────
    st.markdown('<div class="sh">System Pipeline</div>', unsafe_allow_html=True)
    fig_pipe = go.Figure()
    stages = ["Data Inputs\n(SPEI·CHIRPS·SST)", "Preprocess\n(1° Zarr)", "SADC Mean-GRU\n(40 epochs)", "Zero-Shot\nSADC→SEA", "Fine-Tune SEA\n(10 epochs)", "Evaluation\nRMSE·MAE·r"]
    colors = ["#0050aa", "#006c55", "#00d4ff", "#a855f7", "#00ffc8", "#ffd700"]
    for i, (s, c) in enumerate(zip(stages, colors)):
        fig_pipe.add_shape(type="rect", x0=i*1.7, x1=i*1.7+1.5, y0=0.2, y1=0.8,
                           fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18)",
                           line=dict(color=c, width=1.5))
        fig_pipe.add_annotation(x=i*1.7+0.75, y=0.5, text=s.replace("\n","<br>"),
                                 font=dict(size=10, color=c), showarrow=False, align="center")
        if i < len(stages) - 1:
            fig_pipe.add_annotation(x=i*1.7+1.6, y=0.5, text="→",
                                     font=dict(size=18, color="#7eb8d4"), showarrow=False)
    fig_pipe.update_layout(**PL, height=140,
                            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 10.4]),
                            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1]))
    st.plotly_chart(fig_pipe, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
def page_forecast():
    models_dict = load_models()
    ext_series  = load_extended_series()

    st.markdown('<div class="sh">🔮  Live GRU Drought Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#7eb8d4;margin-bottom:20px;font-size:.88rem">'
        'Select a region, model mode, and target month. Jan 2023 – Apr 2026 are '
        '<strong style="color:#ffd700">simulated</strong> via GRU rolling forecast + '
        'climatological inputs. Confidence decays beyond Dec 2022.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_out = st.columns([1, 2], gap="large")

    # ── Controls ─────────────────────────────────────────────────────────────
    with col_ctrl:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        region = st.selectbox("Region", ["SADC — Southern Africa", "SEA — Southeast Asia"])
        is_sea = "SEA" in region

        if is_sea:
            mode = st.selectbox("Model Mode", ["Zero-Shot (SADC → SEA)", "Fine-Tuned (10 epochs SEA)"])
            arr   = ext_series["sea"]
            model = models_dict["fine_tuned"] if "Fine" in mode else models_dict["zero_shot"]
            mode_label = "Fine-Tuned" if "Fine" in mode else "Zero-Shot"
        else:
            st.info("SADC trained model (11,777 parameters)", icon="ℹ️")
            arr   = ext_series["sadc"]
            model = models_dict["sadc"]
            mode_label = "SADC Trained"

        valid_labels = [DATES_EXT[i].strftime("%b %Y") for i in range(12, N_TOTAL)]
        sel = st.select_slider("Forecast Target Month", options=valid_labels, value=valid_labels[-1])
        month_idx = 12 + valid_labels.index(sel)

        st.markdown('</div>', unsafe_allow_html=True)

        # Input window feature chart
        t_end   = month_idx - 1
        t_start = t_end - 12 + 1
        window  = arr[t_start : t_end + 1, :]
        inp_labels = [DATES[t_start + i].strftime("%b %Y") for i in range(12)]

        fig_inp = go.Figure()
        feat_names  = ["SPEI-1", "CHIRPS Precip", "Pacific SST", "Indian Ocean SST"]
        feat_colors = ["#00d4ff", "#00ffc8", "#ffd700", "#ff6b9d"]
        for fi, (fn, fc) in enumerate(zip(feat_names, feat_colors)):
            fig_inp.add_trace(go.Scatter(
                x=inp_labels, y=window[:, fi].tolist(),
                name=fn, line=dict(color=fc, width=1.8),
                mode="lines+markers", marker=dict(size=4),
            ))
        fig_inp.update_layout(
            **{**PL, "margin": dict(l=10, r=10, t=38, b=10)},
            height=230,
            title=dict(text="12-Month Input Window", font=dict(size=12, color="#e0f4ff")),
            xaxis=dict(**AX, tickangle=45, tickfont=dict(size=7.5)),
            yaxis=dict(**AX, title="Standardised Value"),
        )
        st.plotly_chart(fig_inp, use_container_width=True)

    # ── Output ───────────────────────────────────────────────────────────────
    with col_out:
        pred        = run_inference(arr, model, month_idx)
        observed    = float(arr[min(month_idx, N_TOTAL - 1), 0])
        persistence = float(arr[min(month_idx - 1, N_TOTAL - 2), 0])
        forecast_mo = DATES_EXT[month_idx].strftime("%B %Y")
        conf, sigma_mult = forecast_confidence(month_idx)
        is_future   = month_idx >= N_HIST

        if pred is None:
            sev, scol, sbg = "Insufficient History", "#7eb8d4", "rgba(126,184,212,.14)"
            pred = 0.0
        else:
            sev, scol, sbg = spei_severity(pred)

        # Gauge dial
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(pred, 3),
            delta=dict(reference=0.0, valueformat=".3f", font=dict(size=16, color="#7eb8d4")),
            number=dict(font=dict(size=54, color=scol), suffix=""),
            title=dict(text=f"Forecast  ·  {forecast_mo}  ·  {mode_label}", font=dict(size=15, color="#e0f4ff")),
            gauge=dict(
                axis=dict(range=[-3, 2], tickwidth=1.2, tickcolor="#7eb8d4",
                          tickfont=dict(color="#7eb8d4", size=10)),
                bar=dict(color=scol, thickness=0.26),
                bgcolor="rgba(5,14,31,.55)",
                borderwidth=1.2, bordercolor="rgba(0,212,255,.3)",
                steps=[
                    dict(range=[-3.0, -2.0], color="rgba(139,0,0,.4)"),
                    dict(range=[-2.0, -1.5], color="rgba(255,71,87,.32)"),
                    dict(range=[-1.5, -1.0], color="rgba(255,149,0,.28)"),
                    dict(range=[-1.0,  0.0], color="rgba(255,215,0,.18)"),
                    dict(range=[ 0.0,  2.0], color="rgba(0,255,200,.14)"),
                ],
                threshold=dict(line=dict(color="#ffffff", width=3), thickness=0.78, value=pred),
            ),
        ))
        fig_gauge.update_layout(**PL, height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Confidence gauge ──────────────────────────────────────────────
        conf_color = "#00ffc8" if conf >= 0.75 else "#ffd700" if conf >= 0.60 else "#ff4757"
        spei_sigma = round(0.193 * sigma_mult, 3)
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(conf * 100, 1),
            number=dict(font=dict(size=32, color=conf_color), suffix="%"),
            title=dict(
                text=f"Forecast Confidence  ·  ±{spei_sigma} SPEI (1σ)"
                     + ("  <span style='color:#ffd700'>  SIMULATED</span>" if is_future else ""),
                font=dict(size=11, color="#e0f4ff"),
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=9, color="#7eb8d4")),
                bar=dict(color=conf_color, thickness=0.28),
                bgcolor="rgba(5,14,31,.55)",
                borderwidth=1, bordercolor="rgba(0,212,255,.3)",
                steps=[
                    dict(range=[0,  50], color="rgba(255,71,87,.18)"),
                    dict(range=[50, 75], color="rgba(255,215,0,.14)"),
                    dict(range=[75,100], color="rgba(0,255,200,.12)"),
                ],
            ),
        ))
        fig_conf.update_layout(**PL, height=220)
        st.plotly_chart(fig_conf, use_container_width=True)

        # Severity badge
        st.markdown(
            f'<div class="sbadge" style="background:{sbg};border:2px solid {scol};color:{scol}">'
            f'{sev}</div>',
            unsafe_allow_html=True,
        )

        # Comparison row
        ca, cb, cc = st.columns(3)
        ca.markdown(metric_card(f"{pred:.3f}", "GRU Forecast", "SPEI-1"), unsafe_allow_html=True)
        cb.markdown(metric_card(f"{observed:.3f}", "Observed SPEI", "Actual value"), unsafe_allow_html=True)
        cc.markdown(metric_card(f"{persistence:.3f}", "Persistence", "Naïve baseline"), unsafe_allow_html=True)

        # SPEI legend
        st.markdown("""
<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:14px;font-size:.73rem;color:#7eb8d4">
  <span style="color:#00ffc8">● Normal (≥ 0)</span>
  <span style="color:#ffd700">● Mild Dry (−1 to 0)</span>
  <span style="color:#ff9500">● Moderate (−1.5 to −1)</span>
  <span style="color:#ff4757">● Severe (−2 to −1.5)</span>
  <span style="color:#cc0000">● Extreme (&lt; −2)</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VALIDATION TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
def page_timeseries():
    ts = load_csv_timeseries()

    st.markdown('<div class="sh">📈  Validation Time Series  (Sep 2018 – Dec 2022,  n = 52 months)</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🟦  SADC — Mean-GRU vs Persistence",
        "🟣  SEA — Zero-Shot Transfer",
        "🟢  SEA — Fine-Tuned",
    ])

    def ts_figure(df, pred_col, pred_label, pred_color, title, sigma=0.193):
        fig = go.Figure()
        drought_band_traces(fig, df)

        # ±1σ confidence ribbon around model prediction
        upper = (df[pred_col] + sigma).tolist()
        lower = (df[pred_col] - sigma).tolist()
        fig.add_trace(go.Scatter(
            x=df["time"].tolist() + df["time"].tolist()[::-1],
            y=upper + lower[::-1],
            fill="toself", fillcolor=f"rgba({int(pred_color[1:3],16)},"
                                     f"{int(pred_color[3:5],16)},"
                                     f"{int(pred_color[5:7],16)},0.10)",
            line=dict(width=0), name=f"±1σ ({sigma:.3f})",
            showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["true"],
            name="Observed SPEI-1",
            line=dict(color="#3a9fff", width=2.8), mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[pred_col],
            name=pred_label,
            line=dict(color=pred_color, width=2.2, dash="dash"), mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["pred_persistence"],
            name="Persistence (naïve baseline)",
            line=dict(color="#ffd700", width=1.6, dash="dot"), mode="lines",
        ))
        hline_drought(fig)
        fig.update_layout(
            **PL, height=430,
            title=dict(text=title, font=dict(size=14)),
            xaxis=dict(**AX, title="Month"),
            yaxis=dict(**AX, title="SPEI-1  (standardised)", range=[-3, 1.8]),
        )
        return fig

    with tab1:
        fig1 = ts_figure(
            ts["sadc"], "pred_mean_gru",
            "Mean-GRU  (r = 0.835)",  "#00ffc8",
            "SADC Validation — Mean-GRU vs Persistence  (RMSE = 0.193 vs 0.202)",
            {},
        )
        st.plotly_chart(fig1, use_container_width=True)
        ca, cb, cc = st.columns(3)
        ca.markdown(metric_card("0.193", "RMSE", "Mean-GRU"), unsafe_allow_html=True)
        cb.markdown(metric_card("0.162", "MAE",  "Mean-GRU"), unsafe_allow_html=True)
        cc.markdown(metric_card("0.835", "Pearson r", "Mean-GRU"), unsafe_allow_html=True)

    with tab2:
        fig2 = ts_figure(
            ts["sea_zs"], "pred_zeroshot",
            "Zero-Shot Transfer  (r = 0.882)", "#a855f7",
            "SEA Zero-Shot Transfer — SADC Weights Applied Directly  (RMSE = 0.311)",
            {},
        )
        st.plotly_chart(fig2, use_container_width=True)
        ca, cb, cc = st.columns(3)
        ca.markdown(metric_card("0.311", "RMSE", "Zero-Shot"), unsafe_allow_html=True)
        cb.markdown(metric_card("0.252", "MAE",  "Zero-Shot"), unsafe_allow_html=True)
        cc.markdown(metric_card("0.882", "Pearson r", "Zero-Shot"), unsafe_allow_html=True)

    with tab3:
        fig3 = ts_figure(
            ts["sea_ft"], "pred_finetuned",
            "Fine-Tuned  (r = 0.903)", "#00ffc8",
            "SEA Fine-Tuned — 10-Epoch Adaptation from SADC Checkpoint  (RMSE = 0.236)",
            {},
        )
        st.plotly_chart(fig3, use_container_width=True)
        ca, cb, cc = st.columns(3)
        ca.markdown(metric_card("0.236", "RMSE", "Fine-Tuned"), unsafe_allow_html=True)
        cb.markdown(metric_card("0.195", "MAE",  "Fine-Tuned"), unsafe_allow_html=True)
        cc.markdown(metric_card("0.903", "Pearson r", "Fine-Tuned"), unsafe_allow_html=True)

    # ── Future forecast 2023-2026 ─────────────────────────────────────────────
    st.markdown(
        '<div class="sh" style="margin-top:28px;font-size:1.1rem">'
        '🔭  Simulated Forecast — Jan 2023 – Apr 2026'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="color:#ffd700;font-size:.82rem;margin-bottom:12px">'
        '⚠ Beyond Dec 2022 training horizon — GRU rolling forecast with climatological inputs. '
        'Shaded band shows growing ±1σ uncertainty. For indicative purposes only.'
        '</div>',
        unsafe_allow_html=True,
    )
    ext = load_extended_series()
    fut_times_sadc = [DATES_EXT[i] for i in range(N_HIST, N_TOTAL)]
    fut_times_sea  = fut_times_sadc

    fig_fut = go.Figure()
    for region, color, label in [
        ("sadc", "#00d4ff", "SADC (Simulated)"),
        ("sea",  "#00ffc8", "SEA Fine-Tuned (Simulated)"),
    ]:
        vals = [float(ext[region][i, 0]) for i in range(N_HIST, N_TOTAL)]
        sigmas = [0.193 * (1.0 + (i - N_HIST + 1) * 0.07) for i in range(N_HIST, N_TOTAL)]
        upper = [v + s for v, s in zip(vals, sigmas)]
        lower = [v - s for v, s in zip(vals, sigmas)]
        fig_fut.add_trace(go.Scatter(
            x=fut_times_sadc + fut_times_sadc[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_fut.add_trace(go.Scatter(
            x=fut_times_sadc, y=vals,
            name=label, line=dict(color=color, width=2, dash="dash"), mode="lines",
        ))
    hline_drought(fig_fut)
    fig_fut.update_layout(
        **PL, height=380,
        xaxis=dict(**AX, title="Month"),
        yaxis=dict(**AX, title="SPEI-1 Forecast", range=[-3, 1.8]),
    )
    st.plotly_chart(fig_fut, use_container_width=True)

    # Cross-domain comparison
    st.markdown('<div class="sh" style="margin-top:28px;font-size:1.1rem">Transfer Improvement — Side-by-Side</div>',
                unsafe_allow_html=True)
    fig_cmp = go.Figure()
    regions   = ["SADC\nMean-GRU", "SEA\nZero-Shot", "SEA\nFine-Tuned"]
    r_vals    = [0.835, 0.882, 0.903]
    rmse_vals = [0.193, 0.311, 0.236]
    bar_c     = ["#00d4ff", "#a855f7", "#00ffc8"]

    fig_cmp.add_trace(go.Bar(x=regions, y=r_vals, name="Pearson r (↑ better)",
                              marker_color=bar_c, yaxis="y",
                              text=[f"{v:.3f}" for v in r_vals], textposition="outside",
                              textfont=dict(color="#e0f4ff", size=12)))
    fig_cmp.add_trace(go.Bar(x=regions, y=rmse_vals, name="RMSE (↓ better)",
                              marker_color=bar_c, marker_opacity=0.55, yaxis="y2",
                              text=[f"{v:.3f}" for v in rmse_vals], textposition="outside",
                              textfont=dict(color="#ffd700", size=12)))
    fig_cmp.update_layout(
        **PL, height=360, barmode="group",
        xaxis=dict(**AX),
        yaxis=dict(**AX, title="Pearson r", range=[0, 1.05]),
        yaxis2=dict(**AX, title="RMSE", overlaying="y", side="right", range=[0, 0.42]),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SPATIAL MAPS + SEA MOVEMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_maps():
    z = load_zarr_data()

    st.markdown('<div class="sh">🌊  Spatial Climate Maps — Animated Sea & Drought Patterns</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div style="color:#7eb8d4;margin-bottom:16px;font-size:.86rem">
Press <strong style="color:#00d4ff">▶ Play</strong> to animate across Jan 2000 – Dec 2022.
The Pacific SST tab reveals El Niño/La Niña sea-surface temperature movement across the ocean.
The Indian Ocean tab shows IOD oscillation. SPEI tabs show how drought patterns develop and recede.
</div>""", unsafe_allow_html=True)

    # Extend spatial arrays with climatological simulations for 2023-2026
    spei_sadc_ext  = build_extended_spatial(z["spei_sadc"])
    spei_sea_ext   = build_extended_spatial(z["spei_sea"])
    chirps_sadc_ext= build_extended_spatial(z["chirps_sadc"])
    chirps_sea_ext = build_extended_spatial(z["chirps_sea"])
    sst_pac_ext    = build_extended_spatial(z["sst_pac"])
    sst_ind_ext    = build_extended_spatial(z["sst_ind"])

    # Every 3rd month → ~105 frames
    t_idx    = list(range(0, N_TOTAL, 3))
    d_labels = [
        DATES_EXT[i].strftime("%b %Y") + ("*" if i >= N_HIST else "")
        for i in t_idx
    ]
    st.markdown(
        '<div style="color:#ffd700;font-size:.8rem;margin-bottom:8px">'
        '* months marked with ★ are simulated beyond Dec 2022 training data.'
        '</div>', unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔴  SPEI — Southern Africa",
        "🔴  SPEI — Southeast Asia",
        "🌊  Pacific SST  (El Niño/La Niña)",
        "🌊  Indian Ocean SST  (IOD)",
        "🌧️  Rainfall  (CHIRPS)",
    ])

    with tab1:
        st.caption("SPEI-1 over SADC (5°S–35°S). Red = drought, blue = wet. * = simulated beyond 2022.")
        fig = animated_heatmap(
            spei_sadc_ext, z["lat_sadc"], z["lon_sadc"],
            t_idx, d_labels, "RdBu", -2.5, 2.5,
            "SPEI-1  |  Southern Africa (SADC)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.caption("SPEI-1 over Southeast Asia (5°S–25°N). * = simulated beyond 2022.")
        fig = animated_heatmap(
            spei_sea_ext, z["lat_sea"], z["lon_sea"],
            t_idx, d_labels, "RdBu", -2.5, 2.5,
            "SPEI-1  |  Southeast Asia",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.caption("Pacific SST anomalies. Red = El Niño, blue = La Niña. * = simulated beyond 2022.")
        fig = animated_heatmap(
            sst_pac_ext, z["lat_pac"], z["lon_pac"],
            t_idx, d_labels, "RdBu_r", -2.0, 2.0,
            "Pacific SST Anomaly  |  El Niño / La Niña Sea Movement",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.caption("Indian Ocean SST anomalies (IOD). * = simulated beyond 2022.")
        fig = animated_heatmap(
            sst_ind_ext, z["lat_ind"], z["lon_ind"],
            t_idx, d_labels, "RdBu_r", -1.5, 1.5,
            "Indian Ocean SST Anomaly  |  IOD Sea Movement",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.caption("CHIRPS v2.0 precipitation. * = simulated beyond 2022.")
        tab5a, tab5b = st.tabs(["CHIRPS — SADC", "CHIRPS — SEA"])
        with tab5a:
            fig_r = animated_heatmap(
                chirps_sadc_ext, z["lat_sadc"], z["lon_sadc"],
                t_idx, d_labels, "Blues", 0, 250,
                "CHIRPS Precipitation  |  Southern Africa (mm/month)",
            )
            st.plotly_chart(fig_r, use_container_width=True)
        with tab5b:
            fig_r2 = animated_heatmap(
                chirps_sea_ext, z["lat_sea"], z["lon_sea"],
                t_idx, d_labels, "Blues", 0, 400,
                "CHIRPS Precipitation  |  Southeast Asia (mm/month)",
            )
            st.plotly_chart(fig_r2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_performance():
    hist = load_training_history()

    st.markdown('<div class="sh">📊  Model Performance & Training Analysis</div>', unsafe_allow_html=True)

    # ── All-model metrics bar chart ───────────────────────────────────────────
    st.markdown('<div class="sh" style="font-size:1.1rem">Evaluation Metrics — All Models & Regions</div>',
                unsafe_allow_html=True)

    model_names = ["SADC\nMean-GRU", "SADC\nCNN-GRU", "SADC\nPersist.", "SEA\nZero-Shot", "SEA\nFine-Tuned", "SEA\nPersist."]
    rmse_all    = [0.193, 0.227, 0.202, 0.311, 0.236, 0.184]
    mae_all     = [0.162, 0.195, 0.148, 0.252, 0.195, 0.144]
    corr_all    = [0.835, 0.758, 0.807, 0.882, 0.903, 0.917]
    bar_colors  = ["#00d4ff", "#7b61ff", "#ffd700", "#a855f7", "#00ffc8", "#3a9fff"]

    fig_bar = make_subplots(
        rows=1, cols=3,
        subplot_titles=("RMSE  (↓ better)", "MAE  (↓ better)", "Pearson r  (↑ better)"),
    )
    for ci, vals in enumerate([rmse_all, mae_all, corr_all], 1):
        fig_bar.add_trace(go.Bar(
            x=model_names, y=vals,
            marker_color=bar_colors,
            marker_line=dict(color="rgba(0,212,255,.3)", width=1),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=10.5, color="#e0f4ff"),
            showlegend=False,
        ), row=1, col=ci)

    fig_bar.update_layout(
        **PL, height=400,
        xaxis =dict(**AX, tickfont=dict(size=8.5)),
        xaxis2=dict(**AX, tickfont=dict(size=8.5)),
        xaxis3=dict(**AX, tickfont=dict(size=8.5)),
        yaxis =dict(**AX), yaxis2=dict(**AX), yaxis3=dict(**AX),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Training loss curves ──────────────────────────────────────────────────
    st.markdown('<div class="sh" style="font-size:1.1rem;margin-top:22px">Training Loss Curves</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        s    = hist["sadc"]
        ep   = [r["epoch"]     for r in s]
        tr   = [r["train_mse"] for r in s]
        va   = [r["val_mse"]   for r in s]
        best_v = min(va)

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=ep, y=tr, name="Train MSE",
                                    line=dict(color="#00d4ff", width=2.2)))
        fig_s.add_trace(go.Scatter(x=ep, y=va, name="Val MSE",
                                    line=dict(color="#ff4757", width=2.2, dash="dash")))
        fig_s.add_hline(y=best_v, line_color="#ffd700", line_dash="dot", line_width=1.2,
                         annotation_text=f"Best val MSE = {best_v:.4f}",
                         annotation_font_color="#ffd700", annotation_position="bottom right")
        fig_s.update_layout(
            **PL, height=320,
            title=dict(text="SADC Mean-GRU — 40 Epochs  (11,777 params)", font=dict(size=13)),
            xaxis=dict(**AX, title="Epoch"),
            yaxis=dict(**AX, title="MSE Loss"),
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown(metric_card(f"{best_v:.4f}", "Best Val MSE", "Epoch 40"), unsafe_allow_html=True)

    with col2:
        s2   = hist["sea"]
        ep2  = [r["epoch"]     for r in s2]
        tr2  = [r["train_mse"] for r in s2]
        va2  = [r["val_mse"]   for r in s2]
        best_v2 = min(va2)

        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Scatter(x=ep2, y=tr2, name="Train MSE",
                                     line=dict(color="#00ffc8", width=2.2)))
        fig_s2.add_trace(go.Scatter(x=ep2, y=va2, name="Val MSE",
                                     line=dict(color="#a855f7", width=2.2, dash="dash")))
        fig_s2.add_hline(y=best_v2, line_color="#ffd700", line_dash="dot", line_width=1.2,
                          annotation_text=f"Best val MSE = {best_v2:.4f}",
                          annotation_font_color="#ffd700", annotation_position="bottom right")
        fig_s2.update_layout(
            **PL, height=320,
            title=dict(text="SEA Fine-Tune — 10 Epochs  (from SADC checkpoint)", font=dict(size=13)),
            xaxis=dict(**AX, title="Epoch"),
            yaxis=dict(**AX, title="MSE Loss"),
        )
        st.plotly_chart(fig_s2, use_container_width=True)
        st.markdown(metric_card(f"{best_v2:.4f}", "Best Val MSE", "Epoch 10"), unsafe_allow_html=True)

    # ── Transfer learning journey ─────────────────────────────────────────────
    st.markdown('<div class="sh" style="font-size:1.1rem;margin-top:26px">Transfer Learning Journey</div>',
                unsafe_allow_html=True)

    steps     = ["Step 1<br>SADC Training<br>(Source Domain)",
                 "Step 2<br>Zero-Shot Transfer<br>(No Fine-tuning)",
                 "Step 3<br>Fine-Tuned<br>(10 Epochs SEA)"]
    r_steps   = [0.835, 0.882, 0.903]
    rmse_steps= [0.193, 0.311, 0.236]
    s_colors  = ["#00d4ff", "#a855f7", "#00ffc8"]

    fig_xfer = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pearson r  (↑ better)", "RMSE  (↓ better)"),
    )
    kw = dict(mode="lines+markers+text", textposition="top center",
              textfont=dict(color="#e0f4ff", size=12.5))

    fig_xfer.add_trace(go.Scatter(
        x=steps, y=r_steps, line=dict(color="#00ffc8", width=3),
        marker=dict(size=16, color=s_colors, line=dict(color="white", width=2)),
        text=[f"<b>{v:.3f}</b>" for v in r_steps], **kw, showlegend=False,
    ), row=1, col=1)

    fig_xfer.add_trace(go.Scatter(
        x=steps, y=rmse_steps, line=dict(color="#ff9500", width=3),
        marker=dict(size=16, color=s_colors, line=dict(color="white", width=2)),
        text=[f"<b>{v:.3f}</b>" for v in rmse_steps], **kw, showlegend=False,
    ), row=1, col=2)

    fig_xfer.update_layout(
        **PL, height=330,
        xaxis =dict(**AX, tickfont=dict(size=9)),
        xaxis2=dict(**AX, tickfont=dict(size=9)),
        yaxis =dict(**AX, title="Pearson r",   range=[0.75, 0.96]),
        yaxis2=dict(**AX, title="RMSE",        range=[0.15, 0.37]),
    )
    st.plotly_chart(fig_xfer, use_container_width=True)

    # ── Key insight ───────────────────────────────────────────────────────────
    st.markdown("""
<div class="glass" style="border-left:3px solid #00d4ff;padding-left:22px;margin-top:16px">
  <div style="color:#00d4ff;font-weight:700;font-size:1rem;margin-bottom:10px">
    💡  Key Transfer Learning Insight
  </div>
  <div style="color:#b8d8f0;font-size:.9rem;line-height:1.85">
    The GRU hidden states encode <strong style="color:#00ffc8">physically transferable teleconnection
    patterns</strong> — specifically, El Niño–Southern Oscillation (ENSO) coupling between Pacific SST
    anomalies and regional precipitation deficits. Because this coupling operates in both SADC
    (through Walker Circulation) and SEA (through monsoon suppression), the zero-shot correlation
    <strong style="color:#00d4ff">r = 0.882</strong> substantially exceeds random weights.
    Just 10 fine-tuning epochs further calibrate the model's output distribution to SEA climate
    statistics, pushing correlation to <strong style="color:#00ffc8">r = 0.903</strong>.
    The regularisation term (λ = 0.01) prevents catastrophic forgetting of SADC-learned ENSO
    representations during fine-tuning.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Architecture summary ──────────────────────────────────────────────────
    st.markdown('<div class="sh" style="font-size:1.1rem;margin-top:26px">Architecture Comparison</div>',
                unsafe_allow_html=True)

    arch_data = {
        "": ["Mean-GRU (SADC)", "CNN-GRU (SADC)", "Persistence"],
        "Parameters":    ["11,777", "~48,321", "—"],
        "RMSE":          ["0.193 ✓", "0.227", "0.202"],
        "MAE":           ["0.162", "0.195", "0.148 ✓"],
        "Pearson r":     ["0.835 ✓", "0.758", "0.807"],
        "Transfer":      ["Zero-shot r=0.882", "Not evaluated", "N/A"],
    }
    df_arch = pd.DataFrame(arch_data)
    st.dataframe(
        df_arch.set_index(""),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — GEOGRAPHICAL SCENE  (Mapbox GL JS + Canvas rain + SST strips)
# ══════════════════════════════════════════════════════════════════════════════

MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")

# Raw HTML/JS template — uses __PLACEHOLDER__ so no f-string escaping needed
_GEO_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet">
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Fraunces:opsz,wght@9..144,600;9..144,900&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body {
  height:100%; overflow:hidden;
  background: radial-gradient(ellipse at 30% 40%, #0a1a2e 0%, #020812 70%);
  color:#e0f4ff; font-family:'Space Grotesk',system-ui,sans-serif;
}

#scene {
  display:grid;
  grid-template-rows: 44px 1fr 52px;
  height:100vh;
  gap:4px;
  padding:6px;
}

/* ── controls bar ── */
#ctrl {
  display:flex;
  align-items:center;
  gap:10px;
  padding:0 14px;
  background:rgba(2,8,18,.88);
  border:1px solid rgba(0,212,255,.22);
  border-radius:10px;
  backdrop-filter:blur(8px);
}
#play-btn {
  background:rgba(0,212,255,.12);
  border:1px solid rgba(0,212,255,.45);
  border-radius:6px;
  color:#00d4ff;
  padding:4px 16px;
  font-size:12px;
  font-weight:700;
  cursor:pointer;
  flex-shrink:0;
  transition:background .15s, box-shadow .15s;
  letter-spacing:.5px;
}
#play-btn:hover { background:rgba(0,212,255,.28); box-shadow:0 0 12px rgba(0,212,255,.3); }
#month-slider { flex:1; accent-color:#00d4ff; cursor:pointer; }
#month-label {
  font-size:12px; font-weight:700; color:#00d4ff;
  min-width:72px; text-align:center; flex-shrink:0; letter-spacing:.5px;
}
.badge { font-size:10.5px; font-weight:700; padding:3px 11px; border-radius:20px; flex-shrink:0; transition:all .3s; }
#enso-badge {
  font-size:11px; font-weight:700; padding:3px 12px; border-radius:20px;
  border:1px solid rgba(0,212,255,.3); color:#e0f4ff;
  background:rgba(0,212,255,.08); flex-shrink:0;
}

/* ── maps row ── */
#maps-row {
  display:grid;
  grid-template-columns: 58fr 42fr;
  gap:5px;
}

/* Globe panel */
#globe-wrap {
  position:relative;
  border-radius:14px;
  overflow:hidden;
  border:1px solid rgba(0,212,255,.2);
  box-shadow: 0 0 40px rgba(0,212,255,.08), inset 0 0 30px rgba(2,8,18,.5);
}
#globe-map { width:100%; height:100%; }

.globe-badge {
  position:absolute; top:12px; left:50%; transform:translateX(-50%);
  z-index:10; background:rgba(2,8,18,.88); border:1px solid rgba(0,212,255,.35);
  border-radius:8px; padding:5px 16px; font-size:10.5px; font-weight:700;
  color:#00d4ff; letter-spacing:1.2px; text-transform:uppercase;
  backdrop-filter:blur(4px); white-space:nowrap;
}
.globe-enso {
  position:absolute; bottom:12px; left:50%; transform:translateX(-50%);
  z-index:10; background:rgba(2,8,18,.88); border:1px solid rgba(0,212,255,.2);
  border-radius:8px; padding:5px 14px; font-size:10px; font-weight:600;
  color:#7eb8d4; text-align:center; white-space:nowrap;
}

/* Detail column */
#detail-col { display:grid; grid-template-rows:1fr 1fr; gap:5px; }

.map-wrap {
  position:relative; border-radius:12px; overflow:hidden;
  border:1px solid rgba(0,212,255,.18);
  box-shadow:0 0 18px rgba(0,212,255,.07);
}
.mapbox-map { width:100%; height:100%; }
canvas.rain-canvas {
  position:absolute; top:0; left:0;
  width:100% !important; height:100% !important;
  pointer-events:none; z-index:5;
}
.map-badge {
  position:absolute; top:8px; left:8px; z-index:10;
  background:rgba(2,8,18,.9); border:1px solid rgba(0,212,255,.3);
  border-radius:6px; padding:4px 10px; font-size:10px; font-weight:700;
  color:#00d4ff; letter-spacing:.8px; text-transform:uppercase;
}
.spei-badge {
  position:absolute; bottom:8px; left:8px; z-index:10;
  background:rgba(2,8,18,.88); border:1px solid rgba(0,212,255,.2);
  border-radius:6px; padding:4px 10px; font-size:10px; font-weight:700;
  transition:color .4s;
}
#spei-legend {
  position:absolute; bottom:8px; right:8px; z-index:10;
  background:rgba(2,8,18,.9); border:1px solid rgba(0,212,255,.15);
  border-radius:7px; padding:6px 9px; font-size:8.5px; color:#7eb8d4;
}
.leg { display:flex; align-items:center; gap:5px; margin:2px 0; }
.sw  { width:12px; height:6px; border-radius:2px; flex-shrink:0; }

/* ── SST strips ── */
#sst-row { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
.sst-panel {
  position:relative; border-radius:8px; overflow:hidden;
  border:1px solid rgba(0,212,255,.12); background:#020812;
}
canvas.sst-canvas { width:100%; height:100%; display:block; }
.sst-lbl {
  position:absolute; top:50%; left:10px; transform:translateY(-50%);
  font-size:9.5px; color:rgba(0,212,255,.65); font-weight:600;
  letter-spacing:.7px; pointer-events:none; z-index:5;
  text-shadow:0 1px 6px #020812;
}
</style>
</head>
<body>
<div id="scene">

  <!-- Controls -->
  <div id="ctrl">
    <button id="play-btn">&#9654;&nbsp; Play</button>
    <input type="range" id="month-slider" min="0" max="315" value="315" step="1">
    <span id="month-label">Apr 2026</span>
    <span class="badge" id="sadc-badge">SADC</span>
    <span class="badge" id="sea-badge">SEA</span>
    <span id="enso-badge">ENSO: neutral</span>
    <span id="sim-badge" style="display:none;padding:2px 10px;border-radius:14px;font-size:.68rem;font-weight:700;letter-spacing:.6px;border:1px solid rgba(168,85,247,.6);color:#d8b4fe;background:rgba(168,85,247,.12);">&#x1F52E; SIMULATED</span>
    <span id="conf-label" style="display:none;font-size:.68rem;color:#a78bfa;font-weight:600;"></span>
  </div>

  <!-- Maps -->
  <div id="maps-row">

    <!-- Globe (left 58%) -->
    <div id="globe-wrap">
      <div id="globe-map"></div>
      <div class="globe-badge">&#127760; CLIMATE-XFER &mdash; Teleconnection Globe</div>
      <div class="globe-enso" id="globe-enso-txt">SST Pacific: &mdash;</div>
    </div>

    <!-- Detail maps (right 42%) -->
    <div id="detail-col">

      <!-- SADC -->
      <div class="map-wrap">
        <div id="sadc-map" class="mapbox-map"></div>
        <canvas class="rain-canvas" id="rain-sadc"></canvas>
        <div class="map-badge">&#127757; Southern Africa &mdash; SADC</div>
        <div class="spei-badge" id="sadc-spei">SPEI-1: &mdash;</div>
        <div id="spei-legend">
          <div class="leg"><div class="sw" style="background:#0d47a1"></div>Wet (&gt;1.5)</div>
          <div class="leg"><div class="sw" style="background:#90caf9"></div>Normal (0)</div>
          <div class="leg"><div class="sw" style="background:#fff9c4"></div>Near-normal</div>
          <div class="leg"><div class="sw" style="background:#ffe082"></div>Mild dry</div>
          <div class="leg"><div class="sw" style="background:#ff8f00"></div>Moderate</div>
          <div class="leg"><div class="sw" style="background:#e53935"></div>Severe</div>
          <div class="leg"><div class="sw" style="background:#7b0000"></div>Extreme</div>
        </div>
      </div>

      <!-- SEA -->
      <div class="map-wrap">
        <div id="sea-map" class="mapbox-map"></div>
        <canvas class="rain-canvas" id="rain-sea"></canvas>
        <div class="map-badge">&#127758; Southeast Asia</div>
        <div class="spei-badge" id="sea-spei">SPEI-1: &mdash;</div>
      </div>

    </div>
  </div>

  <!-- SST ocean strips -->
  <div id="sst-row">
    <div class="sst-panel">
      <canvas class="sst-canvas" id="canv-pac"></canvas>
      <div class="sst-lbl">&#127754; Pacific SST &mdash; El Ni&#241;o / La Ni&#241;a</div>
    </div>
    <div class="sst-panel">
      <canvas class="sst-canvas" id="canv-ind"></canvas>
      <div class="sst-lbl">&#127754; Indian Ocean SST &mdash; IOD</div>
    </div>
  </div>

</div><!-- /scene -->

<script>
// ── Data injected by Python ──────────────────────────────────
const SPEI_SADC   = __SPEI_SADC__;
const SPEI_SEA    = __SPEI_SEA__;
const CHIRPS_SADC = __CHIRPS_SADC__;
const CHIRPS_SEA  = __CHIRPS_SEA__;
const SST_PAC     = __SST_PAC__;
const SST_IND     = __SST_IND__;
const DATES       = __DATES__;

const SADC_ISO = ['AGO','BWA','COD','SWZ','LSO','MDG','MWI','MOZ','NAM','ZAF','TZA','ZMB','ZWE','COM','MUS','SYC'];
const SEA_ISO  = ['MMR','THA','LAO','KHM','VNM','MYS','SGP','BRN','IDN','PHL','TLS'];
const SADC_CENTER = [25, -20];
const SEA_CENTER  = [115, 8];

// ── CHIRPS normalisation for rain density ────────────────────
const chMaxS = Math.max(...CHIRPS_SADC), chMinS = Math.min(...CHIRPS_SADC);
const chMaxE = Math.max(...CHIRPS_SEA),  chMinE = Math.min(...CHIRPS_SEA);
let rIntS = 0.5, rIntE = 0.5;
function normCh(v, mn, mx) { return Math.max(0, Math.min(1, (v - mn) / (mx - mn + 1e-6))); }

// ── SPEI colour scale ────────────────────────────────────────
function speiCol(v) {
  if (v >= 1.5) return '#0d47a1';
  if (v >= 1.0) return '#1565c0';
  if (v >= 0.5) return '#1e88e5';
  if (v >= 0.0) return '#90caf9';
  if (v >=-0.5) return '#fff9c4';
  if (v >=-1.0) return '#ffe082';
  if (v >=-1.5) return '#ff8f00';
  if (v >=-2.0) return '#e53935';
  return '#7b0000';
}
function speiTxt(v, prefix) {
  const lab = v>=0?'Normal':v>=-1?'Mild Dry':v>=-1.5?'Moderate':v>=-2?'Severe':'Extreme';
  return { text: `${prefix}  SPEI ${v.toFixed(3)}  \u2014  ${lab}`,
           color: v>=0?'#90caf9':v>=-1?'#ffe082':v>=-1.5?'#ff8f00':v>=-2?'#e53935':'#ff6659' };
}
function badgeCss(v, label) {
  if (v >= 0)   return `background:rgba(144,202,249,.13);border:1px solid #90caf9;color:#90caf9;` + label + ': OK';
  if (v >=-1)   return `background:rgba(255,224,130,.12);border:1px solid #ffe082;color:#ffe082;` + label + ': Dry';
  if (v >=-1.5) return `background:rgba(255,143,0,.14);border:1px solid #ff8f00;color:#ff8f00;`  + label + ': Moderate';
  return `background:rgba(229,57,53,.16);border:1px solid #e53935;color:#ff6659;` + label + ': DROUGHT';
}

// ── Mapbox ───────────────────────────────────────────────────
mapboxgl.accessToken = '__TOKEN__';

// Globe
const globeMap = new mapboxgl.Map({
  container: 'globe-map',
  style: 'mapbox://styles/mapbox/satellite-v9',
  center: [70, -6],
  zoom: 1.35,
  projection: 'globe',
  attributionControl: false,
  interactive: true,
});

// Detail maps — topographic with 3D terrain
const sadcMap = new mapboxgl.Map({
  container: 'sadc-map',
  style: 'mapbox://styles/mapbox/outdoors-v12',
  center: [25, -22],
  zoom: 2.6,
  attributionControl: false,
  pitch: 35,
});
const seaMap = new mapboxgl.Map({
  container: 'sea-map',
  style: 'mapbox://styles/mapbox/outdoors-v12',
  center: [113, 5],
  zoom: 2.5,
  attributionControl: false,
  pitch: 35,
});

const N_HIST_JS = 276;   // Jan 2000 – Dec 2022 (historical boundary)
const N_TOTAL_JS = 316;  // Jan 2000 – Apr 2026 (full extended range)
let globeReady = false, sadcReady = false, seaReady = false, curIdx = 315;

// ── Slow cinematic auto-rotation ─────────────────────────────
let autoRotate = true, lastRafTs = 0;
function rotateGlobe(ts) {
  requestAnimationFrame(rotateGlobe);
  if (!globeReady || !autoRotate) { lastRafTs = ts; return; }
  const dt = ts - lastRafTs; lastRafTs = ts;
  if (dt <= 0 || dt > 200) return;
  const c = globeMap.getCenter();
  globeMap.setCenter([(c.lng + dt * 0.007) % 360, c.lat]);
}
requestAnimationFrame(rotateGlobe);
globeMap.on('mousedown',  () => { autoRotate = false; });
globeMap.on('touchstart', () => { autoRotate = false; });

// ── Globe load ───────────────────────────────────────────────
globeMap.on('load', () => {
  globeMap.setFog({
    'space-color': '#020812',
    'star-intensity': 0.25,
    'horizon-blend': 0.02,
    'color': 'rgba(10,26,46,0.6)',
    'high-color': 'rgba(36,92,223,0.5)',
  });

  // Teleconnection great-circle arc
  globeMap.addSource('tc-arc', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'LineString', coordinates: [] } }
  });
  globeMap.addLayer({
    id: 'tc-line', type: 'line', source: 'tc-arc',
    paint: { 'line-color': '#ffd700', 'line-width': 2.5, 'line-opacity': 0 }
  });

  // Region rings
  globeMap.addSource('sadc-pt', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'Point', coordinates: SADC_CENTER } }
  });
  globeMap.addLayer({
    id: 'sadc-ring', type: 'circle', source: 'sadc-pt',
    paint: {
      'circle-radius': 20, 'circle-color': 'rgba(0,0,0,0)',
      'circle-stroke-color': 'rgba(0,212,255,0.7)', 'circle-stroke-width': 2.5,
    }
  });
  globeMap.addSource('sea-pt', {
    type: 'geojson',
    data: { type: 'Feature', geometry: { type: 'Point', coordinates: SEA_CENTER } }
  });
  globeMap.addLayer({
    id: 'sea-ring', type: 'circle', source: 'sea-pt',
    paint: {
      'circle-radius': 20, 'circle-color': 'rgba(0,0,0,0)',
      'circle-stroke-color': 'rgba(0,252,200,0.7)', 'circle-stroke-width': 2.5,
    }
  });

  globeReady = true;
  updateAll(curIdx);
});

// ── Detail map layers — outline only, topographic basemap shows through ───
function addCountryLayer(map, isoList, fillId, initSpei) {
  if (!map.getSource(fillId + '-src')) {
    map.addSource(fillId + '-src', { type: 'vector', url: 'mapbox://mapbox.country-boundaries-v1' });
  }
  // Use 'match' (more reliable than 'in' at low zoom with vector tiles)
  const filt = ['all',
    ['match', ['get', 'iso_3166_1_alpha_3'], isoList, true, false],
    ['any', ['==', ['get', 'worldview'], 'all'], ['==', ['get', 'worldview'], 'US']]
  ];
  // Transparent fill — no colour fill, topographic terrain shows through
  if (!map.getLayer(fillId))
    map.addLayer({ id: fillId, type: 'fill', source: fillId + '-src', 'source-layer': 'country_boundaries',
      filter: filt, paint: { 'fill-color': speiCol(initSpei), 'fill-opacity': 0.0 } });
  // Soft SPEI-coloured inner glow
  if (!map.getLayer(fillId + '-glow'))
    map.addLayer({ id: fillId + '-glow', type: 'line', source: fillId + '-src', 'source-layer': 'country_boundaries',
      filter: filt, paint: { 'line-color': speiCol(initSpei), 'line-width': 7, 'line-blur': 6, 'line-opacity': 0.75 } });
  // Crisp cyan border
  if (!map.getLayer(fillId + '-border'))
    map.addLayer({ id: fillId + '-border', type: 'line', source: fillId + '-src', 'source-layer': 'country_boundaries',
      filter: filt, paint: { 'line-color': 'rgba(0,212,255,0.95)', 'line-width': 2.0 } });
}

sadcMap.on('load', () => {
  sadcMap.addSource('sadc-dem', { type:'raster-dem', url:'mapbox://mapbox.mapbox-terrain-dem-v1', tileSize:512, maxzoom:14 });
  sadcMap.setTerrain({ source:'sadc-dem', exaggeration:1.5 });
  sadcMap.addLayer({ id:'sadc-sky', type:'sky', paint:{ 'sky-type':'atmosphere','sky-atmosphere-sun':[0,70],'sky-atmosphere-sun-intensity':12 }});
  addCountryLayer(sadcMap, SADC_ISO, 'sadc-fill', SPEI_SADC[curIdx]);
  sadcReady = true; updateAll(curIdx);
});
seaMap.on('load', () => {
  seaMap.addSource('sea-dem', { type:'raster-dem', url:'mapbox://mapbox.mapbox-terrain-dem-v1', tileSize:512, maxzoom:14 });
  seaMap.setTerrain({ source:'sea-dem', exaggeration:1.5 });
  seaMap.addLayer({ id:'sea-sky', type:'sky', paint:{ 'sky-type':'atmosphere','sky-atmosphere-sun':[0,70],'sky-atmosphere-sun-intensity':12 }});
  addCountryLayer(seaMap, SEA_ISO, 'sea-fill', SPEI_SEA[curIdx]);
  seaReady = true; updateAll(curIdx);
  // Retry when country-boundary tiles arrive (fixes intermittent shapefile gap)
  seaMap.on('sourcedata', (e) => {
    if (e.sourceId === 'sea-fill-src' && e.isSourceLoaded) {
      addCountryLayer(seaMap, SEA_ISO, 'sea-fill', SPEI_SEA[curIdx]);
      updateAll(curIdx);
    }
  });
});

// ── Globe teleconnection arc (geodesic interpolation) ────────
function geodesicPts(lon1, lat1, lon2, lat2, n) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    pts.push([lon1 + t * (lon2 - lon1), lat1 + t * (lat2 - lat1)]);
  }
  return pts;
}
function updateGlobeArc(sstPac) {
  if (!globeReady) return;
  const active = Math.abs(sstPac) >= 0.6;
  const col = sstPac > 0 ? '#ff6659' : '#90caf9';
  const pts = active ? geodesicPts(SADC_CENTER[0], SADC_CENTER[1], SEA_CENTER[0], SEA_CENTER[1], 40) : [];
  globeMap.getSource('tc-arc').setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: pts } });
  globeMap.setPaintProperty('tc-line', 'line-opacity', active ? 0.88 : 0);
  globeMap.setPaintProperty('tc-line', 'line-color', col);
}

// ── Master update ────────────────────────────────────────────
function updateAll(idx) {
  curIdx = idx;
  const speiS = SPEI_SADC[idx], speiE = SPEI_SEA[idx];
  const chS = CHIRPS_SADC[idx], chE = CHIRPS_SEA[idx];
  const sstP = SST_PAC[idx];

  document.getElementById('month-label').textContent = DATES[idx];

  // Simulated period indicator
  const isSim = idx >= N_HIST_JS;
  const simBadge = document.getElementById('sim-badge');
  const confLbl  = document.getElementById('conf-label');
  if (isSim) {
    const mAhead = idx - N_HIST_JS + 1;
    const conf = Math.max(0.42, 0.85 - mAhead * 0.011);
    simBadge.style.display = 'inline';
    confLbl.style.display  = 'inline';
    confLbl.textContent    = 'Confidence: ' + Math.round(conf * 100) + '%';
  } else {
    simBadge.style.display = 'none';
    confLbl.style.display  = 'none';
  }

  if (sadcReady && sadcMap.getLayer('sadc-fill')) {
    sadcMap.setPaintProperty('sadc-fill',   'fill-color', speiCol(speiS));
    sadcMap.setPaintProperty('sadc-fill-glow', 'line-color', speiCol(speiS));
  }
  if (seaReady && seaMap.getLayer('sea-fill')) {
    seaMap.setPaintProperty('sea-fill',    'fill-color', speiCol(speiE));
    seaMap.setPaintProperty('sea-fill-glow', 'line-color', speiCol(speiE));
  }

  const lS = speiTxt(speiS, 'SADC'), lE = speiTxt(speiE, 'SEA');
  const bdS = document.getElementById('sadc-spei'); bdS.textContent = lS.text; bdS.style.color = lS.color;
  const bdE = document.getElementById('sea-spei');  bdE.textContent = lE.text; bdE.style.color = lE.color;

  const bS = document.getElementById('sadc-badge'), bE = document.getElementById('sea-badge');
  const csS = badgeCss(speiS, 'SADC'), csE = badgeCss(speiE, 'SEA');
  bS.style.cssText = csS.split(';').slice(0,-1).join(';'); bS.textContent = csS.split(';').slice(-1)[0];
  bE.style.cssText = csE.split(';').slice(0,-1).join(';'); bE.textContent = csE.split(';').slice(-1)[0];

  rIntS = normCh(chS, chMinS, chMaxS);
  rIntE = normCh(chE, chMinE, chMaxE);

  const enso = document.getElementById('enso-badge');
  const ge = document.getElementById('globe-enso-txt');
  if (sstP > 0.8) {
    enso.style.cssText = 'border:1px solid rgba(229,57,53,.6);color:#ff6659;background:rgba(229,57,53,.1);font-weight:700;padding:3px 12px;border-radius:20px;flex-shrink:0;';
    enso.textContent = '\uD83D\uDD34 El Ni\u00F1o (' + sstP.toFixed(2) + ')';
    ge.textContent = 'El Ni\u00F1o  ' + sstP.toFixed(2) + ' \u00B0C anomaly'; ge.style.color = '#ff6659';
  } else if (sstP < -0.5) {
    enso.style.cssText = 'border:1px solid rgba(144,202,249,.6);color:#90caf9;background:rgba(30,136,229,.1);font-weight:700;padding:3px 12px;border-radius:20px;flex-shrink:0;';
    enso.textContent = '\uD83D\uDD35 La Ni\u00F1a (' + sstP.toFixed(2) + ')';
    ge.textContent = 'La Ni\u00F1a  ' + sstP.toFixed(2) + ' \u00B0C anomaly'; ge.style.color = '#90caf9';
  } else {
    enso.style.cssText = 'border:1px solid rgba(0,212,255,.3);color:#e0f4ff;background:rgba(0,212,255,.08);font-weight:700;padding:3px 12px;border-radius:20px;flex-shrink:0;';
    enso.textContent = '\u26AA ENSO Neutral (' + sstP.toFixed(2) + ')';
    ge.textContent = 'ENSO Neutral  ' + sstP.toFixed(2) + ' \u00B0C'; ge.style.color = '#7eb8d4';
  }

  updateGlobeArc(sstP);
  drawSST('canv-pac', idx, SST_PAC);
  drawSST('canv-ind', idx, SST_IND);
}

// ── SST ocean strips ─────────────────────────────────────────
function sstRGB(v) {
  const t = Math.max(-1.5, Math.min(1.5, v)) / 1.5;
  let r, g, b;
  if (t >= 0) {
    r = Math.round(21  + t * (229 - 21));
    g = Math.round(101 + t * (57  - 101));
    b = Math.round(192 + t * (53  - 192));
  } else {
    const s = -t;
    r = Math.round(21  * (1 - s));
    g = Math.round(101 * (1 - s));
    b = Math.round(192 + s * (255 - 192));
  }
  return [r, g, b];
}
function drawSST(id, ci, series) {
  const canvas = document.getElementById(id); if (!canvas) return;
  const W = canvas.parentElement.clientWidth || 500;
  const H = canvas.parentElement.clientHeight || 52;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#020812'; ctx.fillRect(0, 0, W, H);
  const N = series.length, barH = H - 14;
  for (let i = 0; i < N; i++) {
    const [r, g, b] = sstRGB(series[i]);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(Math.floor(i * W / N), 0, Math.ceil(W / N) + 1, barH);
  }
  const cx = Math.round(ci * W / N);
  ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2;
  ctx.setLineDash([4, 2]);
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, barH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(126,184,212,0.7)'; ctx.font = '7.5px Inter,sans-serif';
  [2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020,2022,2024,2026].forEach(yr => {
    const xi = Math.round(((yr - 2000) * 12) * W / N);
    if (xi > 10 && xi < W - 15) ctx.fillText(yr, xi, H - 2);
  });
}

// ── Rain particle system ─────────────────────────────────────
class Rain {
  constructor(canvasId, getIntensity) {
    this.c = document.getElementById(canvasId);
    this.ctx = this.c.getContext('2d');
    this.getI = getIntensity; this.drops = [];
    this._resize(); this._init(); this._loop();
    window.addEventListener('resize', () => { this._resize(); this._init(); });
  }
  _resize() {
    const r = this.c.parentElement.getBoundingClientRect();
    this.W = this.c.width = r.width || 400;
    this.H = this.c.height = r.height || 300;
  }
  _init() { this.drops = Array.from({length:320}, () => this._drop(true)); }
  _drop(rand) {
    return { x: Math.random() * this.W, y: rand ? Math.random() * this.H : -20,
             len: 8 + Math.random() * 20, spd: 3 + Math.random() * 7,
             op: 0.2 + Math.random() * 0.5, ang: -0.12 + Math.random() * 0.08 };
  }
  _loop() {
    requestAnimationFrame(() => this._loop());
    const ctx = this.ctx, I = this.getI();
    if (I < 0.02) { ctx.clearRect(0,0,this.W,this.H); return; }
    ctx.fillStyle = 'rgba(2,8,18,0.2)'; ctx.fillRect(0, 0, this.W, this.H);
    const active = Math.floor(I * 300);
    ctx.strokeStyle = 'rgba(100,200,255,0.55)'; ctx.lineWidth = 1.3;
    for (let i = 0; i < Math.min(active, this.drops.length); i++) {
      const d = this.drops[i];
      ctx.globalAlpha = d.op * Math.min(1, I * 1.4);
      ctx.beginPath(); ctx.moveTo(d.x, d.y);
      ctx.lineTo(d.x + Math.sin(d.ang) * d.len, d.y + d.len); ctx.stroke();
      d.y += d.spd; d.x += Math.sin(d.ang) * 0.4;
      if (d.y > this.H + 20) this.drops[i] = this._drop(false);
    }
    ctx.globalAlpha = 1;
  }
}

// ── Slider + play controls ───────────────────────────────────
const slider = document.getElementById('month-slider');
slider.addEventListener('input', () => updateAll(+slider.value));

let playing = false, playTimer = null;
document.getElementById('play-btn').addEventListener('click', function() {
  playing = !playing;
  this.textContent = playing ? '\u23F8  Pause' : '\u25B6  Play';
  if (playing) {
    playTimer = setInterval(() => {
      const n = (curIdx + 1) % N_TOTAL_JS;
      slider.value = n; updateAll(n);
    }, 1500);
  } else {
    clearInterval(playTimer);
  }
});

// ── Boot — jump to current month (Apr 2026) ──────────────────
setTimeout(() => {
  new Rain('rain-sadc', () => rIntS);
  new Rain('rain-sea',  () => rIntE);
  const startIdx = N_TOTAL_JS - 1;
  slider.value = startIdx;
  updateAll(startIdx);
}, 600);
</script>
</body>
</html>"""


@st.cache_resource
def load_geo_series() -> dict:
    """Scalar area-mean series for geo scene (same as load_series but with raw chirps)."""
    z = load_zarr_data()
    def mn(arr):
        return np.nan_to_num(np.nanmean(arr, axis=(1, 2)), nan=0.0).astype(np.float32)
    return {
        "spei_sadc":   mn(z["spei_sadc"]),
        "spei_sea":    mn(z["spei_sea"]),
        "chirps_sadc": mn(z["chirps_sadc"]),
        "chirps_sea":  mn(z["chirps_sea"]),
        "sst_pac":     mn(z["sst_pac"]),
        "sst_ind":     mn(z["sst_ind"]),
    }


@st.cache_resource
def load_geo_series_extended() -> dict:
    """Geo scene scalar series extended to Apr 2026 via GRU simulation.
    Reuses load_extended_series() for SPEI (col 0) to avoid feature-shape mismatch."""
    gs  = load_geo_series()     # historical scalars (276,)
    ext = load_extended_series()  # {"sadc":(316,4), "sea":(316,4)}

    def _extend_scalar(hist: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(99)
        out = list(hist)
        for i in range(N_FUTURE):
            cal  = DATES_EXT[N_HIST + i].month
            same = [hist[j] for j in range(N_HIST) if DATES_EXT[j].month == cal]
            clim = float(np.mean(same))
            out.append(np.float32(clim + rng.normal(0, abs(clim) * 0.03 + 0.01)))
        return np.array(out, dtype=np.float32)

    return {
        "spei_sadc":   ext["sadc"][:, 0].astype(np.float32),
        "spei_sea":    ext["sea"][:, 0].astype(np.float32),
        "chirps_sadc": _extend_scalar(gs["chirps_sadc"]),
        "chirps_sea":  _extend_scalar(gs["chirps_sea"]),
        "sst_pac":     _extend_scalar(gs["sst_pac"]),
        "sst_ind":     _extend_scalar(gs["sst_ind"]),
    }


def build_geo_html(token: str) -> str:
    gs = load_geo_series_extended()
    d_labels = [d.strftime("%b %Y") for d in DATES_EXT]
    return (
        _GEO_TEMPLATE
        .replace("__SPEI_SADC__",   json.dumps([round(float(v), 4) for v in gs["spei_sadc"]]))
        .replace("__SPEI_SEA__",    json.dumps([round(float(v), 4) for v in gs["spei_sea"]]))
        .replace("__CHIRPS_SADC__", json.dumps([round(float(v), 4) for v in gs["chirps_sadc"]]))
        .replace("__CHIRPS_SEA__",  json.dumps([round(float(v), 4) for v in gs["chirps_sea"]]))
        .replace("__SST_PAC__",     json.dumps([round(float(v), 4) for v in gs["sst_pac"]]))
        .replace("__SST_IND__",     json.dumps([round(float(v), 4) for v in gs["sst_ind"]]))
        .replace("__DATES__",       json.dumps(d_labels))
        .replace("__TOKEN__",       token)
    )


def page_geo_scene():
    # header banner
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;
            background:rgba(2,8,18,.78);border:1px solid rgba(0,212,255,.22);
            border-radius:14px;padding:12px 20px;margin-bottom:10px;
            backdrop-filter:blur(10px);">
  <img src="{_LOGO_URI}"
       style="height:52px;border-radius:50%;
              border:1.5px solid rgba(0,212,255,.4);
              box-shadow:0 0 14px rgba(0,212,255,.3);">
  <div style="flex:1">
    <div style="font-size:1.15rem;font-weight:800;
                background:linear-gradient(90deg,#00d4ff,#00ffc8);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;letter-spacing:.5px;">
      CLIMATE-XFER &nbsp;·&nbsp; Live Geographical Scene
    </div>
    <div style="color:#7eb8d4;font-size:.78rem;margin-top:2px;">
      Globe rotates automatically · Topographic basemap · Hollow SPEI borders ·
      Rain density = CHIRPS · Press
      <strong style="color:#00d4ff">▶ Play</strong> to animate 2000–2026 (★ = simulated)
    </div>
  </div>
  <div style="font-size:.72rem;color:#7eb8d4;text-align:right;line-height:1.7">
    <span style="color:#00d4ff;font-weight:700">Tanaka Mbendana</span>
    &nbsp;·&nbsp; LS2525233<br>
    <span style="color:#00ffc8;font-weight:700">Fitrotur Rofiqoh</span>
    &nbsp;·&nbsp; LS2525220<br>
    <span style="color:#a855f7;font-weight:700">Munashe Mafuta</span>
    &nbsp;·&nbsp; LS2557204
  </div>
</div>
""", unsafe_allow_html=True)
    html_str = build_geo_html(MAPBOX_TOKEN)
    components.html(html_str, height=720, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def sidebar() -> str:
    with st.sidebar:
        st.markdown(f"""
<div style="text-align:center;padding:10px 0 16px">
  <img src="{_LOGO_URI}"
       style="height:64px;border-radius:50%;
              border:1.5px solid rgba(0,212,255,0.4);
              box-shadow:0 0 14px rgba(0,212,255,0.28);
              margin-bottom:10px;display:block;margin-left:auto;margin-right:auto;">
  <div style="font-size:1.7rem;font-weight:900;
              background:linear-gradient(135deg,#00d4ff,#00ffc8);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:-0.5px;line-height:1.1;">
    CLIMATE-XFER
  </div>
  <div style="color:#7eb8d4;font-size:.62rem;letter-spacing:2px;
              text-transform:uppercase;margin-top:4px;">
    Drought Intelligence
  </div>
</div>
<hr style="border-color:rgba(0,212,255,.18);margin-bottom:16px"/>
""", unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            options=[
                "🏠  Home & Authors",
                "🌍  Geo Scene",
                "🔮  Live Forecast",
                "📈  Validation Series",
                "🌊  Spatial Maps",
                "📊  Model Performance",
            ],
            label_visibility="collapsed",
        )

        st.markdown("""
<hr style="border-color:rgba(0,212,255,.12);margin:20px 0 14px"/>
<div style="font-size:.7rem;color:#7eb8d4;line-height:2">
  <div style="color:#00d4ff;font-weight:700;font-size:.75rem;margin-bottom:4px">Model</div>
  Architecture: Mean-GRU<br>
  Parameters: 11,777<br>
  History window: 12 months<br>
  Lead time: 1 month<br>
  <hr style="border-color:rgba(0,212,255,.1);margin:10px 0"/>
  <div style="color:#00d4ff;font-weight:700;font-size:.75rem;margin-bottom:4px">Data Sources</div>
  ERA5 · CHIRPS v2.0<br>
  NOAA ERSSTv5<br>
  Jan 2000 – Dec 2022<br>
  276 months · 1° grid<br>
  <hr style="border-color:rgba(0,212,255,.1);margin:10px 0"/>
  <div style="color:#00d4ff;font-weight:700;font-size:.75rem;margin-bottom:4px">Beihang University</div>
  MSc AI &amp; Large Models<br>
  Final Project · 2025
</div>
""", unsafe_allow_html=True)

    return page

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    page = sidebar()

    if "Home"        in page:
        page_hero()
    elif "Geo"       in page:
        page_geo_scene()
    elif "Forecast"  in page:
        page_forecast()
    elif "Validation" in page:
        page_timeseries()
    elif "Spatial"   in page:
        page_maps()
    elif "Performance" in page:
        page_performance()


if __name__ == "__main__":
    main()
