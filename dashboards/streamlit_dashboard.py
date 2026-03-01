"""
GV2-EDGE V9.0 — Dynamic & Responsive Trading Dashboard
=======================================================
Stack : Streamlit + Plotly + Custom CSS
Refresh : streamlit-autorefresh (non-bloquant)
Responsive : CSS grid + media queries (mobile / tablet / desktop)
"""

import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

# Ensure project root is on sys.path so 'src', 'utils', 'config' are importable
# regardless of the working directory when Streamlit is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="GV2-EDGE V9.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================
# CSS — Responsive + Dark Theme
# ============================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:   #080c14;
    --bg-secondary: #0f1623;
    --bg-card:      #141b28;
    --bg-hover:     #1e2638;
    --green:  #10b981; --red:    #ef4444;
    --yellow: #f59e0b; --blue:   #3b82f6;
    --purple: #8b5cf6; --cyan:   #06b6d4;
    --orange: #f97316;
    --text:   #f1f5f9; --muted:  #94a3b8; --dim: #64748b;
    --border: #1e2d45; --border-bright: #2d4060; --radius: 12px;
}

/* ── Base ── */
.stApp { background: radial-gradient(ellipse at 20% 0%,#0d1829 0%,#080c14 60%); font-family:'Outfit',sans-serif; }
.block-container { padding:1rem 2rem 3rem !important; max-width:1600px; }
#MainMenu, footer, header { visibility:hidden; }
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display:none !important; }

/* ── Typography ── */
h1,h2,h3 { font-family:'Outfit',sans-serif !important; font-weight:700 !important; color:var(--text) !important; }
h1 {
    font-size:clamp(1.5rem,3vw,2.4rem) !important; margin:0 !important;
    background:linear-gradient(90deg,#06b6d4 0%,#10b981 60%,#3b82f6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
h2 { font-size:clamp(1rem,2vw,1.3rem) !important; }
h3 { font-size:clamp(.9rem,1.5vw,1.1rem) !important; }

/* ── Cards ── */
.card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius); padding:1rem; margin:.3rem 0;
    transition:border-color .25s, box-shadow .25s, transform .15s;
    position:relative; overflow:hidden;
}
.card::before {
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg,rgba(6,182,212,.03),transparent 60%);
    pointer-events:none;
}
.card:hover { border-color:var(--border-bright); box-shadow:0 4px 24px rgba(6,182,212,.1); transform:translateY(-1px); }
.card-buy-strong {
    background:linear-gradient(135deg,rgba(16,185,129,.1),rgba(6,182,212,.05));
    border-left:3px solid var(--green); border-color:rgba(16,185,129,.35);
}
.card-buy   { background:linear-gradient(135deg,rgba(59,130,246,.1),rgba(139,92,246,.05)); border-left:3px solid var(--blue); border-color:rgba(59,130,246,.3); }
.card-watch { background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(239,68,68,.04)); border-left:3px solid var(--yellow); }
.card-early { background:linear-gradient(135deg,rgba(139,92,246,.1),rgba(245,158,11,.05)); border-left:3px solid var(--purple); }
.card-alert { border-color:var(--green) !important; animation:alertPulse 2s infinite; }
@keyframes alertPulse {
    0%,100% { box-shadow:0 0 12px rgba(16,185,129,.25); }
    50%      { box-shadow:0 0 32px rgba(16,185,129,.55), 0 0 64px rgba(16,185,129,.15); }
}

/* ── KPI Grid ── */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(115px,1fr)); gap:.55rem; margin-bottom:.9rem; }
.kpi {
    text-align:center; padding:.9rem .4rem .75rem;
    background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
    transition:border-color .2s, box-shadow .2s; position:relative; overflow:hidden;
}
.kpi::after { content:''; position:absolute; inset:0; background:linear-gradient(180deg,rgba(255,255,255,.02),transparent); pointer-events:none; }
.kpi:hover  { border-color:var(--border-bright); box-shadow:0 2px 16px rgba(6,182,212,.08); }
.kpi-value  { font-family:'JetBrains Mono',monospace; font-size:clamp(1.2rem,2.3vw,1.75rem); font-weight:700; color:var(--text); line-height:1.1; }
.kpi-label  { font-size:.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-top:.25rem; }
.kpi-delta  { font-family:'JetBrains Mono',monospace; font-size:.72rem; margin-top:.2rem; }
.kpi-sub    { font-size:.62rem; color:var(--dim); margin-top:.08rem; }

/* ── IBKR status badge (always visible) ── */
.ibkr-badge {
    display:inline-flex; align-items:center; gap:.4rem;
    padding:.35rem .75rem; border-radius:20px; font-size:.78rem; font-weight:600;
    font-family:'Outfit',sans-serif; border:1px solid;
    transition:all .3s;
}
.ibkr-connected {
    background:rgba(16,185,129,.12); color:var(--green);
    border-color:rgba(16,185,129,.4); box-shadow:0 0 12px rgba(16,185,129,.15);
}
.ibkr-disconnected {
    background:rgba(239,68,68,.12); color:var(--red);
    border-color:rgba(239,68,68,.4); box-shadow:0 0 12px rgba(239,68,68,.1);
}
.ibkr-reconnecting {
    background:rgba(245,158,11,.12); color:var(--yellow);
    border-color:rgba(245,158,11,.4); animation:reconnPulse 1.5s infinite;
}
@keyframes reconnPulse { 0%,100%{opacity:1} 50%{opacity:.6} }

/* ── Status bar (API health strip) ── */
.status-bar {
    display:flex; flex-wrap:wrap; align-items:center; gap:.5rem;
    background:var(--bg-secondary); border:1px solid var(--border);
    border-radius:10px; padding:.45rem .8rem; margin-bottom:.6rem;
}
.status-chip {
    display:inline-flex; align-items:center; gap:.3rem;
    padding:.18rem .5rem; border-radius:6px; font-size:.68rem; font-weight:600;
    font-family:'JetBrains Mono',monospace;
}
.chip-ok  { background:rgba(16,185,129,.12); color:var(--green); }
.chip-err { background:rgba(239,68,68,.12);  color:var(--red); }
.chip-warn{ background:rgba(245,158,11,.12); color:var(--yellow); }
.chip-dim { background:rgba(100,116,139,.12); color:var(--muted); }

/* ── Signal grid ── */
.sig-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:.6rem; }

/* ── Gap grid ── */
.gap-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(95px,1fr)); gap:.4rem; }
.gap-card { padding:.5rem; border-radius:8px; text-align:center; background:var(--bg-card); border:1px solid var(--border); }
.gap-up   { border-left:3px solid var(--green); }
.gap-down { border-left:3px solid var(--red); }

/* ── Badges ── */
.badge { display:inline-block; padding:.15rem .5rem; border-radius:5px; font-size:.68rem; font-weight:700; font-family:'JetBrains Mono',monospace; letter-spacing:.03em; }
.badge-strong { background:rgba(16,185,129,.18); color:var(--green); border:1px solid rgba(16,185,129,.3); }
.badge-buy    { background:rgba(59,130,246,.18);  color:#60a5fa;     border:1px solid rgba(59,130,246,.3); }
.badge-watch  { background:rgba(245,158,11,.18);  color:var(--yellow);border:1px solid rgba(245,158,11,.3); }
.badge-early  { background:rgba(139,92,246,.18);  color:#a78bfa;     border:1px solid rgba(139,92,246,.3); }

/* ── Misc ── */
.tick { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1rem; color:var(--cyan); letter-spacing:.02em; }
.live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); animation:ldot 1.8s infinite; margin-right:5px; vertical-align:middle; }
@keyframes ldot { 0%,100%{opacity:1;box-shadow:0 0 6px var(--green)} 50%{opacity:.3;box-shadow:none} }

/* ── Log viewer ── */
.log-container {
    background:#060a10; border:1px solid var(--border); border-radius:10px;
    padding:1rem; font-family:'JetBrains Mono',monospace; font-size:.7rem;
    height:500px; overflow-y:auto; white-space:pre-wrap; word-break:break-all;
    line-height:1.6;
}
.log-error { color:#f87171; } .log-warn { color:#fbbf24; }
.log-info  { color:#94a3b8; } .log-ok   { color:#34d399; }

/* ── API monitor ── */
.api-ok   { color:var(--green); } .api-warn { color:var(--yellow); } .api-err { color:var(--red); }
.api-row  { font-family:'JetBrains Mono',monospace; font-size:.68rem; border-bottom:1px solid #111c2e; padding:.2rem 0; }

/* ── Pills ── */
.pill { display:inline-block; padding:.22rem .6rem; border-radius:20px; font-size:.7rem; font-weight:600; margin:.1rem; }
.pill-ok  { background:rgba(16,185,129,.15); color:var(--green); border:1px solid rgba(16,185,129,.4); }
.pill-err { background:rgba(239,68,68,.15);  color:var(--red);   border:1px solid rgba(239,68,68,.4); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap:2px; background:var(--bg-secondary);
    padding:.3rem; border-radius:var(--radius);
    border:1px solid var(--border); flex-wrap:wrap;
}
.stTabs [data-baseweb="tab"] {
    background:transparent; border-radius:8px; color:var(--muted);
    font-weight:500; font-size:clamp(.7rem,.82vw,.84rem); padding:.32rem .58rem;
    transition:all .2s;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,rgba(6,182,212,.15),rgba(16,185,129,.08));
    color:var(--text); box-shadow:0 1px 8px rgba(6,182,212,.12);
}

/* ── Divider ── */
hr { border-color:var(--border) !important; margin:.6rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border-bright); border-radius:4px; }

/* ── Color helpers ── */
.green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--yellow)}
.cyan{color:var(--cyan)}   .muted{color:var(--muted)}
.score-high{color:var(--green)} .score-med{color:var(--yellow)} .score-low{color:var(--red)}

/* ════════════════════════════════════════
   RESPONSIVE
   ════════════════════════════════════════ */
@media (min-width:641px) and (max-width:1024px) {
    .block-container { padding:.8rem .8rem 2rem !important; }
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
    [data-testid="column"] { min-width:calc(50% - .5rem) !important; }
    .kpi-grid { grid-template-columns:repeat(3,1fr) !important; }
    .sig-grid { grid-template-columns:repeat(2,1fr) !important; }
    .gap-grid { grid-template-columns:repeat(3,1fr) !important; }
}

@media (max-width:640px) {
    .block-container { padding:.3rem .4rem 2rem !important; }
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.2rem !important; }
    [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:0 !important; }
    h1 { font-size:1.1rem !important; }
    h2 { font-size:.9rem !important; }
    h3 { font-size:.82rem !important; }
    .kpi-grid { grid-template-columns:repeat(2,1fr) !important; gap:.3rem !important; }
    .kpi { padding:.5rem .3rem !important; }
    .kpi-value { font-size:1rem !important; }
    .kpi-label { font-size:.56rem !important; }
    .kpi-sub,.kpi-delta { font-size:.56rem !important; }
    .sig-grid { grid-template-columns:1fr !important; }
    .gap-grid { grid-template-columns:repeat(2,1fr) !important; }
    .card { padding:.6rem !important; }
    .stTabs [data-baseweb="tab-list"] { overflow-x:auto !important; flex-wrap:nowrap !important; }
    .stTabs [data-baseweb="tab"] { font-size:.6rem !important; padding:.18rem .35rem !important; white-space:nowrap !important; }
    .log-container { height:280px !important; font-size:.6rem !important; }
    [data-testid="stDataFrame"] > div { overflow-x:auto !important; }
    .pill { font-size:.58rem !important; padding:.12rem .3rem !important; }
    .desktop-only { display:none !important; }
    .status-bar { gap:.3rem; padding:.3rem .5rem; }
    .status-chip { font-size:.6rem !important; }
    .ibkr-badge { font-size:.68rem !important; padding:.25rem .5rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ============================
# PATHS
# ============================

DATA_DIR   = _PROJECT_ROOT / "data"   # absolute — Streamlit changes CWD to script dir
SIGNALS_DB = DATA_DIR / "signals_history.db"
LOGS_DIR   = DATA_DIR / "logs"
AUDIT_DIR  = DATA_DIR / "audit_reports"


# ============================
# CACHED DATA LOADERS
# ============================

@st.cache_data(ttl=30)
def load_signals(hours_back: int = 24) -> pd.DataFrame:
    if not SIGNALS_DB.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(SIGNALS_DB), check_same_thread=False)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp DESC",
            conn, params=(cutoff,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_universe_size() -> int:
    for uf in (DATA_DIR/"universe.csv", DATA_DIR/"universe_v3.csv"):
        if uf.exists():
            try:
                return len(pd.read_csv(uf))
            except Exception:
                pass
    return 0


@st.cache_data(ttl=300)
def load_ticker_profiles() -> pd.DataFrame:
    """Load all ticker profiles from SQLite."""
    db = DATA_DIR / "ticker_profiles.db"
    if not db.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db), check_same_thread=False)
        df = pd.read_sql_query("SELECT * FROM ticker_profiles ORDER BY data_quality DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_universe_df() -> pd.DataFrame:
    """Load full universe CSV (ticker, exchange, name)."""
    for uf in (DATA_DIR/"universe.csv", DATA_DIR/"universe_v3.csv"):
        if uf.exists():
            try:
                df = pd.read_csv(uf)
                # Normalise column names
                df.columns = [c.strip().lower() for c in df.columns]
                if "ticker" not in df.columns:
                    return pd.DataFrame()
                df["ticker"]   = df["ticker"].astype(str).str.upper().str.strip()
                df["exchange"] = df.get("exchange", pd.Series([""] * len(df))).fillna("").astype(str).str.upper()
                df["name"]     = df.get("name",     pd.Series([""] * len(df))).fillna("").astype(str)
                return df[["ticker","exchange","name"]].sort_values("ticker").reset_index(drop=True)
            except Exception:
                pass
    return pd.DataFrame(columns=["ticker","exchange","name"])


@st.cache_data(ttl=60)
def load_latest_audit() -> dict | None:
    if not AUDIT_DIR.exists():
        return None
    files = list(AUDIT_DIR.glob("*.json"))
    if not files:
        return None
    try:
        with open(sorted(files)[-1]) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=15)
def load_hot_queue() -> list:
    db = DATA_DIR / "hot_ticker_queue.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db), check_same_thread=False)
        rows = conn.execute(
            "SELECT ticker,priority,score,added_at FROM queue ORDER BY score DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return [{"ticker":r[0],"priority":r[1],"score":r[2],"added_at":r[3]} for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=20)
def load_api_monitor(n: int = 200) -> list[str]:
    path = LOGS_DIR / "api_monitor.log"
    if not path.exists():
        return []
    try:
        with open(path,"r",encoding="utf-8",errors="replace") as f:
            return list(deque(f, maxlen=n))
    except Exception:
        return []


@st.cache_data(ttl=10)
def load_log_tail(log_file: str, n: int = 100) -> list[str]:
    path = LOGS_DIR / log_file
    if not path.exists():
        return []
    try:
        with open(path,"r",encoding="utf-8",errors="replace") as f:
            return list(deque(f, maxlen=n))
    except Exception:
        return []


@st.cache_data(ttl=30)
def load_extended_gaps() -> list:
    for fname in ("extended_hours_cache.json","ah_scan.json","pm_scan.json"):
        try:
            with open(DATA_DIR/fname) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [dict(v, ticker=t) for t,v in data.items() if isinstance(v,dict)]
        except Exception:
            pass
    return []


@st.cache_data(ttl=30)
def load_events_cache() -> list:
    """Load events from file cache. Always returns a list (handles {} or [] or missing file)."""
    try:
        with open(DATA_DIR/"events_cache.json") as f:
            data = json.load(f)
        # File might be {} (dict) from a reset — convert to list of values or return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and data:
            return list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
        return []
    except Exception:
        return []


@st.cache_data(ttl=120)
def get_system_status() -> dict:
    status = {"ibkr":False,"grok":False,"finnhub":False,"telegram":False}
    try:
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        status["ibkr"] = bool(ibkr and ibkr.connected)
    except Exception:
        pass
    try:
        from config import GROK_API_KEY, FINNHUB_API_KEY
        status["grok"]    = bool(GROK_API_KEY and not GROK_API_KEY.startswith("YOUR_"))
        status["finnhub"] = bool(FINNHUB_API_KEY and not FINNHUB_API_KEY.startswith("YOUR_"))
    except Exception:
        pass
    try:
        from config import TELEGRAM_SIGNALS_TOKEN
        status["telegram"] = bool(TELEGRAM_SIGNALS_TOKEN and not TELEGRAM_SIGNALS_TOKEN.startswith("YOUR_"))
    except Exception:
        try:
            from config import TELEGRAM_BOT_TOKEN
            status["telegram"] = bool(TELEGRAM_BOT_TOKEN and not TELEGRAM_BOT_TOKEN.startswith("YOUR_"))
        except Exception:
            pass
    return status


@st.cache_data(ttl=60)
def get_ibkr_info() -> dict | None:
    try:
        from config import USE_IBKR_DATA
        if not USE_IBKR_DATA:
            return None
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        return ibkr.get_connection_stats() if ibkr else None
    except Exception:
        return None


@st.cache_data(ttl=15)
def get_ibkr_status() -> dict:
    """
    Read IBKR status from data/ibkr_status.json written by system_guardian (main process).
    Streamlit and main.py run in separate processes — no shared memory.
    """
    result = {"connected": False, "state": "UNKNOWN", "uptime": None, "latency": None}
    status_file = DATA_DIR / "ibkr_status.json"
    try:
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
            result["connected"] = bool(data.get("connected", False))
            result["state"]     = data.get("state", "UNKNOWN")
            result["uptime"]    = data.get("uptime")
            result["latency"]   = data.get("latency")
            # Stale check: if file not updated in >3 min, mark as stale
            updated = data.get("updated_at", "")
            if updated:
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()
                    if age > 180:
                        result["state"] = f"STALE ({int(age//60)}m)"
                        result["connected"] = False
                except Exception:
                    pass
        else:
            result["state"] = "NO DATA"
    except Exception:
        result["state"] = "ERROR"
    return result


@st.cache_data(ttl=300)
def load_watch_list() -> list:
    """Load watch list — no dynamic args so @st.cache_data works properly (runs once per TTL)."""
    import concurrent.futures
    try:
        from src.watch_list import generate_watch_list
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(generate_watch_list, days_forward=30, min_impact=0.1)
            return fut.result(timeout=8) or []
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_radar_signals(limit: int = 200) -> list:
    """Load recent signals with multi_radar_result from DB."""
    try:
        conn = sqlite3.connect(str(SIGNALS_DB), check_same_thread=False)
        rows = conn.execute(
            "SELECT ticker, signal_type, monster_score, timestamp, metadata "
            "FROM signals WHERE metadata IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        results = []
        for ticker, sig, score, ts, meta_str in rows:
            meta = json.loads(meta_str) if meta_str else {}
            radar = meta.get("multi_radar_result")
            results.append({
                "ticker": ticker, "signal_type": sig,
                "score": score or 0, "timestamp": ts,
                "radar": radar,
            })
        return results
    except Exception:
        return []


# ============================
# HELPERS
# ============================

def get_session() -> str:
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now(timezone.utc) - timedelta(hours=4)
    if now_et.weekday() >= 5:
        return "CLOSED"
    t = now_et.hour*60 + now_et.minute
    if   4*60 <= t <  9*60+30: return "PREMARKET"
    elif 9*60+30 <= t < 16*60: return "RTH"
    elif 16*60 <= t < 20*60:   return "AFTER_HOURS"
    else:                       return "CLOSED"


def fmt_time(ts: str) -> str:
    try: return ts[11:16]
    except Exception: return ""


def score_color(v: float) -> str:
    return "#10b981" if v >= 0.65 else "#f59e0b" if v >= 0.50 else "#ef4444"


def signal_badge(stype: str) -> str:
    m = {"BUY_STRONG":("badge-strong","🚨 BUY_STRONG"),"BUY":("badge-buy","✅ BUY"),
         "WATCH":("badge-watch","👀 WATCH"),"EARLY_SIGNAL":("badge-early","⚡ EARLY")}
    cls, lbl = m.get(stype, ("badge-watch", stype))
    return f'<span class="badge {cls}">{lbl}</span>'


def load_monster_components(row) -> dict:
    return {
        "Event":    min(1.0, float(row.get("event_impact",0) or 0)),
        "Volume":   min(1.0, float(row.get("volume_spike",0) or 0)),
        "Pattern":  min(1.0, float(row.get("pattern_score",0) or 0)),
        "PM Trans": min(1.0, float(row.get("pm_transition_score",0) or 0)),
        "Momentum": min(1.0, float(row.get("momentum",0) or 0)),
        "Options":  0.0, "Accel": 0.0, "Social": 0.0, "Squeeze": 0.0,
    }


# ============================
# CHARTS
# ============================

_BASE = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1a1f2e",
             font=dict(color="#9ca3af"),margin=dict(l=40,r=20,t=30,b=30))


def chart_score_radar(comp: dict):
    labels = list(comp.keys()); values = [comp[k] for k in labels]
    labels.append(labels[0]); values.append(values[0])
    fig = go.Figure(go.Scatterpolar(r=values,theta=labels,fill="toself",
        fillcolor="rgba(6,182,212,.25)",line=dict(color="#06b6d4",width=2)))
    fig.update_layout(**_BASE,polar=dict(
        radialaxis=dict(visible=True,range=[0,1],tickfont=dict(color="#6b7280",size=9),gridcolor="#374151"),
        angularaxis=dict(tickfont=dict(color="#9ca3af",size=10),gridcolor="#374151"),bgcolor="#1a1f2e"),
        showlegend=False,height=260)
    return fig


def chart_timeline(df: pd.DataFrame):
    cmap={"BUY_STRONG":"#10b981","BUY":"#3b82f6","EARLY_SIGNAL":"#8b5cf6","WATCH":"#f59e0b"}
    fig=px.scatter(df,x="timestamp",y="monster_score",color="signal_type",size="monster_score",
        hover_data=["ticker","signal_type","monster_score"],color_discrete_map=cmap)
    fig.update_layout(**_BASE,xaxis=dict(gridcolor="#374151",tickfont=dict(color="#9ca3af")),
        yaxis=dict(gridcolor="#374151",tickfont=dict(color="#9ca3af"),title="Score"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),height=240)
    return fig


def chart_hit_rate_gauge(hr: float):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=hr*100,
        number={"suffix":"%","font":{"color":"#f9fafb","size":32}},
        gauge={"axis":{"range":[0,100],"tickfont":{"color":"#6b7280"}},"bar":{"color":"#06b6d4"},
               "bgcolor":"#1a1f2e","bordercolor":"#374151",
               "steps":[{"range":[0,40],"color":"rgba(239,68,68,.25)"},
                        {"range":[40,65],"color":"rgba(245,158,11,.25)"},
                        {"range":[65,100],"color":"rgba(16,185,129,.25)"}],
               "threshold":{"line":{"color":"#10b981","width":3},"thickness":.75,"value":65}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":"#f9fafb"},height=210,margin=dict(l=20,r=20,t=30,b=10))
    return fig


def chart_score_dist(df: pd.DataFrame):
    fig=px.histogram(df,x="monster_score",nbins=20,color_discrete_sequence=["#06b6d4"])
    fig.update_layout(**_BASE,xaxis=dict(gridcolor="#374151",title="Score"),
        yaxis=dict(gridcolor="#374151",title="Count"),height=240)
    return fig


def chart_type_bar(df: pd.DataFrame):
    cmap={"BUY_STRONG":"#10b981","BUY":"#3b82f6","WATCH":"#f59e0b","EARLY_SIGNAL":"#8b5cf6"}
    tc=df["signal_type"].value_counts().reset_index(); tc.columns=["Signal","Count"]
    fig=px.bar(tc,x="Signal",y="Count",color="Signal",color_discrete_map=cmap,text="Count")
    fig.update_layout(**_BASE,showlegend=False,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),yaxis=dict(gridcolor="#374151"),height=240)
    fig.update_traces(textposition="outside")
    return fig


def chart_radar_bars(radars: dict):
    names=list(radars.keys()); scores=[radars[n].get("score",0) for n in names]
    colors=["#10b981","#3b82f6","#8b5cf6","#f59e0b"]
    fig=go.Figure(go.Bar(x=names,y=scores,marker_color=colors[:len(names)],
        text=[f"{s:.2f}" for s in scores],textposition="outside"))
    fig.update_layout(**_BASE,yaxis=dict(range=[0,1],gridcolor="#374151"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),height=240)
    return fig


def chart_events_pie(ec: pd.Series):
    fig=px.pie(values=ec.values,names=ec.index,hole=.35,
        color_discrete_sequence=["#06b6d4","#3b82f6","#8b5cf6","#f59e0b","#10b981","#ef4444"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#9ca3af"),
        height=240,margin=dict(l=10,r=10,t=20,b=10))
    fig.update_traces(textfont=dict(color="white"))
    return fig


def chart_api_latency(lines: list[str]):
    records = []
    for l in lines:
        try:
            parts=[p.strip() for p in l.split("|")]
            if len(parts) < 8: continue
            prov=parts[4]; lat=float(parts[7].replace("ms","").strip())
            records.append({"provider":prov,"latency":lat})
        except Exception:
            continue
    if not records: return None
    df=pd.DataFrame(records)
    agg=df.groupby("provider")["latency"].mean().reset_index()
    agg.columns=["Provider","Avg ms"]
    fig=px.bar(agg,x="Provider",y="Avg ms",color="Avg ms",
        color_continuous_scale=[[0,"#10b981"],[.5,"#f59e0b"],[1,"#ef4444"]])
    fig.update_layout(**_BASE,height=220,coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),yaxis=dict(gridcolor="#374151"))
    return fig


# ============================
# TOP CONTROLS BAR
# ============================

session = get_session()

# Row 1 — Filters + actions (always visible)
cb1,cb2,cb3,cb4,cb5,cb6,cb7 = st.columns([1,1,2,3,1.5,1,1])
with cb1:
    auto_refresh = st.toggle("🔄 Auto", value=True, help="Auto refresh")
with cb2:
    refresh_sec = st.selectbox("⏱", [10,30,60,120], index=1,
        format_func=lambda x: f"{x}s", label_visibility="collapsed") if auto_refresh else 30
with cb3:
    hours_back = st.selectbox("🕐 Période",
        [6,12,24,48,168], index=2,
        format_func=lambda x: f"Last {x}h" if x<168 else "Last 7d",
        label_visibility="visible")
with cb4:
    signal_filter = st.multiselect("📡 Signaux",
        ["BUY_STRONG","BUY","WATCH","EARLY_SIGNAL"],
        default=["BUY_STRONG","BUY","EARLY_SIGNAL"],
        label_visibility="visible")
with cb5:
    min_score = st.slider("📊 Score min", 0.0, 1.0, 0.40, 0.05, label_visibility="visible")
with cb6:
    if st.button("📊 Audit", use_container_width=True):
        with st.spinner("Running..."):
            try:
                from daily_audit import run_daily_audit
                run_daily_audit(send_telegram=False); st.success("Done!")
            except Exception as e: st.error(str(e))
with cb7:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

# ── Status bar — API health chips (compact, always visible) ──
ibkr_st    = get_ibkr_status()
sys_status = get_system_status()

_ibkr_cls  = "chip-ok" if ibkr_st["connected"] else ("chip-warn" if ibkr_st["state"] in ("RECONNECTING","CONNECTING") else "chip-err")
_ibkr_icon = "🟢" if ibkr_st["connected"] else ("🟡" if ibkr_st["state"] in ("RECONNECTING","CONNECTING") else "🔴")
_ibkr_label = f"IBKR {ibkr_st['state']}"
if ibkr_st["connected"] and ibkr_st.get("uptime"):
    _ibkr_label += f" ↑{ibkr_st['uptime']}"
if ibkr_st.get("latency"):
    _ibkr_label += f" {ibkr_st['latency']}"

chips = f'<span class="status-chip {_ibkr_cls}">{_ibkr_icon} {_ibkr_label}</span>'
for k, ok in sys_status.items():
    if k == "ibkr": continue
    chips += f'<span class="status-chip {"chip-ok" if ok else "chip-err"}">{"🟢" if ok else "🔴"} {k.upper()}</span>'

try:
    from config import (ENABLE_MULTI_RADAR, ENABLE_ACCELERATION_ENGINE, ENABLE_SMALLCAP_RADAR,
                        ENABLE_TICKER_PROFILES)
    chips += f'<span class="status-chip {"chip-ok" if ENABLE_MULTI_RADAR else "chip-dim"}">{"●" if ENABLE_MULTI_RADAR else "○"} Multi-Radar</span>'
    chips += f'<span class="status-chip {"chip-ok" if ENABLE_ACCELERATION_ENGINE else "chip-dim"}">{"●" if ENABLE_ACCELERATION_ENGINE else "○"} Accel V8</span>'
    chips += f'<span class="status-chip {"chip-ok" if ENABLE_TICKER_PROFILES else "chip-dim"}">{"●" if ENABLE_TICKER_PROFILES else "○"} Profiles</span>'
except Exception:
    pass

st.markdown(f'<div class="status-bar">{chips}</div>', unsafe_allow_html=True)


# ============================
# AUTO-REFRESH
# ============================

if auto_refresh:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_sec*1000, key="main_refresh")
    else:
        st.markdown(f'<meta http-equiv="refresh" content="{refresh_sec}">',unsafe_allow_html=True)


# ============================
# HEADER
# ============================

try:
    from zoneinfo import ZoneInfo
    now_et=datetime.now(ZoneInfo("America/New_York"))
except Exception:
    now_et=datetime.now(timezone.utc)-timedelta(hours=4)

sess_map={"PREMARKET":("🌅","Pre-Market","#f59e0b"),"RTH":("📈","RTH","#10b981"),
          "AFTER_HOURS":("🌙","After-Hours","#8b5cf6"),"CLOSED":("💤","Closed","#64748b")}
sess_icon,sess_label,sess_color=sess_map.get(session,("❓",session,"#64748b"))

# IBKR badge HTML
_ibkr_badge_cls = "ibkr-connected" if ibkr_st["connected"] else (
    "ibkr-reconnecting" if ibkr_st["state"] in ("RECONNECTING","CONNECTING") else "ibkr-disconnected")
_ibkr_dot = "🟢" if ibkr_st["connected"] else ("🟡" if "CONNECT" in ibkr_st["state"] else "🔴")
_ibkr_txt = f"IBKR {'CONNECTED' if ibkr_st['connected'] else ibkr_st['state']}"
_ibkr_detail = ""
if ibkr_st["connected"]:
    if ibkr_st.get("uptime"):   _ibkr_detail += f" · ↑{ibkr_st['uptime']}"
    if ibkr_st.get("latency"):  _ibkr_detail += f" · {ibkr_st['latency']}"

hcol1, hcol2, hcol3 = st.columns([4, 3, 2])
with hcol1:
    st.markdown("# 🎯 GV2-EDGE V9.0")
    st.markdown(
        f'<span style="color:#64748b;font-size:.82rem;">Multi-Radar V9 · 3-Layer Pipeline · Small Caps US</span>',
        unsafe_allow_html=True)
with hcol2:
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;padding-top:.6rem;">
            <span class="ibkr-badge {_ibkr_badge_cls}">{_ibkr_dot} {_ibkr_txt}<span style="opacity:.7;font-size:.7rem;">{_ibkr_detail}</span></span>
            <span style="background:rgba(255,255,255,.05);border:1px solid #1e2d45;border-radius:20px;
                         padding:.3rem .7rem;font-size:.78rem;font-weight:600;color:{sess_color};">
                {sess_icon} {sess_label}</span>
        </div>""", unsafe_allow_html=True)
with hcol3:
    st.markdown(
        f"""<div style="text-align:right;padding-top:.5rem;">
            <div style="font-size:.72rem;color:#64748b;margin-bottom:.1rem;">
                <span class="live-dot"></span><span style="color:#10b981;font-weight:600;">LIVE</span>
                &nbsp;·&nbsp;↻{refresh_sec}s
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;font-weight:700;color:#f1f5f9;">
                {now_et.strftime("%H:%M:%S")}
            </div>
            <div style="font-size:.65rem;color:#64748b;">{now_et.strftime("%a %Y-%m-%d")} ET</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:.4rem 0;border-bottom:1px solid #1e2d45;'></div>", unsafe_allow_html=True)


# ============================
# LOAD DATA
# ============================

signals_df  = load_signals(hours_back)
signals_1h  = load_signals(1)
audit_data  = load_latest_audit()
gaps_data   = load_extended_gaps()
universe_sz = load_universe_size()

total_sig  = len(signals_df)
buy_strong = int((signals_df["signal_type"]=="BUY_STRONG").sum()) if not signals_df.empty else 0
hit_rate   = float(audit_data.get("hit_rate",0)) if audit_data else 0
avg_lead   = float(audit_data.get("avg_lead_time_hours",0)) if audit_data else 0
total_1h   = len(signals_1h)
delta_sig  = total_sig - total_1h
strong_1h  = int((signals_1h["signal_type"]=="BUY_STRONG").sum()) if not signals_1h.empty else 0


# ============================
# KPI ROW  (responsive CSS grid)
# ============================

def kpi_html(value, label, sub="", delta="", delta_cls="muted"):
    return f"""<div class="kpi">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-sub">{sub}</div>
        <div class="kpi-delta {delta_cls}">{delta}</div>
    </div>"""

score_cls  = "green" if hit_rate>0.65 else "yellow" if hit_rate>0.40 else "red"
strong_cls = "green" if buy_strong>0 else "muted"
delta_str  = f"+{delta_sig}" if delta_sig>=0 else str(delta_sig)

# IBKR KPI — state + uptime
_ibkr_kpi_val = "🟢 ON" if ibkr_st["connected"] else ("🟡 …" if "CONNECT" in ibkr_st["state"] else "🔴 OFF")
_ibkr_kpi_sub = ibkr_st.get("uptime") or ibkr_st["state"]
_ibkr_kpi_cls = "green" if ibkr_st["connected"] else ("yellow" if "CONNECT" in ibkr_st["state"] else "red")

kpis = (
    kpi_html(total_sig,             "Signals",     f"Last {hours_back}h", delta_str,  "green" if delta_sig>0 else "muted"),
    kpi_html(buy_strong,            "BUY_STRONG",  "Hot picks",           f"+{strong_1h}/1h", strong_cls),
    kpi_html(f"{hit_rate*100:.1f}%","Hit Rate",    "Target ≥65%",         "",          score_cls),
    kpi_html(f"{avg_lead:.1f}h",    "Lead Time",   "Before spike",        "",          "cyan"),
    kpi_html(f"{universe_sz:,}",    "Universe",    "Tickers tracked",     "",          "cyan"),
    kpi_html(_ibkr_kpi_val,         "IBKR",        _ibkr_kpi_sub,         ibkr_st.get("latency",""), _ibkr_kpi_cls),
    kpi_html(session,               "Session",     now_et.strftime("%H:%M"), "",       "yellow" if session!="CLOSED" else "muted"),
)
st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)
st.markdown("<div style='margin:.2rem 0'></div>", unsafe_allow_html=True)


# ============================
# TABS
# ============================

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
    "📡 Live Signals", "📊 Analytics", "📅 Events",
    "🛰️ Multi-Radar V9", "🌐 API Monitor", "📋 Live Logs", "🔍 Audit",
    "🌍 Universe", "🗂️ Profiles",
])


# ─────────────────────────────
# TAB 1 — LIVE SIGNALS
# ─────────────────────────────

with tab1:
    # Hot queue
    hot = load_hot_queue()
    if hot:
        st.markdown("#### 🔥 Hot Ticker Queue")
        hcols = st.columns(min(len(hot),8))
        for i,h in enumerate(hot[:8]):
            pri=h["priority"]
            c={"HOT":"#10b981","WARM":"#f59e0b"}.get(pri,"#6b7280")
            hcols[i].markdown(f"""<div class="card" style="text-align:center;padding:.45rem;border-left:3px solid {c};">
                <div class="tick" style="font-size:.85rem;">{h['ticker']}</div>
                <div style="font-size:.62rem;color:{c};">{pri}</div>
                <div style="font-size:.62rem;color:#6b7280;">{h.get('score',0):.2f}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown("---")

    # Extended Hours Movers
    if gaps_data:
        sorted_gaps=sorted(gaps_data,key=lambda x:abs(x.get("gap_pct",x.get("gap",0))),reverse=True)
        gap_cards="".join([f"""<div class="gap-card {'gap-up' if (g.get('gap_pct',g.get('gap',0)))>0 else 'gap-down'}">
            <div class="tick" style="font-size:.82rem;">{g.get('ticker','?')}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:{'#10b981' if (g.get('gap_pct',g.get('gap',0)))>0 else '#ef4444'};">
                {'+'if g.get('gap_pct',g.get('gap',0))>0 else ''}{g.get('gap_pct',g.get('gap',0)):.1f}%
            </div>
            <div style="font-size:.62rem;color:#6b7280;">${g.get('price',g.get('last',0)):.2f}</div>
        </div>""" for g in sorted_gaps[:8]])
        st.markdown("#### 📈 Extended Hours Movers")
        st.markdown(f'<div class="gap-grid">{gap_cards}</div>',unsafe_allow_html=True)
        st.markdown("---")

    # Signal cards
    st.markdown("### 🔥 Active Signals")
    if signals_df.empty:
        st.info("No signals in range — engine scanning…")
    else:
        fdf=signals_df.copy()
        if signal_filter: fdf=fdf[fdf["signal_type"].isin(signal_filter)]
        fdf=fdf[fdf["monster_score"]>=min_score]
        if fdf.empty:
            st.warning("No signals match filters.")
        else:
            type_order=["BUY_STRONG","BUY","WATCH","EARLY_SIGNAL"]
            fdf["_o"]=fdf["signal_type"].map({t:i for i,t in enumerate(type_order)}).fillna(99)
            fdf=fdf.sort_values(["_o","monster_score"],ascending=[True,False])
            cls_map={"BUY_STRONG":"card-buy-strong","BUY":"card-buy","WATCH":"card-watch","EARLY_SIGNAL":"card-early"}

            def sig_card(row, cls):
                ticker=row.get("ticker","?"); stype=row.get("signal_type","WATCH")
                score=float(row.get("monster_score",0) or 0); ts=fmt_time(str(row.get("timestamp","")))
                entry=row.get("entry_price"); stop=row.get("stop_loss"); shares=row.get("shares")
                ev=float(row.get("event_impact",0) or 0); vol=float(row.get("volume_spike",0) or 0)
                alert="card-alert" if stype=="BUY_STRONG" else ""
                return f"""<div class="card {cls} {alert}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="tick">{ticker}</span>{signal_badge(stype)}
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.25rem;margin-top:.55rem;text-align:center;">
                        <div><div style="color:#9ca3af;font-size:.62rem;">SCORE</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.9rem;color:{score_color(score)};">{score:.2f}</div></div>
                        <div><div style="color:#9ca3af;font-size:.62rem;">ENTRY</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:.88rem;color:#f9fafb;">${f'{entry:.2f}' if entry else '—'}</div></div>
                        <div><div style="color:#9ca3af;font-size:.62rem;">STOP</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:.88rem;color:#ef4444;">${f'{stop:.2f}' if stop else '—'}</div></div>
                        <div><div style="color:#9ca3af;font-size:.62rem;">QTY</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:.88rem;color:#f9fafb;">{int(shares) if shares else '—'}</div></div>
                    </div>
                    <div style="display:flex;gap:.35rem;margin-top:.4rem;font-size:.65rem;color:#9ca3af;">
                        <span>Ev:{ev:.2f}</span><span>Vol:{vol:.2f}</span>
                        <span style="margin-left:auto;">{ts}</span>
                    </div>
                </div>"""

            cards_html="".join([sig_card(row,cls_map.get(row["signal_type"],"card-watch")) for _,row in fdf.head(24).iterrows()])
            st.markdown(f'<div class="sig-grid">{cards_html}</div>',unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)

            if "timestamp" in fdf.columns:
                st.markdown("#### 📈 Timeline")
                st.plotly_chart(chart_timeline(fdf),use_container_width=True)

            with st.expander("📋 Full Table"):
                cols=[c for c in ["timestamp","ticker","signal_type","monster_score","entry_price","stop_loss","shares","event_impact","volume_spike"] if c in fdf.columns]
                st.dataframe(fdf[cols].head(100),use_container_width=True,hide_index=True)


# ─────────────────────────────
# TAB 2 — ANALYTICS
# ─────────────────────────────

with tab2:
    st.markdown("### 📊 Analytics")
    if signals_df.empty:
        st.info("No signals to analyse.")
    else:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### Monster Score V4 Radar")
            buy_rows=signals_df[signals_df["signal_type"].isin(["BUY_STRONG","BUY"])]
            if not buy_rows.empty:
                st.plotly_chart(chart_score_radar(load_monster_components(buy_rows.iloc[0])),use_container_width=True)
            else:
                st.info("No BUY signal yet")
        with c2:
            st.markdown("#### Score Distribution")
            if "monster_score" in signals_df.columns:
                st.plotly_chart(chart_score_dist(signals_df),use_container_width=True)

        st.markdown("#### Signal Type Breakdown")
        st.plotly_chart(chart_type_bar(signals_df),use_container_width=True)
        st.divider()

        st.markdown("#### 🚀 V8/V9 Boosts")
        boost_df=pd.DataFrame({"Boost":["Beat Rate","Extended Hours","Acceleration V8","Insider","Short Squeeze"],
            "Max":[0.15,0.22,0.15,0.15,0.20],
            "Source":["Historical earnings","Gap+AH/PM vol","ACCUMULATING/LAUNCHING","SEC Form 4","Short float"]})
        st.dataframe(boost_df,use_container_width=True,hide_index=True)
        st.divider()

        st.markdown("#### 📋 Proposed Orders")
        order_df=signals_df[signals_df["signal_type"].isin(["BUY_STRONG","BUY"])].copy()
        if not order_df.empty:
            cols=[c for c in ["timestamp","ticker","signal_type","monster_score","entry_price","stop_loss","shares"] if c in order_df.columns]
            st.dataframe(order_df[cols].head(20),use_container_width=True,hide_index=True)
            st.caption("⚠️ IBKR READ ONLY — ordres non exécutés")
        else:
            st.info("No BUY signals yet")


# ─────────────────────────────
# TAB 3 — EVENTS
# ─────────────────────────────

with tab3:
    st.markdown("### 📅 Catalysts & Events")
    # Controls row
    rc1, rc2 = st.columns([5, 1])
    with rc2:
        if st.button("🔄 Refresh", key="refresh_events", help="Re-fetch events from all sources"):
            try:
                from src.event_engine.event_hub import refresh_events
                from src.universe_loader import get_tickers
                with st.spinner("Fetching events…"):
                    tickers = (get_tickers(limit=150) or [])
                    refresh_events(tickers=tickers or None)
                st.cache_data.clear()
                st.success(f"Events refreshed! ({len(tickers)} tickers)")
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    events_cache = load_events_cache()

    # If file cache is empty, try live fetch (earnings + breaking news only, no tickers needed)
    if not events_cache:
        try:
            from src.event_engine.event_hub import build_events
            events_cache = build_events(tickers=None) or []
        except Exception:
            events_cache = []

    ce, cside = st.columns([2, 1])
    with ce:
        if events_cache:
            df_ev = pd.DataFrame(events_cache)
            # Extract headline from metadata for news events
            if "metadata" in df_ev.columns:
                df_ev["headline"] = df_ev["metadata"].apply(
                    lambda m: m.get("headline", "") if isinstance(m, dict) else ""
                )
            if "type" in df_ev.columns:
                st.markdown("#### Event Types")
                st.plotly_chart(chart_events_pie(df_ev["type"].value_counts()), use_container_width=True)
                # Filter by type
                all_types = sorted(df_ev["type"].dropna().unique().tolist())
                sel_types = st.multiselect("Filter by type", all_types, default=all_types, key="ev_type_filter")
                df_ev = df_ev[df_ev["type"].isin(sel_types)] if sel_types else df_ev
            show = [c for c in ["ticker", "type", "boosted_impact", "date", "headline", "is_bearish"] if c in df_ev.columns]
            if show:
                st.markdown(f"#### Recent Events ({len(df_ev)} total)")
                st.dataframe(df_ev[show].head(100), use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_ev.head(50), use_container_width=True, hide_index=True)
        else:
            st.info("No events cached yet — click 🔄 Refresh to fetch live events.")
    with cside:
        st.markdown("#### 📆 Upcoming Catalysts")
        cal_rows = []

        # SOURCE 1: FDA events (PDUFA, trials, conferences)
        try:
            from src.fda_calendar import get_all_fda_events
            for e in get_all_fda_events():
                date = e.get("date") or e.get("start_date", "")
                ticker = e.get("ticker", "—")
                etype  = e.get("type", "FDA")
                name   = e.get("name", "")
                note   = name if name else etype
                cal_rows.append({"Date": date, "Ticker": ticker, "Type": etype, "Note": note})
        except Exception:
            pass

        # SOURCE 2: ALL events from events_cache
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cutoff_earnings = (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d")
        cutoff_news     = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        for e in events_cache:
            etype  = e.get("type", "")
            date   = e.get("date", "")
            ticker = e.get("ticker", "—") or "—"
            meta   = e.get("metadata", {}) or {}

            if etype == "earnings":
                if not (today_str <= date <= cutoff_earnings):
                    continue
                eps  = meta.get("eps_estimate")
                note = f"EPS est. {eps:.2f}" if eps is not None else "earnings"

            elif etype == "news":
                if not (today_str <= date <= cutoff_news):
                    continue
                note = meta.get("headline", meta.get("text", "news"))[:60]

            else:
                # FDA_APPROVAL, M_AND_A, CONTRACT, etc. — no date filter
                if date and date < today_str:
                    continue
                note = meta.get("headline", meta.get("text", etype))[:60]

            cal_rows.append({
                "Date":   date,
                "Ticker": ticker,
                "Type":   etype,
                "Note":   note,
            })

        if cal_rows:
            df_cal = pd.DataFrame(cal_rows).sort_values("Date").drop_duplicates(
                subset=["Date", "Ticker", "Type"]
            ).reset_index(drop=True)
            st.dataframe(
                df_cal,
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "Date":   st.column_config.TextColumn("Date",   width="small"),
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Type":   st.column_config.TextColumn("Type",   width="small"),
                    "Note":   st.column_config.TextColumn("Note",   width="medium"),
                },
            )
            st.caption(f"{len(df_cal)} upcoming catalysts")
        else:
            st.caption("Aucun catalyst à venir")


# ─────────────────────────────
# TAB 4 — MULTI-RADAR V9
# ─────────────────────────────

_RADAR_LABELS = {
    "flow":        ("🌊 Flow",        "#3b82f6"),
    "catalyst":    ("⚡ Catalyst",    "#f59e0b"),
    "smart_money": ("💰 Smart Money", "#8b5cf6"),
    "sentiment":   ("💬 Sentiment",   "#10b981"),
}
_SESSION_WEIGHTS = {
    "Session":      ["AFTER_HOURS","PRE_MARKET","RTH_OPEN","RTH_MIDDAY","RTH_CLOSE","CLOSED"],
    "Flow %":       [15, 30, 35, 40, 30,  5],
    "Catalyst %":   [45, 30, 20, 20, 30, 50],
    "SmartMoney %": [10, 15, 30, 25, 25,  5],
    "Sentiment %":  [30, 25, 15, 15, 15, 40],
}

with tab4:
    st.markdown("### 🛰️ Multi-Radar Engine V9")

    # ── Row 1: Session weights + confluence matrix ──────────────────
    rw1, rw2 = st.columns([3, 2])
    with rw1:
        st.markdown("#### ⚖️ Session Weights")
        _sess_idx = {"AFTER_HOURS":0,"PRE_MARKET":1,"RTH_OPEN":2,"RTH_MIDDAY":3,"RTH_CLOSE":4,"CLOSED":5}
        cur_idx = _sess_idx.get(session, -1)
        df_sw = pd.DataFrame(_SESSION_WEIGHTS)
        # Mark active session with indicator — no Styler (avoids Streamlit compatibility issues)
        df_sw.insert(0, "▶", ["◀" if i == cur_idx else "" for i in range(len(df_sw))])
        st.dataframe(df_sw, use_container_width=True, hide_index=True)
        st.caption(f"Session active : **{session}**")
    with rw2:
        st.markdown("#### 🔀 Confluence Matrix")
        st.markdown(
            "| Flow \\ Catalyst | HIGH ≥0.6 | MED 0.3-0.6 | LOW <0.3 |\n"
            "|:---|:---:|:---:|:---:|\n"
            "| **HIGH** | 🟢 BUY_STRONG | 🔵 BUY | 👁 WATCH |\n"
            "| **MED**  | 🔵 BUY | 👁 WATCH | 🟡 EARLY |\n"
            "| **LOW**  | 👁 WATCH | 🟡 EARLY | ⬜ NO_SIGNAL |"
        )
        st.caption("Smart Money HIGH → +1 niveau · Sentiment HIGH + 2+ radars → +1 · 4/4 UNANIMOUS → +0.15")

    st.divider()

    # ── Row 2: General Watch List ────────────────────────────────────
    st.markdown("#### 📋 Watch List Générale")
    wl_days = st.slider("Horizon (jours)", 1, 30, 14, key="wl_days")
    wl_imp  = st.slider("Impact minimum", 0.1, 1.0, 0.5, step=0.05, key="wl_imp")
    # load_watch_list() fetches all data once (cached 5min) — filter client-side
    _all_wl = load_watch_list()
    watch_list_data = [
        w for w in _all_wl
        if w.get("impact", 0) >= wl_imp
        and w.get("days_to_event", 99) <= wl_days
    ]

    if watch_list_data:
        wl_rows = []
        for w in watch_list_data:
            wl_rows.append({
                "Ticker":  w.get("ticker", "—"),
                "Type":    w.get("event_type", "—"),
                "Date":    w.get("event_date", "—"),
                "J-":      w.get("days_to_event", "—"),
                "Impact":  round(w.get("impact", 0), 2),
                "Prob %":  round(w.get("probability", 0) * 100, 1),
                "Raison":  w.get("reason", "—"),
            })
        st.dataframe(
            pd.DataFrame(wl_rows),
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "Impact":  st.column_config.ProgressColumn("Impact",  min_value=0, max_value=1, format="%.2f"),
                "Prob %":  st.column_config.ProgressColumn("Prob %",  min_value=0, max_value=100, format="%.1f%%"),
                "J-":      st.column_config.NumberColumn("J-", help="Jours avant l'événement"),
            },
        )
        st.caption(f"{len(watch_list_data)} tickers en surveillance")
    else:
        st.info("Aucun ticker en watch list (pas d'events à venir avec cet impact)")

    st.divider()

    # ── Row 3: Radar States + Per-Radar Watch Lists ──────────────────
    st.markdown("#### 🎯 État des Radars & Watch List par Radar")
    radar_signals = load_radar_signals(200)

    # Build per-radar maps from DB signals
    per_radar: dict = {k: [] for k in _RADAR_LABELS}
    radar_last_score: dict = {k: None for k in _RADAR_LABELS}
    radar_last_state: dict = {k: "—" for k in _RADAR_LABELS}

    for sig in radar_signals:
        r = sig.get("radar")
        if not isinstance(r, dict) or "radars" not in r:
            continue
        for rname in _RADAR_LABELS:
            rdata = r["radars"].get(rname, {})
            if not rdata:
                continue
            score = rdata.get("score", 0)
            state = rdata.get("state", "—")
            # Update last known score/state (first hit = most recent)
            if radar_last_score[rname] is None:
                radar_last_score[rname] = score
                radar_last_state[rname] = state
            # Collect active tickers for per-radar watch list
            if rdata.get("is_active") or score >= 0.3:
                per_radar[rname].append({
                    "Ticker":      sig["ticker"],
                    "Signal":      sig["signal_type"],
                    "Score radar": round(score, 2),
                    "State":       state,
                    "Score total": round(sig["score"], 2),
                    "Time":        sig["timestamp"][:16] if sig["timestamp"] else "—",
                })

    # 4 radar state cards
    rc1, rc2, rc3, rc4 = st.columns(4)
    for col, (rname, (label, color)) in zip([rc1, rc2, rc3, rc4], _RADAR_LABELS.items()):
        sc = radar_last_score[rname]
        st_txt = radar_last_state[rname]
        n_tickers = len(per_radar[rname])
        sc_str = f"{sc:.2f}" if sc is not None else "N/A"
        bar = int((sc or 0) * 10)
        col.markdown(
            f"""<div class="card" style="border-left:3px solid {color};padding:.6rem .8rem;">
            <div style="color:{color};font-weight:700;font-size:.9rem;">{label}</div>
            <div style="font-size:1.4rem;font-weight:700;margin:.2rem 0;">{sc_str}</div>
            <div style="font-size:.75rem;color:#9ca3af;">{'█'*bar}{'░'*(10-bar)}</div>
            <div style="font-size:.72rem;color:#6b7280;margin-top:.3rem;">
                State: <b style="color:#f9fafb">{st_txt}</b><br>
                Tickers actifs: <b style="color:#f9fafb">{n_tickers}</b>
            </div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-radar watch list tabs
    rtab_flow, rtab_cat, rtab_sm, rtab_sent = st.tabs([
        "🌊 Flow", "⚡ Catalyst", "💰 Smart Money", "💬 Sentiment"
    ])
    for rtab, rname in zip([rtab_flow, rtab_cat, rtab_sm, rtab_sent], _RADAR_LABELS):
        with rtab:
            rows = per_radar[rname]
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).drop_duplicates("Ticker").sort_values("Score radar", ascending=False),
                    use_container_width=True, hide_index=True, height=280,
                    column_config={
                        "Score radar":  st.column_config.ProgressColumn("Score radar",  min_value=0, max_value=1, format="%.2f"),
                        "Score total":  st.column_config.ProgressColumn("Score total",  min_value=0, max_value=1, format="%.2f"),
                    },
                )
            else:
                st.info("Aucun signal récent pour ce radar (DB vide ou marché fermé)")


# ─────────────────────────────
# TAB 5 — API MONITOR
# ─────────────────────────────

with tab5:
    st.markdown("### 🌐 API Monitor — Live")
    st.caption("Source : `data/logs/api_monitor.log`")

    api_lines=load_api_monitor(200)
    if not api_lines:
        st.info("api_monitor.log vide — attend les premières requêtes")
    else:
        total_c=len(api_lines)
        ok_c   =sum(1 for l in api_lines if "| OK(" in l)
        warn_c =sum(1 for l in api_lines if "RATE_LIMIT" in l or "FORBIDDEN" in l)
        err_c  =sum(1 for l in api_lines if "| ERROR |" in l or "ERR=" in l)
        ok_pct =ok_c/total_c*100 if total_c else 0

        a1,a2,a3,a4=st.columns(4)
        a1.markdown(f'<div class="kpi"><div class="kpi-value">{total_c}</div><div class="kpi-label">Total Calls</div></div>',unsafe_allow_html=True)
        a2.markdown(f'<div class="kpi"><div class="kpi-value green">{ok_c}</div><div class="kpi-label">Success</div><div class="kpi-delta green">{ok_pct:.1f}%</div></div>',unsafe_allow_html=True)
        a3.markdown(f'<div class="kpi"><div class="kpi-value yellow">{warn_c}</div><div class="kpi-label">Rate Limits</div></div>',unsafe_allow_html=True)
        a4.markdown(f'<div class="kpi"><div class="kpi-value red">{err_c}</div><div class="kpi-label">Errors</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        lat_fig=chart_api_latency(api_lines)
        if lat_fig:
            st.markdown("#### Avg Latency by Provider")
            st.plotly_chart(lat_fig,use_container_width=True)

        st.markdown("#### Provider Breakdown")
        providers={}
        for line in api_lines:
            try:
                parts=[p.strip() for p in line.split("|")]
                if len(parts)<8: continue
                prov=parts[4]; tag=parts[6]
                if prov not in providers: providers[prov]={"ok":0,"warn":0,"err":0,"total":0}
                providers[prov]["total"]+=1
                if tag.startswith("OK"): providers[prov]["ok"]+=1
                elif "RATE_LIMIT" in tag or "FORBIDDEN" in tag: providers[prov]["warn"]+=1
                elif "ERR" in tag or "TIMEOUT" in tag: providers[prov]["err"]+=1
            except Exception: continue
        if providers:
            prov_df=pd.DataFrame([{"Provider":p,"Total":v["total"],"OK":v["ok"],"Warn":v["warn"],"Errors":v["err"],"Rate%":f"{v['ok']/v['total']*100:.1f}%" if v["total"] else "0%"} for p,v in sorted(providers.items(),key=lambda x:-x[1]["total"])])
            st.dataframe(prov_df,use_container_width=True,hide_index=True)

        st.markdown("#### Live Log")
        api_html=[]
        for line in api_lines[-80:]:
            esc=line.strip().replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            if "| ERROR |" in line or "ERR=" in line: api_html.append(f'<div class="api-row api-err">{esc}</div>')
            elif "RATE_LIMIT" in line or "FORBIDDEN" in line or "| WARNING |" in line: api_html.append(f'<div class="api-row api-warn">{esc}</div>')
            else: api_html.append(f'<div class="api-row api-ok">{esc}</div>')
        st.markdown(f'<div class="log-container">{"".join(api_html)}</div>',unsafe_allow_html=True)


# ─────────────────────────────
# TAB 6 — LIVE LOGS
# ─────────────────────────────

with tab6:
    st.markdown("### 📋 Live System Logs")
    log_files=sorted(f.name for f in LOGS_DIR.glob("*.log")) if LOGS_DIR.exists() else []
    lc1,lc2=st.columns([1,3])
    PRIORITY_LOGS=["main.log","signal_producer.log","multi_radar.log",
                   "execution_gate.log","ibkr_connector.log","api_monitor.log","telegram_alerts.log"]
    with lc1:
        ordered=[l for l in PRIORITY_LOGS if l in log_files]+[l for l in log_files if l not in PRIORITY_LOGS]
        selected_log=st.selectbox("Log file",ordered,index=0) if log_files else None
        n_lines=st.select_slider("Lines",[50,100,200,500],value=100)
        level_filter=st.multiselect("Level",["INFO","WARNING","ERROR","DEBUG"],default=["INFO","WARNING","ERROR"])
        keyword=st.text_input("Keyword",placeholder="BUY_STRONG, ERROR…")
        st.markdown("---"); st.markdown("**Sizes**")
        for lf in ordered[:12]:
            try:
                sz=(LOGS_DIR/lf).stat().st_size
                sz_str=f"{sz/1024:.0f}KB" if sz<1048576 else f"{sz/1048576:.1f}MB"
                st.caption(f"{'**' if lf==selected_log else ''}{lf}{'**' if lf==selected_log else ''} — {sz_str}")
            except Exception: pass

    with lc2:
        if selected_log:
            lines=load_log_tail(selected_log,n_lines)
            if level_filter: lines=[l for l in lines if any(lv in l for lv in level_filter)]
            if keyword: lines=[l for l in lines if keyword.lower() in l.lower()]
            st.markdown(f"#### `{selected_log}` — {len(lines)} lines")
            html_lines=[]
            for line in lines:
                esc=line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                if "| ERROR |" in line or ("ERROR" in line[:50] and "|" in line):
                    html_lines.append(f'<span class="log-error">{esc}</span>')
                elif "| WARNING |" in line or ("WARNING" in line[:50] and "|" in line):
                    html_lines.append(f'<span class="log-warn">{esc}</span>')
                elif "OK(" in line or "STARTED" in line or "✅" in line:
                    html_lines.append(f'<span class="log-ok">{esc}</span>')
                else:
                    html_lines.append(f'<span class="log-info">{esc}</span>')
            st.markdown(f'<div class="log-container">{"".join(html_lines)}</div>',unsafe_allow_html=True)
        else:
            st.info("Select a log file")

    st.markdown("---"); st.markdown("#### ⚡ Recent ERRORs — all logs")
    if log_files:
        err_lines=[]
        for lf in log_files:
            for line in load_log_tail(lf,80):
                if "| ERROR |" in line or ("ERROR" in line[:50] and "|" in line):
                    err_lines.append(f"[{lf}] {line.strip()}")
        if err_lines:
            html="".join([f'<div style="font-family:JetBrains Mono,monospace;font-size:.7rem;color:#ef4444;padding:.1rem 0;">{l.replace("<","&lt;")}</div>' for l in err_lines[-20:]])
            st.markdown(f'<div class="log-container" style="height:220px;">{html}</div>',unsafe_allow_html=True)
        else:
            st.success("No recent errors ✅")


# ─────────────────────────────
# TAB 7 — AUDIT
# ─────────────────────────────

with tab7:
    st.markdown("### 🔍 Performance Audit")
    if audit_data:
        cg,cm=st.columns([1,2])
        with cg:
            st.markdown("#### Hit Rate")
            st.plotly_chart(chart_hit_rate_gauge(audit_data.get("hit_rate",0)),use_container_width=True)
        with cm:
            st.markdown("#### Metrics")
            metrics=[("Total Signals",audit_data.get("total_signals",0),"📡"),
                     ("True Positives",audit_data.get("true_positives",0),"🟢"),
                     ("False Positives",audit_data.get("false_positives",0),"🔴"),
                     ("Missed Movers",audit_data.get("missed_movers",0),"⚠️"),
                     ("Early Catch",f"{audit_data.get('early_catch_rate',0)*100:.1f}%","⏰"),
                     ("Avg Lead",f"{audit_data.get('avg_lead_time_hours',0):.1f}h","📊")]
            ca,cb,cc=st.columns(3)
            for i,(lbl,val,ico) in enumerate(metrics):
                [ca,cb,cc][i%3].markdown(f"""<div class="card" style="text-align:center;">
                    <div style="color:#9ca3af;font-size:.7rem;">{ico} {lbl}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;font-weight:700;color:#f9fafb;">{val}</div>
                </div>""",unsafe_allow_html=True)
        with st.expander("📋 Full Audit JSON"): st.json(audit_data)
        if audit_data.get("missed_details"):
            st.markdown("#### ❌ Missed Movers")
            st.dataframe(pd.DataFrame(audit_data["missed_details"]),use_container_width=True,hide_index=True)
    else:
        st.info("No audit report yet.")
        if st.button("🚀 Run Daily Audit Now"):
            with st.spinner("Running…"):
                try:
                    from daily_audit import run_daily_audit
                    run_daily_audit(send_telegram=False)
                    st.cache_data.clear(); st.success("Done!"); st.rerun()
                except Exception as e: st.error(f"Audit failed: {e}")


# ─────────────────────────────
# TAB 8 — UNIVERSE
# ─────────────────────────────

with tab8:
    st.markdown("### 🌍 Universe — All Tracked Tickers")

    uni_df = load_universe_df()

    if uni_df.empty:
        st.warning("Universe CSV not found. Run the system at least once to generate `data/universe.csv`.")
    else:
        # ── Stats row ──
        exchanges = uni_df["exchange"].unique().tolist()
        exc_counts = uni_df["exchange"].value_counts()

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Total Tickers", f"{len(uni_df):,}")
        u2.metric("Exchanges", len([e for e in exchanges if e]))
        top_exc = exc_counts.index[0] if len(exc_counts) else "—"
        u3.metric("Largest Exchange", top_exc, f"{exc_counts.iloc[0]:,} tickers" if len(exc_counts) else "")
        # Coverage: tickers with a name
        named = int((uni_df["name"].str.len() > 0).sum())
        u4.metric("With Company Name", f"{named:,}")

        st.markdown("---")

        # ── Filters ──
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            search_q = st.text_input("🔍 Search ticker or name", placeholder="e.g. AAPL, Apple, TSLA…")
        with fc2:
            exc_options = ["All"] + sorted([e for e in exchanges if e])
            exc_filter = st.selectbox("Exchange", exc_options)
        with fc3:
            sort_col = st.selectbox("Sort by", ["ticker", "exchange", "name"])

        # ── Apply filters ──
        filtered = uni_df.copy()
        if search_q:
            q = search_q.strip().upper()
            mask = (filtered["ticker"].str.contains(q, na=False) |
                    filtered["name"].str.upper().str.contains(q, na=False))
            filtered = filtered[mask]
        if exc_filter != "All":
            filtered = filtered[filtered["exchange"] == exc_filter]
        filtered = filtered.sort_values(sort_col).reset_index(drop=True)

        # ── Result count ──
        st.caption(f"Showing **{len(filtered):,}** of **{len(uni_df):,}** tickers")

        # ── Exchange breakdown chips ──
        chips_html = ""
        for exc, cnt in exc_counts.items():
            if not exc:
                continue
            color = {"XNAS": "#3b82f6", "XNYS": "#10b981", "ARCX": "#8b5cf6",
                     "XASE": "#f59e0b", "BATS": "#06b6d4"}.get(exc, "#6b7280")
            chips_html += (f'<span style="display:inline-block;margin:.2rem;padding:.2rem .55rem;'
                           f'border-radius:20px;font-size:.72rem;font-weight:600;'
                           f'background:{color}22;color:{color};border:1px solid {color};">'
                           f'{exc} <b>{cnt:,}</b></span>')
        if chips_html:
            st.markdown(chips_html, unsafe_allow_html=True)
            st.markdown("")

        # ── Main table ──
        st.dataframe(
            filtered.rename(columns={"ticker": "Ticker", "exchange": "Exchange", "name": "Company Name"}),
            use_container_width=True,
            hide_index=True,
            height=min(600, 40 + len(filtered) * 35),
            column_config={
                "Ticker":       st.column_config.TextColumn("Ticker", width="small"),
                "Exchange":     st.column_config.TextColumn("Exchange", width="small"),
                "Company Name": st.column_config.TextColumn("Company Name"),
            }
        )

        # ── Download button ──
        csv_bytes = filtered.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"universe_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ─────────────────────────────
# TAB 9 — TICKER PROFILES
# ─────────────────────────────

with tab9:
    st.markdown("### 🗂️ Ticker Profile Store — Strategic DB")

    prof_df = load_ticker_profiles()

    if prof_df.empty:
        st.info("No profiles yet. Run the weekend batch (`ScanType.TICKER_PROFILES`) or `asyncio.run(update_ticker_profile('AAPL'))` to populate.")
    else:
        # ── KPI row ──
        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        updated_today = int((prof_df["updated_at"].str[:10] == today_iso).sum()) if "updated_at" in prof_df.columns else 0
        avg_q = prof_df["data_quality"].mean() if "data_quality" in prof_df.columns else 0
        squeeze_setups = int(((prof_df.get("short_interest_pct", 0) or 0) > 20).sum()) if "short_interest_pct" in prof_df.columns else 0
        htb_count = int(((prof_df.get("borrow_rate", 0) or 0) > 100).sum()) if "borrow_rate" in prof_df.columns else 0

        pk1, pk2, pk3, pk4, pk5 = st.columns(5)
        pk1.metric("Total Profiles", f"{len(prof_df):,}")
        pk2.metric("Avg Data Quality", f"{avg_q:.0%}")
        pk3.metric("Squeeze Setups (SI>20%)", f"{squeeze_setups:,}")
        pk4.metric("HTB (borrow>100%)", f"{htb_count:,}")
        pk5.metric("Updated Today", f"{updated_today:,}")

        st.markdown("---")

        # ── Filters ──
        pf1, pf2, pf3, pf4, pf5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
        with pf1:
            p_search = st.text_input("🔍 Ticker", placeholder="AAPL, TSLA…", key="p_search")
        with pf2:
            p_max_float = st.number_input("Float max (M shares)", min_value=0.0, value=0.0, step=1.0, key="p_float",
                                          help="0 = no limit")
        with pf3:
            p_min_si = st.number_input("SI min (%)", min_value=0.0, value=0.0, step=1.0, key="p_si")
        with pf4:
            p_min_quality = st.slider("Quality min", 0.0, 1.0, 0.0, 0.1, key="p_qual")
        with pf5:
            p_risk_flags = st.multiselect("Risk flags", ["ATM Active", "Shelf Active", "Warrants", "Death Spiral (3+RS)"],
                                          key="p_flags")

        # ── Apply filters ──
        fdf = prof_df.copy()
        if p_search:
            fdf = fdf[fdf["ticker"].str.upper().str.contains(p_search.upper(), na=False)]
        if p_max_float > 0 and "float_shares" in fdf.columns:
            fdf = fdf[fdf["float_shares"].fillna(float("inf")) <= p_max_float * 1_000_000]
        if p_min_si > 0 and "short_interest_pct" in fdf.columns:
            fdf = fdf[fdf["short_interest_pct"].fillna(0) >= p_min_si]
        if p_min_quality > 0:
            fdf = fdf[fdf["data_quality"].fillna(0) >= p_min_quality]
        if "ATM Active" in p_risk_flags and "atm_active" in fdf.columns:
            fdf = fdf[fdf["atm_active"] == 1]
        if "Shelf Active" in p_risk_flags and "shelf_active" in fdf.columns:
            fdf = fdf[fdf["shelf_active"] == 1]
        if "Warrants" in p_risk_flags and "warrants_outstanding" in fdf.columns:
            fdf = fdf[fdf["warrants_outstanding"] == 1]
        if "Death Spiral (3+RS)" in p_risk_flags and "reverse_split_count" in fdf.columns:
            fdf = fdf[fdf["reverse_split_count"].fillna(0) >= 3]

        st.caption(f"Showing **{len(fdf):,}** of **{len(prof_df):,}** profiles")

        # ── Display columns — format for readability ──
        DISPLAY_COLS = {
            "ticker":             "Ticker",
            "market_cap":         "Mkt Cap",
            "float_shares":       "Float (M)",
            "short_interest_pct": "SI %",
            "days_to_cover":      "DTC",
            "borrow_rate":        "Borrow %",
            "insider_pct":        "Insider %",
            "institutional_pct":  "Inst %",
            "reverse_split_count":"RS Count",
            "shelf_active":       "Shelf",
            "atm_active":         "ATM",
            "dilution_tier":      "Dilution Tier",
            "top_gainer_count":   "Top Gains",
            "avg_move_pct":       "Avg Move %",
            "data_quality":       "Quality",
            "updated_at":         "Updated",
        }
        avail = [c for c in DISPLAY_COLS if c in fdf.columns]
        disp = fdf[avail].copy()

        # Format numbers
        if "market_cap" in disp.columns:
            disp["market_cap"] = disp["market_cap"].apply(
                lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x >= 1e9
                else (f"${x/1e6:.0f}M" if pd.notna(x) and x > 0 else "—"))
        if "float_shares" in disp.columns:
            disp["float_shares"] = disp["float_shares"].apply(
                lambda x: f"{x/1e6:.1f}" if pd.notna(x) and x > 0 else "—")
        for pct_col in ("short_interest_pct", "borrow_rate", "insider_pct", "institutional_pct"):
            if pct_col in disp.columns:
                disp[pct_col] = disp[pct_col].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0 else "—")
        for flag_col in ("shelf_active", "atm_active"):
            if flag_col in disp.columns:
                disp[flag_col] = disp[flag_col].apply(lambda x: "⚠️" if x == 1 else "")
        if "data_quality" in disp.columns:
            disp["data_quality"] = disp["data_quality"].apply(
                lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        if "updated_at" in disp.columns:
            disp["updated_at"] = disp["updated_at"].str[:16].str.replace("T", " ")
        if "days_to_cover" in disp.columns:
            disp["days_to_cover"] = disp["days_to_cover"].apply(
                lambda x: f"{x:.1f}d" if pd.notna(x) and x > 0 else "—")

        disp = disp.rename(columns=DISPLAY_COLS)

        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            height=min(650, 50 + len(disp) * 35),
        )

        # ── Download ──
        dl1, dl2 = st.columns([1, 5])
        with dl1:
            csv_bytes = fdf.to_csv(index=False).encode()
            st.download_button("⬇️ CSV", data=csv_bytes,
                               file_name=f"ticker_profiles_{today_iso}.csv", mime="text/csv")

        # ── Selected ticker detail panel ──
        st.markdown("---")
        st.markdown("#### 🔎 Ticker Detail")
        detail_ticker = st.text_input("Enter ticker to inspect", placeholder="e.g. AAPL", key="p_detail").upper().strip()
        if detail_ticker:
            row = prof_df[prof_df["ticker"] == detail_ticker]
            if row.empty:
                st.warning(f"No profile found for **{detail_ticker}**. Run `update_ticker_profile('{detail_ticker}')` first.")
            else:
                r = row.iloc[0]
                d1, d2, d3, d4 = st.columns(4)
                def _v(val, fmt=None):
                    if pd.isna(val) or val is None: return "—"
                    return fmt.format(val) if fmt else str(val)

                with d1:
                    st.markdown("**Capital Structure**")
                    st.metric("Market Cap",  _v(r.get("market_cap"),  "${:,.0f}"))
                    st.metric("Float",       f"{r['float_shares']/1e6:.1f}M" if pd.notna(r.get('float_shares')) and r.get('float_shares',0)>0 else "—")
                    st.metric("Shares Out",  f"{r['shares_outstanding']/1e6:.1f}M" if pd.notna(r.get('shares_outstanding')) and r.get('shares_outstanding',0)>0 else "—")
                    st.metric("Insider %",   _v(r.get("insider_pct"), "{:.1f}%"))
                    st.metric("Inst %",      _v(r.get("institutional_pct"), "{:.1f}%"))
                with d2:
                    st.markdown("**Short Squeeze**")
                    st.metric("Short Interest",  _v(r.get("short_interest_pct"), "{:.1f}%"))
                    st.metric("Days to Cover",   _v(r.get("days_to_cover"), "{:.1f}d"))
                    borrow = r.get("borrow_rate") or 0
                    st.metric("Borrow Rate", f"{borrow:.1f}%" if borrow > 0 else "—",
                              delta="HTB ⚠️" if borrow > 100 else None)
                with d3:
                    st.markdown("**Risk Flags**")
                    rs = int(r.get("reverse_split_count") or 0)
                    st.metric("Reverse Splits", f"{rs}x", delta="☠️ Death Spiral" if rs >= 3 else None)
                    st.metric("Last RS",         _v(r.get("last_reverse_split")))
                    st.metric("Shelf Active",    "⚠️ YES" if r.get("shelf_active") == 1 else "No")
                    st.metric("ATM Active",      "⚠️ YES" if r.get("atm_active") == 1 else "No")
                    st.metric("Dilution Tier",   _v(r.get("dilution_tier")))
                with d4:
                    st.markdown("**History & Technical**")
                    st.metric("Top Gainer Count",  _v(r.get("top_gainer_count")))
                    st.metric("Avg Move",           _v(r.get("avg_move_pct"), "{:.1f}%"))
                    st.metric("Best Session",       _v(r.get("best_session")))
                    st.metric("Catalyst Affinity",  _v(r.get("catalyst_affinity")))
                    st.metric("ATR-14",             _v(r.get("atr_14"), "${:.3f}"))
                    st.metric("Avg Daily Vol",      f"{r['avg_daily_volume']:,.0f}" if pd.notna(r.get('avg_daily_volume')) and r.get('avg_daily_volume',0)>0 else "—")

                # Quality bar
                q = float(r.get("data_quality") or 0)
                q_color = "#10b981" if q >= 0.7 else "#f59e0b" if q >= 0.4 else "#ef4444"
                st.markdown(
                    f'<div style="margin:.5rem 0 .2rem;font-size:.8rem;color:#9ca3af;">Data Quality</div>'
                    f'<div style="background:#1a1f2e;border-radius:6px;height:12px;overflow:hidden;">'
                    f'<div style="background:{q_color};width:{q*100:.0f}%;height:100%;border-radius:6px;"></div></div>'
                    f'<div style="font-size:.75rem;color:{q_color};margin-top:.2rem;">{q:.0%} — updated {_v(r.get("updated_at",""))[:16]}</div>',
                    unsafe_allow_html=True)


# ============================
# FOOTER
# ============================

st.markdown("---")
f1,f2,f3=st.columns(3)
f1.caption("🎯 GV2-EDGE V9.0 — Multi-Radar Detection Architecture")
f2.caption(f"Updated: {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET")
f3.caption("Small Caps US — Anticipation > Reaction 🚀")
