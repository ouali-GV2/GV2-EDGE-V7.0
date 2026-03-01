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
    --bg-primary:   #0a0e17;
    --bg-secondary: #111827;
    --bg-card:      #1a1f2e;
    --bg-hover:     #252b3b;
    --green:  #10b981; --red:    #ef4444;
    --yellow: #f59e0b; --blue:   #3b82f6;
    --purple: #8b5cf6; --cyan:   #06b6d4;
    --text:   #f9fafb; --muted:  #9ca3af; --dim: #6b7280;
    --border: #374151; --radius: 10px;
}

.stApp { background: linear-gradient(135deg,#0a0e17 0%,#0f172a 100%); font-family:'Outfit',sans-serif; }
#MainMenu, footer, header { visibility:hidden; }
/* Hide sidebar entirely — controls moved to top bar */
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }
/* Controls top bar */
.ctrl-bar {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .5rem .8rem;
    margin-bottom: .8rem;
    display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
}
h1,h2,h3 { font-family:'Outfit',sans-serif !important; font-weight:600 !important; color:var(--text) !important; }
h1 {
    font-size:clamp(1.4rem,3vw,2.2rem) !important;
    background:linear-gradient(90deg,#06b6d4,#10b981);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}

/* Cards */
.card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius); padding:.9rem; margin:.3rem 0;
    transition:border-color .2s,box-shadow .2s;
}
.card:hover { border-color:var(--cyan); box-shadow:0 0 8px rgba(6,182,212,.15); }
.card-buy-strong { background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(6,182,212,.08)); border-left:3px solid var(--green); }
.card-buy        { background:linear-gradient(135deg,rgba(59,130,246,.12),rgba(139,92,246,.08)); border-left:3px solid var(--blue); }
.card-watch      { background:linear-gradient(135deg,rgba(245,158,11,.12),rgba(239,68,68,.08));  border-left:3px solid var(--yellow); }
.card-early      { background:linear-gradient(135deg,rgba(139,92,246,.12),rgba(245,158,11,.08)); border-left:3px solid var(--purple); }
.card-alert      { border:1px solid var(--green); animation:alertPulse 2s infinite; }
@keyframes alertPulse { 0%,100%{box-shadow:0 0 12px rgba(16,185,129,.3)} 50%{box-shadow:0 0 28px rgba(16,185,129,.6)} }

/* KPI Grid — auto-fit responsive */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:.6rem; margin-bottom:1rem; }
.kpi {
    text-align:center; padding:1rem .5rem;
    background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
    transition:border-color .2s;
}
.kpi:hover { border-color:var(--cyan); }
.kpi-value { font-family:'JetBrains Mono',monospace; font-size:clamp(1.3rem,2.5vw,1.9rem); font-weight:700; color:var(--text); }
.kpi-label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin-top:.2rem; }
.kpi-delta { font-family:'JetBrains Mono',monospace; font-size:.76rem; margin-top:.2rem; }
.kpi-sub   { font-size:.65rem; color:var(--dim); margin-top:.1rem; }

/* Signal card grid — responsive */
.sig-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(255px,1fr)); gap:.55rem; }

/* Gap grid */
.gap-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(100px,1fr)); gap:.45rem; }
.gap-card { padding:.55rem; border-radius:8px; text-align:center; background:var(--bg-card); border:1px solid var(--border); }
.gap-up   { border-left:3px solid var(--green); }
.gap-down { border-left:3px solid var(--red); }

/* Badges */
.badge { display:inline-block; padding:.13rem .45rem; border-radius:4px; font-size:.68rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-strong { background:rgba(16,185,129,.2);  color:var(--green); }
.badge-buy    { background:rgba(59,130,246,.2);  color:var(--blue); }
.badge-watch  { background:rgba(245,158,11,.2);  color:var(--yellow); }
.badge-early  { background:rgba(139,92,246,.2);  color:var(--purple); }

.tick { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1rem; color:var(--cyan); }

.live-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 1.5s infinite; margin-right:5px; }
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 4px var(--green)} 50%{opacity:.4;box-shadow:none} }

/* Log viewer */
.log-container {
    background:#0d1117; border:1px solid var(--border); border-radius:8px;
    padding:.9rem; font-family:'JetBrains Mono',monospace; font-size:.71rem;
    height:480px; overflow-y:auto; white-space:pre-wrap; word-break:break-all;
}
.log-error { color:#ef4444; } .log-warn { color:#f59e0b; }
.log-info  { color:#9ca3af; } .log-ok   { color:#10b981; }

/* API monitor */
.api-ok   { color:var(--green); }
.api-warn { color:var(--yellow); }
.api-err  { color:var(--red); }
.api-row  { font-family:'JetBrains Mono',monospace; font-size:.7rem; border-bottom:1px solid #1e2436; padding:.18rem 0; }

/* Status pills */
.pill { display:inline-block; padding:.2rem .55rem; border-radius:20px; font-size:.7rem; font-weight:600; margin:.1rem; }
.pill-ok   { background:rgba(16,185,129,.15); color:var(--green);  border:1px solid var(--green); }
.pill-err  { background:rgba(239,68,68,.15);  color:var(--red);    border:1px solid var(--red); }

/* Sidebar */
section[data-testid="stSidebar"] { background:var(--bg-secondary); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:3px; background:var(--bg-card); padding:.3rem; border-radius:var(--radius); flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:6px; color:var(--muted); font-weight:500; font-size:clamp(.72rem,.85vw,.86rem); padding:.35rem .6rem; }
.stTabs [aria-selected="true"] { background:var(--bg-hover); color:var(--text); }

/* Scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg-secondary); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

/* Colors */
.green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--yellow)}
.cyan{color:var(--cyan)}   .muted{color:var(--muted)}
.score-high{color:var(--green)} .score-med{color:var(--yellow)} .score-low{color:var(--red)}

/* ════════════════════════════════════════
   RESPONSIVE — Desktop first
   ════════════════════════════════════════ */

/* ── Tablet (641-1024px) ── */
@media (min-width:641px) and (max-width:1024px) {
    .block-container { padding:1rem 1rem 2rem !important; }
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
    [data-testid="column"] { min-width:calc(50% - .5rem) !important; }
    .kpi-grid  { grid-template-columns:repeat(3,1fr) !important; }
    .sig-grid  { grid-template-columns:repeat(2,1fr) !important; }
    .gap-grid  { grid-template-columns:repeat(3,1fr) !important; }
}

/* ── Mobile (<640px) ── */
@media (max-width:640px) {
    /* Reduce outer padding */
    .block-container { padding:.3rem .3rem 2rem !important; }

    /* Stack ALL Streamlit columns */
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.2rem !important; }
    [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:0 !important; }

    /* Header */
    h1 { font-size:1.15rem !important; }
    h2 { font-size:.95rem !important; }
    h3 { font-size:.85rem !important; }

    /* KPI: 2 per row */
    .kpi-grid { grid-template-columns:repeat(2,1fr) !important; gap:.35rem !important; }
    .kpi { padding:.55rem .3rem !important; }
    .kpi-value { font-size:1.05rem !important; }
    .kpi-label { font-size:.58rem !important; }
    .kpi-sub,.kpi-delta { font-size:.58rem !important; }

    /* Signal/gap grids */
    .sig-grid { grid-template-columns:1fr !important; }
    .gap-grid { grid-template-columns:repeat(2,1fr) !important; }

    /* Cards: tighter */
    .card { padding:.6rem !important; margin:.2rem 0 !important; }
    .tick { font-size:.85rem !important; }

    /* Tabs: smaller text, horizontal scroll */
    .stTabs [data-baseweb="tab-list"] { overflow-x:auto !important; flex-wrap:nowrap !important; }
    .stTabs [data-baseweb="tab"] { font-size:.62rem !important; padding:.2rem .38rem !important; white-space:nowrap !important; }

    /* Log: shorter */
    .log-container { height:260px !important; font-size:.62rem !important; }

    /* Tables: horizontal scroll */
    [data-testid="stDataFrame"] > div { overflow-x:auto !important; }

    /* Charts: full width */
    [data-testid="stPlotlyChart"] { width:100% !important; }

    /* Expanders: tighter */
    [data-testid="stExpander"] { padding:.3rem !important; }

    /* Pills: smaller */
    .pill { font-size:.6rem !important; padding:.15rem .35rem !important; }

    /* Hide desktop-only decorative elements */
    .desktop-only { display:none !important; }
}
</style>
""", unsafe_allow_html=True)


# ============================
# PATHS
# ============================

DATA_DIR   = Path("data")
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

# Row 2 — Status expanders (collapsed by default)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    with st.expander("🛡️ APIs & IBKR"):
        status = get_system_status()
        cols_s = st.columns(2)
        for i,(k,ok) in enumerate(status.items()):
            cls = "pill-ok" if ok else "pill-err"
            cols_s[i%2].markdown(
                f'<span class="pill {cls}">{"🟢" if ok else "🔴"} {k.upper()}</span>',
                unsafe_allow_html=True)
        ibkr_info = get_ibkr_info()
        if ibkr_info:
            state = ibkr_info.get("state","?")
            icons = {"CONNECTED":"🟢","RECONNECTING":"🟡","DISCONNECTED":"🔴","CONNECTING":"🟡","FAILED":"🔴"}
            st.markdown(f"{icons.get(state,'⚪')} IBKR **{state}**")
            if ibkr_info.get("connected"):
                up  = ibkr_info.get("uptime_seconds",0)
                lat = ibkr_info.get("heartbeat_latency_ms",0)
                ustr = f"{up/3600:.1f}h" if up>=3600 else f"{up/60:.0f}min"
                st.caption(f"Up: {ustr}  Lat: {lat:.0f}ms")
with ex2:
    with st.expander("🧠 V9 Modules"):
        try:
            from config import (ENABLE_MULTI_RADAR,ENABLE_ACCELERATION_ENGINE,ENABLE_SMALLCAP_RADAR,
                ENABLE_PRE_HALT_ENGINE,ENABLE_RISK_GUARD,ENABLE_MARKET_MEMORY,
                ENABLE_CATALYST_V3,ENABLE_PRE_SPIKE_RADAR)
            mods={"Signal Producer (L1)":True,"Order Computer (L2)":True,"Execution Gate (L3)":True,
                  "Multi-Radar V9":ENABLE_MULTI_RADAR,"Acceleration V8":ENABLE_ACCELERATION_ENGINE,
                  "SmallCap Radar":ENABLE_SMALLCAP_RADAR,"Pre-Halt Engine":ENABLE_PRE_HALT_ENGINE,
                  "Risk Guard V8":ENABLE_RISK_GUARD,"Market Memory":ENABLE_MARKET_MEMORY,
                  "Catalyst V3":ENABLE_CATALYST_V3,"Pre-Spike Radar":ENABLE_PRE_SPIKE_RADAR}
        except Exception:
            mods={"Signal Producer":True,"Order Computer":True,"Execution Gate":True}
        for name,active in mods.items():
            st.markdown(f"{'🟢' if active else '⚫'} {name}")
with ex3:
    with st.expander("🕐 Session"):
        try:
            from zoneinfo import ZoneInfo
            _et = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            _et = datetime.now(timezone.utc) - timedelta(hours=4)
        st.markdown(f"**{_et.strftime('%A %H:%M:%S')} ET**")
        st.caption(f"Session : **{session}**  •  GV2-EDGE v9.0")

st.divider()


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

ct, cs, ctime = st.columns([5,1,1])
with ct:
    st.markdown("# 🎯 GV2-EDGE V9.0")
    st.markdown("**Multi-Radar Detection** — 3-Layer Pipeline — Small Caps US")

sess_map={"PREMARKET":("🌅","Pre-Market","#f59e0b"),"RTH":("📈","RTH","#10b981"),
          "AFTER_HOURS":("🌙","After-Hours","#8b5cf6"),"CLOSED":("💤","Closed","#6b7280")}
icon,label,color=sess_map.get(session,("❓",session,"#6b7280"))
with cs:
    st.markdown(f"""<div style="text-align:center;padding:.8rem;">
        <div style="font-size:1.4rem;">{icon}</div>
        <div style="color:{color};font-weight:600;font-size:.85rem;">{label}</div>
    </div>""",unsafe_allow_html=True)

try:
    from zoneinfo import ZoneInfo
    now_et=datetime.now(ZoneInfo("America/New_York"))
except Exception:
    now_et=datetime.now(timezone.utc)-timedelta(hours=4)

with ctime:
    st.markdown(f"""<div style="text-align:right;padding:.8rem;color:#9ca3af;font-size:.78rem;">
        <div><span class="live-dot"></span><b style="color:#10b981;">LIVE</b></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.92rem;color:#f9fafb;">{now_et.strftime("%H:%M:%S")}</div>
        <div style="font-size:.65rem;">{now_et.strftime("%Y-%m-%d")} ET | ↻{refresh_sec}s</div>
    </div>""",unsafe_allow_html=True)

st.markdown("---")


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

kpis = (
    kpi_html(total_sig,             "Signals",     f"Last {hours_back}h", delta_str,  "green" if delta_sig>0 else "muted"),
    kpi_html(buy_strong,            "BUY_STRONG",  "Hot",                 f"+{strong_1h}/1h", strong_cls),
    kpi_html(f"{hit_rate*100:.1f}%","Hit Rate",    "Target ≥65%",         "",          score_cls),
    kpi_html(f"{avg_lead:.1f}h",    "Lead Time",   "Before spike",        "",          "cyan"),
    kpi_html(f"{universe_sz:,}",    "Universe",    "Tickers tracked",     "",          "cyan"),
    kpi_html(session,               "Session",     now_et.strftime("%H:%M"), "",       "yellow" if session!="CLOSED" else "muted"),
)
st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================
# TABS
# ============================

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📡 Live Signals", "📊 Analytics", "📅 Events",
    "🛰️ Multi-Radar V9", "🌐 API Monitor", "📋 Live Logs", "🔍 Audit",
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

@st.cache_data(ttl=300)
def load_watch_list(days_forward: int = 14, min_impact: float = 0.5) -> list:
    try:
        from src.watch_list import generate_watch_list
        return generate_watch_list(days_forward=days_forward, min_impact=min_impact) or []
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
        df_sw = pd.DataFrame(_SESSION_WEIGHTS)
        _sess_idx = {"AFTER_HOURS":0,"PRE_MARKET":1,"RTH_OPEN":2,"RTH_MIDDAY":3,"RTH_CLOSE":4,"CLOSED":5}
        cur_idx = _sess_idx.get(session, -1)
        st.dataframe(
            df_sw.style.apply(
                lambda row: ["background-color:#1e3a5f" if row.name == cur_idx else "" for _ in row],
                axis=1
            ),
            use_container_width=True, hide_index=True,
        )
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
    watch_list_data = load_watch_list(days_forward=wl_days, min_impact=wl_imp)

    if watch_list_data:
        wl_rows = []
        for w in watch_list_data:
            wl_rows.append({
                "Ticker":      w.get("ticker", "—"),
                "Type":        w.get("event_type", "—"),
                "Date":        w.get("event_date", "—"),
                "J-":          w.get("days_to_event", "—"),
                "Impact":      round(w.get("impact", 0), 2),
                "Prob %":      round(w.get("probability", 0) * 100, 1),
                "Raison":      w.get("reason", "—"),
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


# ============================
# FOOTER
# ============================

st.markdown("---")
f1,f2,f3=st.columns(3)
f1.caption("🎯 GV2-EDGE V9.0 — Multi-Radar Detection Architecture")
f2.caption(f"Updated: {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET")
f3.caption("Small Caps US — Anticipation > Reaction 🚀")
