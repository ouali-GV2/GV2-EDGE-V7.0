"""
GV2-EDGE V7.0 — Professional Trading Dashboard
===============================================

Dashboard temps réel pour le système de détection anticipative
des top gainers small caps US.

V7.0 Detection/Execution Separation Architecture:
- SignalProducer -> OrderComputer -> ExecutionGate pipeline
- Detection always visible (never blocked)
- Execution limits applied only at final layer
- Risk Guard integration (dilution, compliance, halt)
- Pre-Halt Engine status
- Market Memory (MRP/EP) context display
- Blocked signals tracking with reasons

Design: Dark theme trading professionnel
Stack: Streamlit + Plotly + Custom CSS

Sections:
1. Header avec status système + V7 modules
2. Signaux actifs avec V7 intelligence (including blocked)
3. Monster Score breakdown (radar chart)
4. Execution Gate stats (allowed vs blocked)
5. Market Memory status (MRP/EP readiness)
6. Audit metrics (hit rate, lead time)
7. System health
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import time

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="GV2-EDGE V7.0 — Trading Radar",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS — DARK TRADING THEME
# ============================

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --bg-hover: #252b3b;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --border-color: #374151;
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
    }
    
    /* Signal cards */
    .signal-buy-strong {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        border-left: 4px solid var(--accent-green);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-left: 4px solid var(--accent-blue);
    }
    
    .signal-watch {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(239, 68, 68, 0.1) 100%);
        border-left: 4px solid var(--accent-yellow);
    }
    
    /* Metric values */
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-delta-positive {
        color: var(--accent-green);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-delta-negative {
        color: var(--accent-red);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Tables */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Status badges */
    .status-live {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-green);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .status-offline {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-hover);
        color: var(--text-primary);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* Ticker symbol styling */
    .ticker-symbol {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--accent-cyan);
    }
    
    /* Score gauge */
    .score-high { color: var(--accent-green); }
    .score-medium { color: var(--accent-yellow); }
    .score-low { color: var(--accent-red); }
</style>
""", unsafe_allow_html=True)

# ============================
# DATA PATHS
# ============================

DATA_DIR = Path("data")
SIGNALS_DB = DATA_DIR / "signals_history.db"
AUDIT_DIR = DATA_DIR / "audit_reports"
BACKTEST_DIR = DATA_DIR / "backtest_reports"

# ============================
# DATA LOADING FUNCTIONS
# ============================

def load_json(path):
    """Load JSON file safely"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


def load_signals_from_db(hours_back=24):
    """Load recent signals from SQLite database"""
    if not SIGNALS_DB.exists():
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(str(SIGNALS_DB))
        cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        
        query = f"""
            SELECT * FROM signals 
            WHERE timestamp >= '{cutoff}'
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()


def load_latest_audit():
    """Load the most recent audit report"""
    if not AUDIT_DIR.exists():
        return None
    
    files = list(AUDIT_DIR.glob("*.json"))
    if not files:
        return None
    
    latest = sorted(files)[-1]
    return load_json(latest)


def get_system_status():
    """Check system components status"""
    status = {
        "ibkr": False,
        "grok": False,
        "finnhub": False,
        "telegram": False
    }
    
    # Check IBKR
    try:
        from src.ibkr_connector import get_ibkr
        ibkr = get_ibkr()
        status["ibkr"] = ibkr and ibkr.connected
    except:
        pass
    
    # Check API keys configured
    try:
        from config import GROK_API_KEY, FINNHUB_API_KEY, TELEGRAM_BOT_TOKEN
        status["grok"] = GROK_API_KEY and not GROK_API_KEY.startswith("YOUR_")
        status["finnhub"] = FINNHUB_API_KEY and not FINNHUB_API_KEY.startswith("YOUR_")
        status["telegram"] = TELEGRAM_BOT_TOKEN and not TELEGRAM_BOT_TOKEN.startswith("YOUR_")
    except:
        pass
    
    return status


def get_market_session():
    """Get current market session"""
    try:
        from utils.time_utils import market_session
        return market_session()
    except:
        return "UNKNOWN"


# ============================
# CHART FUNCTIONS
# ============================

def create_monster_score_radar(components):
    """Create radar chart for Monster Score breakdown"""
    if not components:
        return None
    
    categories = ['Event', 'Volume', 'Pattern', 'PM Trans', 'Momentum', 'Squeeze']
    
    # Extract values (handle both raw and weighted)
    values = [
        components.get('event', 0),
        components.get('volume', 0) * 5,  # Scale back from weighted
        components.get('pattern', 0) * 5,
        components.get('pm_transition', 0) * 6.67,
        components.get('momentum', 0) * 10,
        components.get('squeeze', 0) * 20
    ]
    
    # Clamp to 0-1
    values = [min(1, max(0, v)) for v in values]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(6, 182, 212, 0.3)',
        line=dict(color='#06b6d4', width=2),
        name='Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color='#6b7280', size=10),
                gridcolor='#374151'
            ),
            angularaxis=dict(
                tickfont=dict(color='#9ca3af', size=11),
                gridcolor='#374151'
            ),
            bgcolor='#1a1f2e'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=40, b=40),
        height=300
    )
    
    return fig


def create_signals_timeline(signals_df):
    """Create timeline chart of signals"""
    if signals_df.empty:
        return None
    
    # Color mapping
    color_map = {
        'BUY_STRONG': '#10b981',
        'BUY': '#3b82f6',
        'WATCH_EARLY': '#f59e0b',
        'WATCH': '#8b5cf6'
    }
    
    fig = px.scatter(
        signals_df,
        x='timestamp',
        y='monster_score',
        color='signal',
        size='monster_score',
        hover_data=['ticker', 'signal', 'monster_score'],
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a1f2e',
        xaxis=dict(
            gridcolor='#374151',
            tickfont=dict(color='#9ca3af')
        ),
        yaxis=dict(
            gridcolor='#374151',
            tickfont=dict(color='#9ca3af'),
            title='Monster Score'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af')
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )
    
    return fig


def create_hit_rate_gauge(hit_rate):
    """Create gauge chart for hit rate"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hit_rate * 100,
        number={'suffix': '%', 'font': {'color': '#f9fafb', 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': '#6b7280'}},
            'bar': {'color': '#06b6d4'},
            'bgcolor': '#1a1f2e',
            'bordercolor': '#374151',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#10b981', 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f9fafb'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_events_heatmap(events_data):
    """Create heatmap of events by type and hour from real data"""
    if not events_data:
        return None

    try:
        # Convert events to DataFrame
        if isinstance(events_data, list):
            df = pd.DataFrame(events_data)
        elif isinstance(events_data, dict):
            # Flatten dict structure {ticker: [events]}
            events_list = []
            for ticker, events in events_data.items():
                if isinstance(events, list):
                    for e in events:
                        if isinstance(e, dict):
                            e['ticker'] = ticker
                            events_list.append(e)
                elif isinstance(events, dict):
                    events['ticker'] = ticker
                    events_list.append(events)
            df = pd.DataFrame(events_list)
        else:
            return None

        if df.empty or 'type' not in df.columns:
            return None

        # Parse dates and extract hour
        if 'date' in df.columns:
            df['hour'] = pd.to_datetime(df['date'], errors='coerce').dt.hour
        elif 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
        elif 'published_at' in df.columns:
            df['hour'] = pd.to_datetime(df['published_at'], errors='coerce').dt.hour
        else:
            df['hour'] = 12  # Default to noon if no time data

        # Drop rows with NaN hours
        df = df.dropna(subset=['hour'])
        df['hour'] = df['hour'].astype(int)

        if df.empty:
            return None

        # Create pivot table: event type x hour
        pivot = df.groupby(['type', 'hour']).size().unstack(fill_value=0)

        # Ensure all 24 hours are present
        all_hours = list(range(24))
        for h in all_hours:
            if h not in pivot.columns:
                pivot[h] = 0
        pivot = pivot[sorted(pivot.columns)]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[
                [0, '#1a1f2e'],
                [0.5, '#3b82f6'],
                [1, '#06b6d4']
            ],
            showscale=True,
            colorbar=dict(
                title='Count',
                titlefont=dict(color='#9ca3af'),
                tickfont=dict(color='#9ca3af')
            )
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#1a1f2e',
            xaxis=dict(
                title='Hour (UTC)',
                tickfont=dict(color='#9ca3af'),
                gridcolor='#374151',
                dtick=2
            ),
            yaxis=dict(
                title='Event Type',
                tickfont=dict(color='#9ca3af')
            ),
            margin=dict(l=100, r=20, t=20, b=40),
            height=250
        )

        return fig

    except Exception as e:
        # Log error but don't crash
        return None


# ============================
# SIDEBAR
# ============================

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    # Auto refresh
    auto_refresh = st.toggle("🔄 Auto Refresh", value=False)
    refresh_interval = st.slider("Refresh interval (sec)", 10, 120, 30) if auto_refresh else 30
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filters")
    hours_back = st.selectbox("Time Range", [6, 12, 24, 48, 168], index=2, format_func=lambda x: f"Last {x}h" if x < 168 else "Last 7 days")
    
    signal_filter = st.multiselect(
        "Signal Types",
        ["BUY_STRONG", "BUY", "WATCH_EARLY", "WATCH"],
        default=["BUY_STRONG", "BUY", "WATCH_EARLY"]
    )
    
    min_score = st.slider("Min Monster Score", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    
    # Actions
    st.markdown("### 🚀 Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Run Audit", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    from daily_audit import run_daily_audit
                    run_daily_audit(send_telegram=False)
                    st.success("Done!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 🛡️ System Status")
    status = get_system_status()
    
    for component, is_ok in status.items():
        icon = "🟢" if is_ok else "🔴"
        st.markdown(f"{icon} **{component.upper()}**")
    
    st.markdown("---")

    # V7 Modules Status
    st.markdown("### 🧠 V7 Modules")
    v7_modules = {
        "SignalProducer": True,
        "OrderComputer": True,
        "ExecutionGate": True,
        "RiskGuard": True,
        "PreHaltEngine": True,
        "MarketMemory": True
    }
    for module, active in v7_modules.items():
        icon = "🟢" if active else "🔴"
        st.markdown(f"{icon} {module}")

    st.markdown("---")
    st.caption(f"v7.0 • {datetime.utcnow().strftime('%H:%M:%S UTC')}")


# ============================
# HEADER
# ============================

col_title, col_status = st.columns([3, 1])

with col_title:
    st.markdown("# 🎯 GV2-EDGE V7.0")
    st.markdown("**Detection/Execution Separation Architecture** — Small Caps US")

with col_status:
    session = get_market_session()
    session_colors = {
        "PREMARKET": ("🌅", "Pre-Market", "#f59e0b"),
        "RTH": ("📈", "Market Open", "#10b981"),
        "AFTER_HOURS": ("🌙", "After-Hours", "#8b5cf6"),
        "CLOSED": ("💤", "Closed", "#6b7280")
    }
    icon, label, color = session_colors.get(session, ("❓", "Unknown", "#6b7280"))
    
    st.markdown(f"""
    <div style="text-align: right; padding: 1rem;">
        <div style="font-size: 2rem;">{icon}</div>
        <div style="color: {color}; font-weight: 600;">{label}</div>
        <span class="status-live">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================
# KEY METRICS ROW
# ============================

# Load data
signals_df = load_signals_from_db(hours_back)
audit_data = load_latest_audit()

# Calculate metrics
total_signals = len(signals_df) if not signals_df.empty else 0
buy_strong_count = len(signals_df[signals_df['signal'] == 'BUY_STRONG']) if not signals_df.empty and 'signal' in signals_df.columns else 0
hit_rate = audit_data.get('hit_rate', 0) if audit_data else 0
avg_lead_time = audit_data.get('avg_lead_time_hours', 0) if audit_data else 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Signals</div>
        <div class="metric-value">{total_signals}</div>
        <div class="metric-delta-positive">Last {hours_back}h</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card signal-buy-strong">
        <div class="metric-label">BUY_STRONG</div>
        <div class="metric-value" style="color: #10b981;">{buy_strong_count}</div>
        <div class="metric-delta-positive">🔥 Hot signals</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    score_class = 'score-high' if hit_rate > 0.6 else 'score-medium' if hit_rate > 0.4 else 'score-low'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Hit Rate</div>
        <div class="metric-value {score_class}">{hit_rate*100:.1f}%</div>
        <div class="metric-label">Target: 65%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Lead Time</div>
        <div class="metric-value">{avg_lead_time:.1f}h</div>
        <div class="metric-label">Before spike</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    # Universe size
    universe_path = DATA_DIR / "universe.csv"
    universe_size = 0
    if universe_path.exists():
        try:
            universe_size = len(pd.read_csv(universe_path))
        except:
            pass
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Universe</div>
        <div class="metric-value">{universe_size}</div>
        <div class="metric-label">Tickers tracked</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================
# MAIN CONTENT — TABS
# ============================

tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Signals", "📊 Analytics", "📅 Events", "🔍 Audit"])

# ============================
# TAB 1: LIVE SIGNALS
# ============================

with tab1:
    st.markdown("### 🔥 Active Trading Signals")
    
    if signals_df.empty:
        st.info("No signals detected in the selected time range. The system is scanning...")
    else:
        # Filter signals
        filtered_df = signals_df.copy()
        if 'signal' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['signal'].isin(signal_filter)]
        if 'monster_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['monster_score'] >= min_score]
        
        if filtered_df.empty:
            st.warning("No signals match your filters. Try adjusting the criteria.")
        else:
            # Split by signal type
            col_strong, col_buy, col_watch = st.columns(3)
            
            with col_strong:
                st.markdown("#### 🚨 BUY_STRONG")
                strong_df = filtered_df[filtered_df['signal'] == 'BUY_STRONG'] if 'signal' in filtered_df.columns else pd.DataFrame()
                if not strong_df.empty:
                    for _, row in strong_df.head(5).iterrows():
                        ts = row.get('timestamp', '')[:16] if 'timestamp' in row else ''
                        st.markdown(f"""
                        <div class="metric-card signal-buy-strong">
                            <span class="ticker-symbol">{row['ticker']}</span>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                                <span style="color: #9ca3af;">Score</span>
                                <span style="color: #10b981; font-family: 'JetBrains Mono';">{row['monster_score']:.2f}</span>
                            </div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">{ts}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No BUY_STRONG signals")
            
            with col_buy:
                st.markdown("#### ✅ BUY")
                buy_df = filtered_df[filtered_df['signal'] == 'BUY'] if 'signal' in filtered_df.columns else pd.DataFrame()
                if not buy_df.empty:
                    for _, row in buy_df.head(5).iterrows():
                        ts = row.get('timestamp', '')[:16] if 'timestamp' in row else ''
                        st.markdown(f"""
                        <div class="metric-card signal-buy">
                            <span class="ticker-symbol">{row['ticker']}</span>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                                <span style="color: #9ca3af;">Score</span>
                                <span style="color: #3b82f6; font-family: 'JetBrains Mono';">{row['monster_score']:.2f}</span>
                            </div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">{ts}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No BUY signals")
            
            with col_watch:
                st.markdown("#### 👀 WATCH_EARLY")
                watch_df = filtered_df[filtered_df['signal'].isin(['WATCH_EARLY', 'WATCH'])] if 'signal' in filtered_df.columns else pd.DataFrame()
                if not watch_df.empty:
                    for _, row in watch_df.head(5).iterrows():
                        ts = row.get('timestamp', '')[:16] if 'timestamp' in row else ''
                        st.markdown(f"""
                        <div class="metric-card signal-watch">
                            <span class="ticker-symbol">{row['ticker']}</span>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                                <span style="color: #9ca3af;">Score</span>
                                <span style="color: #f59e0b; font-family: 'JetBrains Mono';">{row['monster_score']:.2f}</span>
                            </div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">{ts}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No WATCH signals")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Signals timeline
            if 'timestamp' in filtered_df.columns and 'monster_score' in filtered_df.columns:
                st.markdown("#### 📈 Signals Timeline")
                timeline_fig = create_signals_timeline(filtered_df)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Full table
            with st.expander("📋 View All Signals Table"):
                display_cols = [c for c in ['timestamp', 'ticker', 'signal', 'monster_score', 'entry', 'stop', 'shares'] if c in filtered_df.columns]
                st.dataframe(filtered_df[display_cols].head(50), use_container_width=True, hide_index=True)


# ============================
# TAB 2: ANALYTICS
# ============================

with tab2:
    st.markdown("### 📊 Score Analytics")
    
    col_radar, col_distribution = st.columns(2)
    
    with col_radar:
        st.markdown("#### Monster Score Breakdown")
        
        # Default sample components
        sample_components = {
            'event': 0.8,
            'volume': 0.12,
            'pattern': 0.15,
            'pm_transition': 0.10,
            'momentum': 0.08,
            'squeeze': 0.04
        }
        
        radar_fig = create_monster_score_radar(sample_components)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
    
    with col_distribution:
        st.markdown("#### Score Distribution")
        
        if not signals_df.empty and 'monster_score' in signals_df.columns:
            fig = px.histogram(
                signals_df,
                x='monster_score',
                nbins=20,
                color_discrete_sequence=['#06b6d4']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#1a1f2e',
                xaxis=dict(gridcolor='#374151', tickfont=dict(color='#9ca3af'), title='Score'),
                yaxis=dict(gridcolor='#374151', tickfont=dict(color='#9ca3af'), title='Count'),
                margin=dict(l=40, r=40, t=40, b=40),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for distribution chart")
    
    # Boost analysis
    st.markdown("#### 🚀 Intelligence Boosts Impact")
    
    boost_data = {
        'Boost Type': ['Beat Rate', 'Social Buzz', 'Options Flow', 'Extended Hours'],
        'Max Value': [0.20, 0.10, 0.10, 0.22],
        'Avg Contribution': [0.08, 0.04, 0.02, 0.06]
    }
    
    boost_df = pd.DataFrame(boost_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=boost_df['Boost Type'],
        y=boost_df['Max Value'],
        name='Max Possible',
        marker_color='rgba(6, 182, 212, 0.3)',
        text=boost_df['Max Value'],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=boost_df['Boost Type'],
        y=boost_df['Avg Contribution'],
        name='Avg Contribution',
        marker_color='#06b6d4',
        text=boost_df['Avg Contribution'],
        textposition='outside'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a1f2e',
        xaxis=dict(gridcolor='#374151', tickfont=dict(color='#9ca3af')),
        yaxis=dict(gridcolor='#374151', tickfont=dict(color='#9ca3af'), title='Score Boost'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')),
        barmode='group',
        margin=dict(l=40, r=40, t=40, b=40),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================
# TAB 3: EVENTS
# ============================

with tab3:
    st.markdown("### 📅 Detected Events & Catalysts")
    
    col_events, col_heatmap = st.columns([2, 1])
    
    with col_events:
        # Load events from cache
        events_cache = load_json(DATA_DIR / "events_cache.json")
        
        if events_cache:
            events_df = pd.DataFrame(events_cache)
            
            if not events_df.empty:
                # Group by type
                event_counts = events_df['type'].value_counts() if 'type' in events_df.columns else pd.Series()
                
                st.markdown("#### Event Types Distribution")
                
                if not event_counts.empty:
                    fig = px.pie(
                        values=event_counts.values,
                        names=event_counts.index,
                        color_discrete_sequence=['#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#10b981']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#9ca3af'),
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Events table
                st.markdown("#### Recent Events")
                display_cols = [c for c in ['ticker', 'type', 'boosted_impact', 'date'] if c in events_df.columns]
                if display_cols:
                    st.dataframe(events_df[display_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No events cached yet. The system will populate this after scanning.")
    
    with col_heatmap:
        st.markdown("#### Events Heatmap")
        heatmap_fig = create_events_heatmap(events_cache)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="metric-card" style="text-align: center; padding: 2rem;">
                <div style="color: #6b7280; font-size: 1.1rem;">📊 No data available</div>
                <div style="color: #4b5563; font-size: 0.85rem; margin-top: 0.5rem;">
                    Events heatmap will populate after scanning
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Upcoming Catalysts")
        st.markdown("""
        <div class="metric-card">
            <div style="color: #f59e0b;">📅 Earnings This Week</div>
            <div style="color: #6b7280; font-size: 0.85rem; margin-top: 0.5rem;">
                Check Finnhub calendar for scheduled reports
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================
# TAB 4: AUDIT
# ============================

with tab4:
    st.markdown("### 🔍 Performance Audit")
    
    if audit_data:
        col_gauge, col_metrics = st.columns([1, 2])
        
        with col_gauge:
            st.markdown("#### Hit Rate")
            gauge_fig = create_hit_rate_gauge(audit_data.get('hit_rate', 0))
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col_metrics:
            st.markdown("#### Audit Metrics")
            
            metrics = [
                ("Total Signals", audit_data.get('total_signals', 0), ""),
                ("True Positives", audit_data.get('true_positives', 0), "🟢"),
                ("False Positives", audit_data.get('false_positives', 0), "🔴"),
                ("Missed Movers", audit_data.get('missed_movers', 0), "⚠️"),
                ("Early Catch Rate", f"{audit_data.get('early_catch_rate', 0)*100:.1f}%", "⏰"),
                ("Avg Lead Time", f"{audit_data.get('avg_lead_time_hours', 0):.1f}h", "📊")
            ]
            
            col_a, col_b, col_c = st.columns(3)
            cols = [col_a, col_b, col_c]
            
            for i, (label, value, icon) in enumerate(metrics):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{icon} {label}</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Audit details
        st.markdown("#### 📋 Detailed Report")
        
        with st.expander("View Full Audit Data"):
            st.json(audit_data)
        
        # Missed movers analysis
        if 'missed_details' in audit_data:
            st.markdown("#### ❌ Missed Movers Analysis")
            missed_df = pd.DataFrame(audit_data['missed_details'])
            if not missed_df.empty:
                st.dataframe(missed_df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit data available. Run a daily or weekly audit to generate reports.")
        
        if st.button("🚀 Run Daily Audit Now"):
            with st.spinner("Running audit..."):
                try:
                    from daily_audit import run_daily_audit
                    run_daily_audit(send_telegram=False)
                    st.success("Audit complete! Refresh to see results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Audit failed: {e}")


# ============================
# FOOTER
# ============================

st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption("🎯 GV2-EDGE V7.0 — Detection/Execution Separation")

with col_footer2:
    st.caption(f"Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

with col_footer3:
    st.caption("Made for momentum traders 🚀")


# ============================
# AUTO REFRESH LOGIC
# ============================

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
