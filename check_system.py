#!/usr/bin/env python3
"""
GV2-EDGE V9.0 — DIAGNOSTIC SYSTÈME COMPLET
==========================================

Lance une vérification complète du système en conditions réelles :
  Phase 1 — Imports modules critiques
  Phase 2 — APIs externes (Finnhub, Grok, Telegram, SEC EDGAR)
  Phase 3 — IBKR Gateway (connectivité)
  Phase 4 — Pipeline mini-cycle (1 ticker réel bout en bout)
  Phase 5 — Bases de données SQLite (intégrité)
  Phase 6 — Logs récents (erreurs, warnings)
  Phase 7 — Ressources système (CPU, RAM, disque)

Usage:
    python check_system.py              # Vérification complète
    python check_system.py --quick      # Seulement imports + APIs (30s)
    python check_system.py --ticker GME # Ticker pour le mini-cycle (défaut: AAPL)
"""

import os
import sys
import time
import json
import sqlite3
import socket
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# Force UTF-8 on Windows terminals (script targets Linux/Hetzner)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Assurer que le répertoire du projet est dans le path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Couleurs terminal ───────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK    = f"{GREEN}[OK]{RESET}"
FAIL  = f"{RED}[FAIL]{RESET}"
WARN  = f"{YELLOW}[WARN]{RESET}"
INFO  = f"{BLUE}[INFO]{RESET}"

results = {"ok": 0, "fail": 0, "warn": 0}


def _ok(msg: str):
    results["ok"] += 1
    print(f"  {OK}  {msg}")


def _fail(msg: str, detail: str = ""):
    results["fail"] += 1
    print(f"  {FAIL} {msg}")
    if detail:
        print(f"         {YELLOW}{detail}{RESET}")


def _warn(msg: str, detail: str = ""):
    results["warn"] += 1
    print(f"  {WARN} {msg}")
    if detail:
        print(f"         {detail}")


def _header(title: str):
    print(f"\n{BOLD}{BLUE}{'─'*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*60}{RESET}")


def _check(label: str, fn, warn_only=False) -> bool:
    """Exécute fn(), retourne True si OK."""
    try:
        fn()
        _ok(label)
        return True
    except Exception as e:
        msg = str(e).split("\n")[0][:120]
        if warn_only:
            _warn(label, msg)
            return False
        else:
            _fail(label, msg)
            return False


# ============================================================================
# PHASE 1 — IMPORTS MODULES CRITIQUES
# ============================================================================

def phase_imports():
    _header("Phase 1 — Imports modules critiques")

    # (module_path, attribute_to_check, warn_only)
    modules = [
        # Modèles de base
        ("src.models.signal_types",          "SignalType",           False),
        ("src.models.signal_types",          "UnifiedSignal",        False),
        ("src.engines.signal_producer",       "DetectionInput",       False),
        # Engines principaux
        ("src.engines.signal_producer",      "get_signal_producer",  False),
        ("src.engines.order_computer",       "OrderComputer",        False),
        ("src.engines.execution_gate",       "ExecutionGate",        False),
        ("src.engines.acceleration_engine",  "get_acceleration_engine", False),
        ("src.engines.smallcap_radar",       "SmallCapRadar",        False),
        ("src.engines.ticker_state_buffer",  "TickerStateBuffer",    False),
        ("src.engines.multi_radar_engine",   "get_multi_radar_engine", False),
        # Scoring
        ("src.scoring.monster_score",        None,                   False),
        # Risk Guard
        ("src.risk_guard.unified_guard",     None,                   False),
        ("src.risk_guard.dilution_detector", None,                   False),
        ("src.risk_guard.compliance_checker", None,                  False),
        ("src.risk_guard.halt_monitor",      None,                   False),
        # Event engine
        ("src.event_engine.event_hub",       None,                   False),
        ("src.event_engine.nlp_event_parser", None,                  False),
        # Ingestors
        ("src.ingestors.sec_filings_ingestor", None,                 False),
        ("src.ingestors.global_news_ingestor", None,                 False),
        # Schedulers
        ("src.schedulers.hot_ticker_queue",  None,                   False),
        ("src.schedulers.scan_scheduler",    None,                   False),
        # Boosters
        ("src.boosters.insider_boost",       None,                   False),
        ("src.boosters.squeeze_boost",       None,                   False),
        # Alertes + utils
        ("alerts.telegram_alerts",           None,                   False),
        ("utils.logger",                     "get_logger",           False),
        ("utils.time_utils",                 None,                   False),
        ("utils.api_guard",                  None,                   False),
        ("utils.cache",                      "Cache",                False),
        # Market memory
        ("src.market_memory.context_scorer", None,                   False),
        ("src.market_memory.memory_store",   None,                   False),
        # Modules optionnels
        ("src.ibkr_connector",               "get_ibkr",             True),
        ("src.ibkr_streaming",               None,                   True),
        ("src.finnhub_ws_screener",          None,                   True),
        ("src.signal_logger",                "init_db",              False),
        ("src.feature_engine",               None,                   False),
        ("src.anticipation_engine",          None,                   False),
        ("src.fda_calendar",                 None,                   True),
        ("src.repeat_gainer_memory",         None,                   True),
        ("src.social_buzz",                  None,                   True),
        ("src.top_gainers_source",           None,                   True),
    ]

    for mod_path, attr, warn_only in modules:
        label = f"{mod_path}" + (f".{attr}" if attr else "")
        try:
            m = __import__(mod_path, fromlist=[attr] if attr else [""])
            if attr:
                getattr(m, attr)
            _ok(label)
        except Exception as e:
            msg = str(e).split("\n")[0][:100]
            if warn_only:
                _warn(label, msg)
            else:
                _fail(label, msg)


# ============================================================================
# PHASE 2 — APIs EXTERNES
# ============================================================================

def phase_apis():
    _header("Phase 2 — Connectivité APIs externes")

    try:
        import config
        finnhub_key = config.FINNHUB_API_KEY
        grok_key    = config.GROK_API_KEY
        tg_token    = config.TELEGRAM_BOT_TOKEN
        tg_chat     = config.TELEGRAM_CHAT_ID
    except Exception as e:
        _fail("config.py chargement", str(e))
        return

    # Finnhub
    def check_finnhub():
        import requests
        r = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": "AAPL", "token": finnhub_key},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        price = data.get("c", 0)
        if price <= 0:
            raise ValueError(f"Prix invalide reçu: {price}")
        print(f"         AAPL price = ${price:.2f}")

    _check("Finnhub REST API (quote AAPL)", check_finnhub)

    # Grok / xAI
    def check_grok():
        if not grok_key:
            raise ValueError("GROK_API_KEY non définie dans .env")
        import requests
        r = requests.get(
            "https://api.x.ai/v1/models",
            headers={"Authorization": f"Bearer {grok_key}"},
            timeout=8
        )
        r.raise_for_status()
        models = r.json().get("data", [])
        names = [m["id"] for m in models[:3]]
        print(f"         Modèles disponibles: {names}")

    _check("Grok / xAI API (list models)", check_grok, warn_only=not grok_key)

    # Telegram
    def check_telegram():
        if not tg_token:
            raise ValueError("TELEGRAM_BOT_TOKEN non défini dans .env")
        import requests
        r = requests.get(
            f"https://api.telegram.org/bot{tg_token}/getMe",
            timeout=8
        )
        r.raise_for_status()
        bot = r.json()["result"]["username"]
        print(f"         Bot: @{bot}")

    _check("Telegram Bot API", check_telegram, warn_only=not tg_token)

    # SEC EDGAR
    def check_sec():
        import requests
        r = requests.get(
            "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom"
            "&startdt=2024-01-01&enddt=2024-01-02&hits.hits.total.value=true",
            headers={"User-Agent": "GV2-EDGE check@gv2.io"},
            timeout=10
        )
        r.raise_for_status()
        print(f"         SEC EDGAR HTTP {r.status_code}")

    _check("SEC EDGAR (EDGAR full-text search)", check_sec)

    # Finnhub WebSocket (juste ping DNS/TCP)
    def check_finnhub_ws():
        s = socket.create_connection(("ws.finnhub.io", 443), timeout=5)
        s.close()
        print(f"         TCP:443 ws.finnhub.io → OK")

    _check("Finnhub WebSocket (TCP ping)", check_finnhub_ws)


# ============================================================================
# PHASE 3 — IBKR GATEWAY
# ============================================================================

def phase_ibkr():
    _header("Phase 3 — IBKR Gateway")

    try:
        import config
        host = config.IBKR_HOST
        port = config.IBKR_PORT
    except Exception:
        host, port = "127.0.0.1", 7497

    # TCP ping
    def check_tcp():
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        print(f"         TCP {host}:{port} → connecté")

    tcp_ok = _check(f"IBKR Gateway TCP {host}:{port}", check_tcp, warn_only=True)

    if not tcp_ok:
        _warn("IBKR Gateway injoignable — les données IBKR utiliseront Finnhub comme fallback")
        return

    # Tentative connexion ib_insync
    def check_ib_insync():
        from ib_insync import IB
        ib = IB()
        try:
            import config as cfg
            ib.connect(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=99, timeout=10, readonly=True)
            version = ib.client.serverVersion()
            print(f"         IB Gateway serverVersion={version}")
            # Tester un quote rapide
            from ib_insync import Stock
            contract = Stock("AAPL", "SMART", "USD")
            ib.qualifyContracts(contract)
            [ticker] = ib.reqTickers(contract)
            ib.sleep(1)
            print(f"         AAPL last={ticker.last}  bid={ticker.bid}  ask={ticker.ask}")
        finally:
            ib.disconnect()

    _check("IBKR ib_insync connexion + quote AAPL", check_ib_insync, warn_only=True)


# ============================================================================
# PHASE 4 — PIPELINE MINI-CYCLE
# ============================================================================

def phase_pipeline(ticker: str = "AAPL"):
    _header(f"Phase 4 — Pipeline mini-cycle [{ticker}]")

    import config

    # Étape 1: AccelerationEngine
    def check_accel():
        from src.engines.acceleration_engine import get_acceleration_engine
        accel = get_acceleration_engine()
        # score() lit le buffer interne — retourne DORMANT si pas de données IBKR,
        # ce qui est normal sans Gateway connecté.
        result = accel.score(ticker)
        print(f"         state={result.state}  samples={result.samples}  "
              f"vol_z={result.volume_zscore:.2f}")

    _check("AccelerationEngine.score()", check_accel)

    # Étape 2: SmallCapRadar
    def check_smallcap():
        from src.engines.smallcap_radar import SmallCapRadar
        radar = SmallCapRadar()
        # scan() parcourt tous les tickers du buffer interne (0 si pas d'IBKR)
        result = radar.scan()
        print(f"         tickers_scanned={result.tickers_scanned}  "
              f"critical={len(result.critical)}  high={len(result.high)}")

    _check("SmallCapRadar.scan()", check_smallcap)

    # Étape 3: Monster Score
    def check_monster():
        from src.scoring.monster_score import compute_monster_score
        score = compute_monster_score(ticker)
        if score is None:
            raise ValueError("compute_monster_score returned None")
        print(f"         monster_score={score:.3f}")

    _check("Monster Score compute_monster_score()", check_monster, warn_only=True)

    # Étape 4: SignalProducer.detect() [async]
    def check_signal_producer():
        import asyncio
        from src.engines.signal_producer import get_signal_producer, DetectionInput
        from src.models.signal_types import PreSpikeState, PreHaltState
        from src.engines.acceleration_engine import get_acceleration_engine

        accel_result = get_acceleration_engine().score(ticker)

        inp = DetectionInput(
            ticker=ticker,
            monster_score=0.72,
            event_score=0.60,
            has_catalyst=True,
            volume_zscore=2.8,
            price_change_pct=4.2,
            pre_spike_state=PreSpikeState.CHARGING,
            pre_halt_state=PreHaltState.LOW,
            accel_state=accel_result.state,
            accel_score=accel_result.acceleration_score,
            vol_zscore_accel=accel_result.volume_zscore,
            repeat_gainer_score=0.0,
            social_buzz_score=0.3,
        )
        producer = get_signal_producer()

        async def _run():
            return await producer.detect(inp)

        signal = asyncio.run(_run())
        print(f"         signal={signal.signal_type.value}  score={signal.final_score:.3f}")

    _check("SignalProducer.detect() [async]", check_signal_producer)

    # Étape 5: OrderComputer
    def check_order_computer():
        from src.engines.order_computer import OrderComputer
        from src.models.signal_types import SignalType
        import config as cfg

        capital = getattr(cfg, "MANUAL_CAPITAL", getattr(cfg, "TRADING_CAPITAL", 1000))
        computer = OrderComputer()
        order = computer.compute_order(
            ticker=ticker,
            signal_type=SignalType.BUY,
            current_price=180.0,
            atr=2.5,
            available_capital=capital * 0.8,
        )
        if order:
            print(f"         shares={order.shares}  entry={order.entry_price:.2f}"
                  f"  stop={order.stop_loss:.2f}  R/R={order.risk_reward:.1f}")
        else:
            print(f"         order=None (capital ou filtre)")

    _check("OrderComputer.compute_order()", check_order_computer)

    # Étape 6: UnifiedGuard [async]
    def check_risk_guard():
        import asyncio
        from src.risk_guard.unified_guard import get_unified_guard

        guard = get_unified_guard()

        async def _run():
            return await guard.assess(ticker, price=180.0, volume=800_000)

        assessment = asyncio.run(_run())
        print(f"         risk_level={assessment.risk_level}  "
              f"action={assessment.recommended_action}  "
              f"size_mult={assessment.size_multiplier:.2f}")

    _check("UnifiedGuard.assess() [async]", check_risk_guard)

    # Étape 7: ExecutionGate
    def check_execution_gate():
        from src.engines.execution_gate import ExecutionGate
        from src.engines.order_computer import OrderComputer
        from src.models.signal_types import SignalType
        import config as cfg

        capital = getattr(cfg, "MANUAL_CAPITAL", getattr(cfg, "TRADING_CAPITAL", 1000))
        computer = OrderComputer()
        order = computer.compute_order(
            ticker=ticker,
            signal_type=SignalType.BUY,
            current_price=180.0,
            atr=2.5,
            available_capital=capital * 0.8,
        )
        gate = ExecutionGate()
        decision = gate.evaluate(order, daily_trades=0)
        print(f"         status={decision.status.value}  "
              f"reason={decision.block_reason.value if decision.block_reason else 'None'}")

    _check("ExecutionGate.evaluate()", check_execution_gate)

    # Étape 8: Multi-Radar [async]
    def check_multi_radar():
        import asyncio
        from src.engines.multi_radar_engine import get_multi_radar_engine

        engine = get_multi_radar_engine()

        async def _run():
            return await engine.scan_ticker(ticker)

        result = asyncio.run(_run())
        print(f"         signal={result.signal_type.value}  "
              f"flow={result.flow_score:.2f}  "
              f"catalyst={result.catalyst_score:.2f}  "
              f"lead={result.lead_radar}")

    _check("MultiRadarEngine.scan_ticker() [async]", check_multi_radar)

    # Étape 9: Feature Engine
    def check_feature_engine():
        from src.feature_engine import compute_features
        features = compute_features(ticker)
        if features:
            keys = list(features.keys())[:5]
            print(f"         {len(features)} features  ex: {keys}")
        else:
            raise ValueError("features est None ou vide (normal sans Finnhub key)")

    _check("FeatureEngine.compute_features()", check_feature_engine, warn_only=True)

    # Étape 10: Signal Logger (DB write)
    def check_signal_logger():
        from src.signal_logger import init_db, log_signal
        init_db()
        log_signal({
            "ticker": ticker,
            "signal": "WATCH",
            "monster_score": 0.55,
            "confidence": 0.55,
            "metadata": json.dumps({"check": "diagnostic"}),
        })
        print(f"         Signal WATCH écrit dans signals_history.db")

    _check("SignalLogger.log_signal() → SQLite", check_signal_logger)


# ============================================================================
# PHASE 5 — BASES DE DONNÉES SQLite
# ============================================================================

def phase_databases():
    _header("Phase 5 — Bases de données SQLite")

    db_paths = [
        ("data/signals_history.db",          "signals"),
        ("data/repeat_gainers.db",            None),
        ("data/market_memory.db",             None),
        ("data/nlp_cache.db",                 None),
        ("data/catalyst_cache.db",            None),
    ]

    for db_path, table in db_paths:
        full = PROJECT_ROOT / db_path
        if not full.exists():
            _warn(f"{db_path}", "Fichier absent (normal si jamais démarré)")
            continue

        def check_db(path=full, tbl=table):
            conn = sqlite3.connect(str(path), check_same_thread=False)
            try:
                # Intégrité
                res = conn.execute("PRAGMA integrity_check").fetchone()
                if res[0] != "ok":
                    raise ValueError(f"Integrity check: {res[0]}")
                size_kb = path.stat().st_size / 1024
                rows = ""
                if tbl:
                    try:
                        cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                        rows = f"  {cnt} lignes dans '{tbl}'"
                    except Exception:
                        pass
                print(f"         {size_kb:.1f} KB, integrity=ok{rows}")
            finally:
                conn.close()

        _check(f"SQLite {db_path}", check_db)


# ============================================================================
# PHASE 6 — LOGS RÉCENTS (erreurs)
# ============================================================================

def phase_logs():
    _header("Phase 6 — Analyse logs récents (dernières 500 lignes)")

    log_dir = PROJECT_ROOT / "data" / "logs"
    if not log_dir.exists():
        _warn("data/logs/ absent", "Aucun log généré")
        return

    critical_logs = [
        "signal_producer.log",
        "multi_radar.log",
        "execution_gate.log",
        "order_computer.log",
    ]

    for log_name in critical_logs:
        log_path = log_dir / log_name
        if not log_path.exists():
            _warn(f"{log_name}", "Absent")
            continue

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            last_500 = lines[-500:]
            errors   = [l.strip() for l in last_500 if " ERROR " in l or " CRITICAL " in l]
            warnings = [l.strip() for l in last_500 if " WARNING " in l]

            size_kb = log_path.stat().st_size / 1024
            if errors:
                _fail(
                    f"{log_name} ({size_kb:.0f} KB) — {len(errors)} erreur(s)",
                    errors[-1][:120]   # dernière erreur
                )
            elif len(warnings) > 20:
                _warn(
                    f"{log_name} ({size_kb:.0f} KB) — {len(warnings)} warnings",
                    warnings[-1][:120]
                )
            else:
                _ok(f"{log_name} ({size_kb:.0f} KB) — {len(warnings)} warnings, 0 erreur")
        except Exception as e:
            _warn(f"{log_name}", str(e)[:80])


# ============================================================================
# PHASE 7 — RESSOURCES SYSTÈME
# ============================================================================

def phase_system():
    _header("Phase 7 — Ressources système")

    # psutil
    def check_psutil():
        import psutil
        cpu  = psutil.cpu_percent(interval=1)
        ram  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        ram_used  = ram.used  / (1024**3)
        ram_total = ram.total / (1024**3)
        disk_free = disk.free / (1024**3)

        print(f"         CPU:  {cpu:.1f}%")
        print(f"         RAM:  {ram_used:.1f} / {ram_total:.1f} GB ({ram.percent:.1f}%)")
        print(f"         Disk: {disk_free:.1f} GB libres ({disk.percent:.1f}% utilisé)")

        if cpu > 85:
            raise ValueError(f"CPU critique: {cpu:.0f}%")
        if ram.percent > 90:
            raise ValueError(f"RAM critique: {ram.percent:.0f}%")
        if disk.percent > 90:
            raise ValueError(f"Disque critique: {disk.percent:.0f}%")

    _check("CPU / RAM / Disque (seuils: <85% / <90% / <90%)", check_psutil, warn_only=True)

    # Espace data/
    def check_data_dir():
        data_dir = PROJECT_ROOT / "data"
        if not data_dir.exists():
            _warn("data/ absent", "Sera créé au premier démarrage")
            return
        total_mb = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file()) / 1024**2
        print(f"         data/ = {total_mb:.1f} MB")
        if total_mb > 5000:
            raise ValueError(f"data/ > 5 GB — penser à nettoyer les vieux logs")

    _check("Taille data/ (seuil: <5 GB)", check_data_dir, warn_only=True)

    # Timezone
    def check_tz():
        from utils.time_utils import market_session, is_market_open, is_premarket, is_after_hours
        session = market_session()
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")
        open_   = is_market_open()
        pm      = is_premarket()
        ah      = is_after_hours()
        print(f"         Heure: {now_utc}  Session: {session}  "
              f"open={open_}  premarket={pm}  afterhours={ah}")

    _check("Heure UTC + session marché", check_tz)

    # Python version
    def check_python():
        v = sys.version_info
        if v < (3, 11):
            raise ValueError(f"Python {v.major}.{v.minor} — requis 3.11+")
        print(f"         Python {v.major}.{v.minor}.{v.micro} — OK")

    _check("Python >= 3.11", check_python)


# ============================================================================
# RAPPORT FINAL
# ============================================================================

def print_summary():
    total = results["ok"] + results["fail"] + results["warn"]
    print(f"\n{'='*60}")
    print(f"{BOLD}  BILAN DIAGNOSTIC GV2-EDGE V9.0{RESET}")
    print(f"{'='*60}")
    print(f"  {GREEN}{results['ok']:3d} OK{RESET}   {RED}{results['fail']:3d} FAIL{RESET}"
          f"   {YELLOW}{results['warn']:3d} WARN{RESET}   ({total} checks total)")
    print()

    if results["fail"] == 0 and results["warn"] == 0:
        print(f"  {GREEN}{BOLD}Système opérationnel à 100% — prêt pour le marché.{RESET}")
    elif results["fail"] == 0:
        print(f"  {YELLOW}{BOLD}Système opérationnel avec {results['warn']} avertissement(s).{RESET}")
        print(f"  Les WARNs sont souvent des modules optionnels ou des APIs non configurées.")
    elif results["fail"] <= 3:
        print(f"  {YELLOW}{BOLD}{results['fail']} point(s) critique(s) à corriger.{RESET}")
        print(f"  Relancez avec la sortie complète pour identifier les FAIL.")
    else:
        print(f"  {RED}{BOLD}{results['fail']} FAIL — vérifier les dépendances et variables d'env.{RESET}")

    print(f"\n  Lancé le : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GV2-EDGE V9.0 — Diagnostic système")
    parser.add_argument("--quick",  action="store_true", help="Seulement phases 1+2 (imports + APIs)")
    parser.add_argument("--ticker", default="AAPL",      help="Ticker pour le mini-cycle (défaut: AAPL)")
    parser.add_argument("--no-ibkr", action="store_true", help="Sauter la vérification IBKR")
    args = parser.parse_args()

    print(f"\n{BOLD}GV2-EDGE V9.0 — DIAGNOSTIC SYSTÈME{RESET}")
    print(f"Ticker test : {args.ticker}")
    print(f"Mode        : {'QUICK' if args.quick else 'COMPLET'}")

    t0 = time.time()

    phase_imports()
    phase_apis()

    if not args.quick:
        if not args.no_ibkr:
            phase_ibkr()
        phase_pipeline(args.ticker)
        phase_databases()
        phase_logs()
        phase_system()

    elapsed = time.time() - t0
    print(f"\n  Durée : {elapsed:.1f}s")
    print_summary()

    # Exit code non-zéro si des FAIL
    sys.exit(1 if results["fail"] > 0 else 0)


if __name__ == "__main__":
    main()
