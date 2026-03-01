"""
NLP LOCAL — Ollama (Llama 3.2:3b) pour tâches BATCH
====================================================

Utilisé UNIQUEMENT pour les tâches hors temps réel :
  - weekend_mode (scanning batch)
  - daily_audit / weekly_deep_audit
  - nlp_classifier (batch classification)

Pipeline batch NLP :
  Ollama local (Llama 3.2:3b) → Groq API (fallback) → keyword fallback

Le pipeline REALTIME (nlp_enrichi.py, nlp_event_parser.py) n'est PAS touché.
"""

import json
import time
import requests
from typing import Optional, Dict

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, ENABLE_OLLAMA_BATCH
from utils.logger import get_logger

logger = get_logger("NLP_LOCAL")

_OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/v1/chat/completions"
_OLLAMA_HEALTH_URL = f"{OLLAMA_BASE_URL}/api/tags"

# Circuit breaker : évite de tenter Ollama si down
_ollama_down_until: float = 0.0
_OLLAMA_COOLDOWN = 120  # 2 min avant de réessayer après échec


def is_ollama_available() -> bool:
    """Ping rapide pour savoir si Ollama tourne."""
    global _ollama_down_until
    if time.time() < _ollama_down_until:
        return False
    try:
        r = requests.get(_OLLAMA_HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        _ollama_down_until = time.time() + _OLLAMA_COOLDOWN
        return False


def _parse_ollama_response(result: dict) -> Optional[Dict]:
    """Extrait et parse le JSON depuis la réponse Ollama (format OpenAI-compatible)."""
    try:
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception:
        return None


def call_ollama(prompt: str, text: str, temperature: float = 0.1) -> Optional[Dict]:
    """
    Appel Ollama (Llama 3.2:3b) en local — BATCH uniquement.
    Retourne un dict JSON parsé, ou None si indisponible.
    """
    global _ollama_down_until

    if not ENABLE_OLLAMA_BATCH:
        return None

    if not is_ollama_available():
        return None

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",   "content": text},
            ],
            "temperature": temperature,
            "stream": False,
        }
        t0 = time.time()
        r = requests.post(_OLLAMA_CHAT_URL, json=payload, timeout=60)
        elapsed = int((time.time() - t0) * 1000)

        if r.status_code != 200:
            logger.warning(f"Ollama HTTP {r.status_code} ({elapsed}ms)")
            _ollama_down_until = time.time() + _OLLAMA_COOLDOWN
            return None

        result = _parse_ollama_response(r.json())
        if result is not None:
            logger.debug(f"Ollama OK ({elapsed}ms, model={OLLAMA_MODEL})")
        return result

    except requests.exceptions.Timeout:
        logger.warning("Ollama timeout (>60s) — circuit breaker 2min")
        _ollama_down_until = time.time() + _OLLAMA_COOLDOWN
        return None
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        _ollama_down_until = time.time() + _OLLAMA_COOLDOWN
        return None


def call_batch_nlp(prompt: str, text: str, temperature: float = 0.1) -> Optional[Dict]:
    """
    NLP pour tâches BATCH : Ollama → Groq → None.
    Interface identique à _call_nlp() du pipeline realtime.
    """
    # 1. Ollama local (gratuit, 0 latence réseau)
    result = call_ollama(prompt, text, temperature)
    if result is not None:
        return result

    # 2. Groq API (fallback)
    try:
        from src.nlp_enrichi import _call_groq
        result = _call_groq(prompt, text, temperature)
        if result is not None:
            logger.debug("Batch NLP: fallback vers Groq API")
            return result
    except Exception as e:
        logger.debug(f"Groq fallback error: {e}")

    return None


def get_ollama_status() -> dict:
    """Retourne l'état d'Ollama (pour dashboard)."""
    global _ollama_down_until
    available = is_ollama_available()
    status = {
        "available": available,
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "enabled": ENABLE_OLLAMA_BATCH,
        "circuit_breaker_until": _ollama_down_until if _ollama_down_until > time.time() else None,
    }
    if available:
        # Essaie de récupérer la liste des modèles téléchargés
        try:
            r = requests.get(_OLLAMA_HEALTH_URL, timeout=3)
            data = r.json()
            status["models_pulled"] = [m.get("name") for m in data.get("models", [])]
            status["model_ready"] = any(
                OLLAMA_MODEL in m.get("name", "") for m in data.get("models", [])
            )
        except Exception:
            status["models_pulled"] = []
            status["model_ready"] = False
    return status
