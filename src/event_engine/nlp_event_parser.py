import json
from datetime import datetime

from config import GROK_API_KEY
from utils.api_guard import safe_post
from utils.logger import get_logger
from utils.data_validator import validate_event

logger = get_logger("NLP_EVENT_PARSER")

GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"


SYSTEM_PROMPT = """
You are a financial event extraction engine for small-cap US stocks.

Extract ALL significant events (bullish AND bearish) from the text.

--- BULLISH EVENTS ---

TIER 1 - CRITICAL BULLISH (impact: 0.90-1.00):
- FDA_APPROVAL: Drug/device approved by FDA
- PDUFA_DECISION: Positive PDUFA decision (FDA deadline met positively)
- BUYOUT_CONFIRMED: Confirmed acquisition/buyout announcement

TIER 2 - HIGH BULLISH (impact: 0.75-0.89):
- FDA_TRIAL_POSITIVE: Positive Phase II/III trial results, met endpoints
- BREAKTHROUGH_DESIGNATION: FDA Breakthrough Therapy designation
- FDA_FAST_TRACK: FDA Fast Track designation granted
- MERGER_ACQUISITION: M&A announcement, takeover bid
- EARNINGS_BEAT_BIG: Earnings beat >20% above estimates
- MAJOR_CONTRACT: Large contract win (>$50M or transformational)

TIER 3 - MEDIUM-HIGH BULLISH (impact: 0.60-0.74):
- GUIDANCE_RAISE: Company raises forward guidance
- EARNINGS_BEAT: Standard earnings beat (<20% above estimates)
- PARTNERSHIP: Strategic partnership/collaboration announced
- PRICE_TARGET_RAISE: Significant price target increase by analyst

TIER 4 - MEDIUM BULLISH (impact: 0.45-0.59):
- ANALYST_UPGRADE: Analyst upgrades rating (Sell→Hold, Hold→Buy)
- SHORT_SQUEEZE_SIGNAL: Short squeeze setup or trigger
- UNUSUAL_VOLUME_NEWS: News explaining unusual volume spike

TIER 5 - SPECULATIVE BULLISH (impact: 0.30-0.44):
- BUYOUT_RUMOR: Unconfirmed buyout/acquisition rumor
- SOCIAL_MEDIA_SURGE: Viral social media attention (WSB, Twitter)
- BREAKING_POSITIVE: Other positive breaking news

--- BEARISH / RISK EVENTS (S5-6) ---
Use NEGATIVE impact values for risk events.

TIER 1 - CRITICAL BEARISH (impact: -0.85 to -1.00):
- FDA_REJECTION: FDA rejects drug/device (Complete Response Letter, CRL)
- DELISTING_RISK: Exchange delisting notice or non-compliance warning

TIER 2 - HIGH BEARISH (impact: -0.65 to -0.84):
- DILUTION_RISK: Share offering, ATM program, toxic financing, shelf registration
- SEC_INVESTIGATION: SEC investigation, subpoena, or enforcement action
- EARNINGS_MISS: Significant earnings miss (>10% below estimates) or guidance cut

TIER 3 - MEDIUM BEARISH (impact: -0.40 to -0.64):
- INSIDER_SELLING: Large insider sell (>$500K, not routine plan)
- ANALYST_DOWNGRADE: Analyst downgrades to Sell or Strong Sell
- GUIDANCE_CUT: Company lowers forward guidance

For each event return a JSON list:

[
 {
  "ticker": "XYZ",
  "type": "EVENT_TYPE",
  "impact": float (-1.0 to 1.0, negative for bearish),
  "date": "YYYY-MM-DD",
  "summary": "short explanation",
  "is_bearish": true or false
 }
]

IMPORTANT:
- Extract BOTH bullish and bearish events from the same text
- FDA_APPROVAL impact ≈ +0.95, ANALYST_UPGRADE ≈ +0.50
- FDA_REJECTION impact ≈ -0.90, DILUTION_RISK ≈ -0.70
- Only output valid JSON. No text outside JSON.
"""


def call_grok(text):
    payload = {
        "model": "grok-3-fast",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    r = safe_post(GROK_ENDPOINT, json=payload, headers=headers)
    # S4-5 FIX: check HTTP status before parsing to avoid confusing error messages
    if r.status_code != 200:
        logger.warning(f"Grok API returned HTTP {r.status_code}: {r.text[:200]}")
        return {}
    return r.json()


def _strip_markdown_json(content: str) -> str:
    """S4-5 FIX: Grok sometimes wraps JSON in markdown fences (```json ... ```).
    Strip those before json.loads() to avoid JSONDecodeError."""
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        content = content.split("\n", 1)[-1]
        # Remove closing fence
        if content.endswith("```"):
            content = content[: content.rfind("```")]
    return content.strip()


def parse_events_from_text(text):
    try:
        response = call_grok(text)

        if not response:
            return []

        content = response["choices"][0]["message"]["content"]

        content = _strip_markdown_json(content)

        events = json.loads(content)

        clean_events = []

        for e in events:
            if validate_event(e):
                clean_events.append(e)

        logger.info(f"NLP extracted {len(clean_events)} events")

        return clean_events

    except Exception as e:
        logger.error(f"NLP parse failed: {e}")
        return []


# ============================
# Batch helper
# ============================

def parse_many_texts(texts):
    all_events = []

    for t in texts:
        ev = parse_events_from_text(t)
        all_events.extend(ev)

    return all_events


if __name__ == "__main__":
    sample = """
    TCGL announces FDA approval for its new cancer drug.
    FEED receives major $200M contract with US government.
    """

    print(parse_events_from_text(sample))
