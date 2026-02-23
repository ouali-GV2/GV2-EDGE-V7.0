"""
AUTO-TUNING WEIGHTS — GV2-EDGE V9 (A16)
==========================================

Optimisation automatique des poids du Monster Score basee sur les resultats.
Remplace score_optimizer.py (mort, zero imports).

Methode: Bayesian-like optimization sur historique 30 jours
- Maximise: detection rate des vrais top gainers
- Minimise: faux positifs (score > 0.65 mais pas de move)
- Contrainte: sum(weights) = 1.0, 0.01 <= weight <= 0.40

Frequence: Hebdomadaire (dimanche batch)
"""

import json
import logging
import threading
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("WEIGHT_OPTIMIZER")


# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_FILE = "data/monster_weights_history.json"
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.40
NUM_CANDIDATES = 50          # Candidats par iteration
NUM_ITERATIONS = 20          # Iterations d'optimisation
MUTATION_RATE = 0.15         # Taux de mutation des poids
EVAL_LOOKBACK_DAYS = 30      # Historique de 30 jours

# Default V4 weights
DEFAULT_WEIGHTS = {
    "event": 0.25,
    "volume": 0.17,
    "pattern": 0.17,
    "pm_transition": 0.13,
    "options_flow": 0.10,
    "acceleration": 0.07,
    "momentum": 0.04,
    "squeeze": 0.04,
    "social_buzz": 0.03,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WeightSet:
    """Un jeu de poids avec ses metriques de performance."""
    weights: Dict[str, float]
    detection_rate: float = 0.0
    precision: float = 0.0
    false_positive_rate: float = 0.0
    fitness: float = 0.0
    evaluated_on: Optional[datetime] = None

    def validate(self) -> bool:
        """Verifie que les poids sont valides."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            return False
        for w in self.weights.values():
            if w < MIN_WEIGHT or w > MAX_WEIGHT:
                return False
        return True


@dataclass
class OptimizationResult:
    """Resultat d'une optimisation."""
    best_weights: Dict[str, float]
    best_fitness: float
    detection_rate: float
    precision: float
    iterations: int
    candidates_evaluated: int
    previous_weights: Dict[str, float]
    improvement_pct: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Weight Optimizer
# ============================================================================

class WeightOptimizer:
    """
    Optimisation des poids Monster Score.

    Algorithme:
    1. Charge l'historique des signaux (30 jours)
    2. Genere des candidats (mutations du jeu actuel)
    3. Evalue chaque candidat sur l'historique
    4. Selectionne le meilleur jeu de poids
    5. Sauvegarde dans l'historique

    Fitness = detection_rate * 0.6 + precision * 0.4
    """

    def __init__(self):
        self._current_weights = dict(DEFAULT_WEIGHTS)
        self._history: List[WeightSet] = []
        self._lock = threading.Lock()
        self._load_history()
        logger.info("WeightOptimizer initialized")

    def _load_history(self) -> None:
        """Charge l'historique des poids."""
        try:
            path = Path(WEIGHTS_FILE)
            if path.exists():
                data = json.loads(path.read_text())
                self._history = []
                for item in data.get("history", []):
                    ws = WeightSet(
                        weights=item["weights"],
                        detection_rate=item.get("detection_rate", 0),
                        precision=item.get("precision", 0),
                        fitness=item.get("fitness", 0),
                    )
                    self._history.append(ws)

                if data.get("current"):
                    self._current_weights = data["current"]
                    logger.info(f"Loaded weights from history: {self._current_weights}")
        except Exception as e:
            logger.debug(f"Could not load weight history: {e}")

    def _save_history(self) -> None:
        """Sauvegarde l'historique."""
        try:
            path = Path(WEIGHTS_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "current": self._current_weights,
                "history": [
                    {
                        "weights": ws.weights,
                        "detection_rate": ws.detection_rate,
                        "precision": ws.precision,
                        "fitness": ws.fitness,
                    }
                    for ws in self._history[-50:]  # Garder les 50 derniers
                ],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not save weight history: {e}")

    def get_current_weights(self) -> Dict[str, float]:
        """Retourne les poids actuels."""
        with self._lock:
            return dict(self._current_weights)

    def get_weight_history(self) -> List[Dict]:
        """Retourne l'historique des optimisations."""
        return [
            {
                "weights": ws.weights,
                "fitness": ws.fitness,
                "detection_rate": ws.detection_rate,
                "precision": ws.precision,
            }
            for ws in self._history
        ]

    def optimize_weekly(self, signal_history: Optional[List[Dict]] = None) -> OptimizationResult:
        """
        Optimisation hebdomadaire des poids.

        Args:
            signal_history: Liste de signaux historiques avec resultats.
                Chaque entry: {
                    "ticker": str,
                    "monster_score": float,
                    "components": {name: raw_score},
                    "was_top_gainer": bool,  # Le ticker a-t-il fait +30%?
                    "signal_type": str,      # BUY, BUY_STRONG, etc.
                    "actual_move_pct": float, # Mouvement reel
                }

        Returns:
            OptimizationResult
        """
        if not signal_history:
            signal_history = self._load_signal_history()

        if len(signal_history) < 20:
            logger.warning(f"Not enough history ({len(signal_history)} signals) — skipping optimization")
            return OptimizationResult(
                best_weights=self._current_weights,
                best_fitness=0,
                detection_rate=0,
                precision=0,
                iterations=0,
                candidates_evaluated=0,
                previous_weights=self._current_weights,
                improvement_pct=0,
            )

        previous_weights = dict(self._current_weights)
        previous_fitness = self._evaluate_weights(self._current_weights, signal_history)

        best_weights = dict(self._current_weights)
        best_fitness = previous_fitness

        total_evaluated = 0

        for iteration in range(NUM_ITERATIONS):
            # Generer des candidats
            candidates = self._generate_candidates(best_weights, NUM_CANDIDATES)

            for candidate in candidates:
                fitness = self._evaluate_weights(candidate, signal_history)
                total_evaluated += 1

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = dict(candidate)

        # Evaluer les metriques du meilleur jeu
        metrics = self._compute_metrics(best_weights, signal_history)

        improvement = ((best_fitness - previous_fitness) / max(0.01, previous_fitness)) * 100

        # Appliquer si amelioration significative
        if improvement > 1.0:
            with self._lock:
                self._current_weights = dict(best_weights)

            ws = WeightSet(
                weights=best_weights,
                detection_rate=metrics["detection_rate"],
                precision=metrics["precision"],
                fitness=best_fitness,
                evaluated_on=datetime.now(timezone.utc),
            )
            self._history.append(ws)
            self._save_history()

            logger.info(
                f"WEIGHTS OPTIMIZED: fitness {previous_fitness:.4f} -> {best_fitness:.4f} "
                f"(+{improvement:.1f}%) | DR={metrics['detection_rate']:.2f} P={metrics['precision']:.2f}"
            )
        else:
            logger.info(f"No improvement found (best: {best_fitness:.4f}, current: {previous_fitness:.4f})")

        return OptimizationResult(
            best_weights=best_weights,
            best_fitness=best_fitness,
            detection_rate=metrics["detection_rate"],
            precision=metrics["precision"],
            iterations=NUM_ITERATIONS,
            candidates_evaluated=total_evaluated,
            previous_weights=previous_weights,
            improvement_pct=round(improvement, 2),
        )

    def _generate_candidates(self, base: Dict[str, float], count: int) -> List[Dict[str, float]]:
        """Genere des candidats par mutation."""
        candidates = []

        for _ in range(count):
            candidate = dict(base)

            # Muter 1-3 poids
            keys = list(candidate.keys())
            n_mutations = random.randint(1, 3)
            mutate_keys = random.sample(keys, min(n_mutations, len(keys)))

            for key in mutate_keys:
                delta = random.gauss(0, MUTATION_RATE)
                candidate[key] = max(MIN_WEIGHT, min(MAX_WEIGHT, candidate[key] + delta))

            # Normaliser pour que sum = 1.0
            total = sum(candidate.values())
            if total > 0:
                candidate = {k: v / total for k, v in candidate.items()}

            candidates.append(candidate)

        return candidates

    def _evaluate_weights(self, weights: Dict[str, float], history: List[Dict]) -> float:
        """
        Evalue un jeu de poids sur l'historique.

        Fitness = detection_rate * 0.6 + precision * 0.4
        """
        metrics = self._compute_metrics(weights, history)
        return metrics["detection_rate"] * 0.6 + metrics["precision"] * 0.4

    def _compute_metrics(self, weights: Dict[str, float], history: List[Dict]) -> Dict:
        """
        Calcule les metriques pour un jeu de poids.

        Returns:
            {detection_rate, precision, false_positive_rate}
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for entry in history:
            components = entry.get("components", {})
            was_top = entry.get("was_top_gainer", False)

            # Recalculer le score avec les nouveaux poids
            score = 0.0
            for name, weight in weights.items():
                raw = components.get(name, 0)
                score += raw * weight

            # Signal = score >= 0.65
            predicted_signal = score >= 0.65

            if predicted_signal and was_top:
                true_positives += 1
            elif predicted_signal and not was_top:
                false_positives += 1
            elif not predicted_signal and was_top:
                false_negatives += 1
            else:
                true_negatives += 1

        total_top = true_positives + false_negatives
        total_signals = true_positives + false_positives

        detection_rate = true_positives / max(1, total_top)
        precision = true_positives / max(1, total_signals)
        fpr = false_positives / max(1, false_positives + true_negatives)

        return {
            "detection_rate": round(detection_rate, 4),
            "precision": round(precision, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def _load_signal_history(self) -> List[Dict]:
        """Charge l'historique des signaux depuis SQLite."""
        history = []
        try:
            from src.signal_logger import get_signal_history

            signals = get_signal_history(days_back=EVAL_LOOKBACK_DAYS)
            for s in signals:
                history.append({
                    "ticker": s.get("ticker"),
                    "monster_score": s.get("monster_score", 0),
                    "components": s.get("components", {}),
                    "was_top_gainer": s.get("actual_move_pct", 0) >= 30,
                    "signal_type": s.get("signal"),
                    "actual_move_pct": s.get("actual_move_pct", 0),
                })
        except Exception as e:
            logger.warning(f"Could not load signal history: {e}")

        return history

    def get_status(self) -> Dict:
        """Status de l'optimiseur."""
        return {
            "current_weights": self._current_weights,
            "history_count": len(self._history),
            "last_fitness": self._history[-1].fitness if self._history else None,
        }


# ============================================================================
# Singleton
# ============================================================================

_optimizer: Optional[WeightOptimizer] = None
_optimizer_lock = threading.Lock()


def get_weight_optimizer() -> WeightOptimizer:
    """Get singleton WeightOptimizer instance."""
    global _optimizer
    with _optimizer_lock:
        if _optimizer is None:
            _optimizer = WeightOptimizer()
    return _optimizer
