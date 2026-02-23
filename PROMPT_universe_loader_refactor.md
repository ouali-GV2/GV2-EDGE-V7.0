# Refactoring de `src/universe_loader.py` — GV2-EDGE V7

## Contexte

GV2-EDGE est un scanner de momentum small caps US dont l'objectif est d'**anticiper les futurs top gainers avant leur hausse**. L'`universe_loader.py` actuel est le goulot d'étranglement principal qui empêche cet objectif.

---

## Problème à résoudre

### Ce que fait le code actuel (à corriger)

```python
# build_universe_v2() — src/universe_loader.py ligne 266
for i, ticker in enumerate(tickers[:500]):  # Limite DEMO jamais retirée
    profile = fetch_stock_profile(ticker)   # 1 appel API Finnhub par ticker
    quote = fetch_quote(ticker)             # 1 appel API Finnhub par ticker
    time.sleep(0.15)                        # 0.15s de délai
```

**Résultat** : 500 tickers × 2 appels × 0.15s = **~4 minutes de rebuild**, 1 000 appels Finnhub consommés, et seulement 500 tickers couverts sur ~4 000+ éligibles.

### Les 3 bugs structurels

**Bug 1 — Couverture tronquée**
La limite `tickers[:500]` est un artefact de développement jamais retiré. L'univers réel des common stocks NASDAQ/NYSE/AMEX éligibles dépasse 4 000 tickers. Un futur top gainer hors des 500 premiers est invisible pour tout le système.

**Bug 2 — Filtre volume prématuré**
```python
# filter_universe() ligne 210
filtered = filtered[filtered["volume"] >= MIN_AVG_VOLUME]  # 500K
```
Ce filtre exclut les tickers encore calmes **aujourd'hui** mais qui vont spiquer demain. C'est précisément les tickers que GV2-EDGE doit détecter. Un ticker avec 150K de volume moyen peut faire ×10 en volume le jour d'un catalyst FDA — il ne doit pas être exclu a priori.

**Bug 3 — Méthode de construction coûteuse et inutile**
Faire `profile2` + `quote` pour chaque ticker revient à payer en appels API pour obtenir des données (market cap, price) qui sont disponibles directement dans la réponse `/stock/symbol` de Finnhub, ou qui peuvent être vérifiées au moment du scan dans le Feature Engine.

---

## Ce qu'il faut implémenter

### Principe architectural

> **Univers large et minimal à la construction. Filtres dynamiques au moment du scan.**

L'universe loader doit uniquement répondre à la question : "quels tickers sont des common stocks US cotés sur un exchange majeur (non OTC), dans la tranche de prix small cap ?"  
Tout le reste (volume, market cap, liquidité) est vérifié en temps réel par le Feature Engine et le Monster Score — pas en amont.

### Nouveau flux de construction

```
1. GET /stock/symbol?exchange=US  →  1 seul appel API
   └── Retourne ~8 000 symbols US avec champs : symbol, type, exchange, displaySymbol
   
2. Filtre statique immédiat (sur les champs déjà présents, 0 appel supplémentaire) :
   - type == "Common Stock"
   - exchange ne contient pas "OTC", "Pink", "Grey"
   - symbol sans caractères spéciaux (/, -, warrants)
   
3. Sauvegarde CSV → data/universe.csv
   └── Colonnes minimales : ticker, exchange, name
   
4. Résultat attendu : ~2 500–3 500 tickers, rebuild en < 5 secondes
```

### Refresh policy

- **Rebuild complet** : 1 fois par semaine (dimanche soir, via le weekend mode existant)
- **Cache mémoire** : TTL 24h (au lieu de 1h actuellement — l'univers ne change pas en intraday)
- **Fallback** : si le rebuild échoue, charger `data/universe.csv` existant sans bloquer

---

## Spécifications du nouveau code

### Fonction `fetch_finnhub_symbols()` — à modifier

```python
# AVANT : retourne seulement symbol, displaySymbol, description
# APRÈS : retourne aussi exchange, type pour permettre filtrage sans appels supplémentaires
```

Finnhub `/stock/symbol` retourne déjà : `symbol`, `displaySymbol`, `description`, `type`, `exchange`, `currency`. Utiliser **tous ces champs** dans le filtre.

### Fonction `filter_universe()` — à réécrire

Filtres à conserver :
- `type == "Common Stock"` — exclut ETFs, warrants, preferred
- Exchange non-OTC : exclure si `exchange` contient `OTC`, `Pink`, `Grey`, `Expert`
- Symbole propre : exclure si le ticker contient `.`, `/`, `+`, `$`, ou se termine par `W`, `R`, `U`, `WS` (warrants, rights, units)
- Pas de filtre sur le prix — le Feature Engine s'en charge
- Pas de filtre sur le volume — le Feature Engine s'en charge  
- Pas de filtre sur le market cap — le Feature Engine s'en charge

Filtres à supprimer :
- `MIN_AVG_VOLUME` (supprimé de l'import config dans ce fichier)
- `MAX_MARKET_CAP` (supprimé de l'import config dans ce fichier)
- Le détecteur SPAC/shell basé sur le nom (trop agressif, supprime de vrais candidats)
- `time.sleep(0.15)` dans la boucle

### Fonction `build_universe()` — à réécrire

```python
def build_universe():
    """
    Construction en 1 appel API.
    Cible : ~2500-3500 tickers, < 5 secondes.
    """
    symbols = fetch_finnhub_symbols()      # 1 appel
    universe = filter_universe(symbols)    # filtre en mémoire, 0 appel
    return universe
```

Supprimer complètement :
- `fetch_stock_profile()` — inutile à ce stade
- `fetch_quote()` dans le context du builder — inutile à ce stade
- La boucle `for ticker in tickers[:500]`

### Fonction `load_universe()` — à modifier

```python
# TTL cache : passer de 60*60 (1h) à 60*60*24 (24h)
# L'univers des symboles cotés ne change pas en intraday
cache = Cache(ttl=60 * 60 * 24)
```

### Colonnes du DataFrame retourné

Le DataFrame doit contenir au minimum : `ticker`, `exchange`, `name`

Tous les consommateurs actuels n'utilisent que `row["ticker"]` ou `universe["ticker"].tolist()` — aucune rupture de compatibilité.

---

## Ce qu'il ne faut PAS modifier

- La signature de `load_universe(force_refresh=False)` — utilisée dans main.py, daily_audit, weekly_audit, backtest, watch_list, afterhours_scanner
- La signature de `get_tickers(limit=None)` — utilisée dans quelques modules
- Le fallback sur `data/universe.csv` si le rebuild échoue
- Le système de cache (juste changer le TTL)

---

## Fichiers à modifier

```
src/universe_loader.py     ← réécriture principale
config.py                  ← retirer MIN_AVG_VOLUME et MAX_MARKET_CAP 
                              de l'import dans universe_loader uniquement
                              (les garder dans config.py pour les autres modules)
```

---

## Critères de validation

Après le refactoring, vérifier que :

1. `load_universe()` s'exécute en < 10 secondes (au lieu de ~4 minutes)
2. `len(load_universe())` retourne entre 2 000 et 4 000 tickers
3. Aucun ticker OTC dans le résultat (`exchange` ne contient pas "OTC")
4. Aucun warrant dans le résultat (pas de ticker se terminant par `W`, `WS`, `R`)
5. La colonne `ticker` est présente et sans NaN
6. `main.py` s'exécute sans erreur (compatibilité interface préservée)
7. Le `data/universe.csv` généré est lisible par les autres modules

---

## Note sur la suppression du filtre volume

Le Feature Engine (`src/feature_engine.py`) calcule déjà `volume_spike` et `momentum` pour chaque ticker scanné. Un ticker sans volume obtiendra un `volume_spike` de 0 et un `momentum` de 0, donc un Monster Score trop faible pour déclencher un signal. Le filtre est redondant en amont et **contre-productif** pour l'objectif d'anticipation.
