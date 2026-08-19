# Rocket Report — Architecture

## Overview
MLB betting analytics platform (app name: formerly baseball-predictions). Generates daily wagering recommendations, backtests models, and displays results via a Streamlit multi-page dashboard.

## Data Flow
```
MLB Stats API / Odds API / Weather API
        ↓
src/ingestion/          → data_files/raw/*.csv
        ↓
Parquet consolidation   → data_files/processed/*.parquet
        ↓
Feature Engineering     → pandas DataFrames
        ↓
XGBoost Models          → probabilities
        ↓
Pick Generation         → picks_history.parquet
        ↓
Streamlit Dashboard     → predictions.py (entry)
```

## ML Models
Three XGBoost models serialised with `joblib` in `models/`:
| Model | Target | Threshold |
|-------|--------|-----------|
| Underdog Moneyline | Upset probability (dogs ≥ +120) | edge > 6% |
| Run Line | Team covers ±1.5 | edge > 3% |
| Totals | Over/Under posted total | edge > 3% |

All models trained with scikit-learn pipelines. Confidence tiers: **HIGH** (edge >6%, half-Kelly), **MEDIUM** (3–6%, quarter-Kelly), **LOW** (1–3%, tracked only).

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| MLB Stats API | Game data, lineups, standings | None (public) |
| The Odds API | Live market odds | `ODDS_API_KEY` |
| Weather API | Park weather | `WEATHER_API_KEY` |
| FanGraphs (fg_guts/fg_park) | Park factors, advanced metrics | None (scraped) |
| Chadwick Bureau | Player IDs | None (public) |

## Key Components
- `src/ingestion/` — fetchers per data source, saves CSVs to `data_files/raw/`
- `src/models/` — feature engineering + XGBoost training
- `src/picks/daily_pipeline.py` — daily orchestration: fetch → predict → kelly size → save
- `src/bankroll/kelly.py` — Kelly Criterion bet sizing
- `src/data/queries.py` — all data access (reads Parquet, returns DataFrames)
- `src/data/cache.py` — `@st.cache_data` wrappers around query functions
- `streamlit_app/pages/` — 7 Streamlit pages (Today, Stats, Matchup, Models, Performance, Pick 6, About)

## Storage
- Primary: Parquet via pyarrow (`data_files/processed/`)
- Raw downloads: CSV (`data_files/raw/`)
- Model artifacts: `.joblib` (`models/`) — gitignored
- Schemas defined in `src/data/schemas.py`

## Deployment
Streamlit Cloud, Docker (`docker-compose.yml`), GitHub Actions CI/CD for nightly model retraining.
