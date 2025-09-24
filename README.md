# Mexico’s Energy–Economy: VAR, IRF & FEVD (Replication)

Replication materials for the paper “Mexico’s energy–economy: VAR-IRF and FEVD”.

## Repo structure
- `code/`: Python scripts (`01_preprocess.py`, `02_var_estimation.py`, `03_irf_fevd.py`)
- `data/raw/`: links/instructions to download public data (WDI, INEGI, Banxico, SENER, PEMEX, IEA)
- `data/processed/`: harmonized datasets used in the paper (if small)
- `figures/`: output charts (GIRF, IRF, FEVD)

## Environment
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

