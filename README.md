# Zindi: Google Mumbai Data Challenge — Air Quality Mapping & Forecasting (`$6,000 USD` Cash)

- **Competition URL:** `https://zindi.africa/competitions/google-mumbai-data-challenge`
- **Prize Pool:** `$6,000 USD` Cash (1st: `$3,000`, 2nd: `$2,000`, 3rd: `$1,000`)
- **Launch Window:** Opens **October 26, 2026 at 12:00 UTC** (`secret_code_required = False`, currently only ~20 registered participants!)

## Ready-to-Run Day-1 Solver (`solve_mumbai_aq.py`)
As soon as `Train.csv`, `Test.csv`, and `SampleSubmission.csv` unlock tomorrow at 12:00 UTC, place them in this directory and run:
```bash
python solve_mumbai_aq.py
```
It automatically extracts Mumbai coastal/BKC spatial kriging coordinates, Arabian Sea land/sea-breeze diurnal harmonics, and fits a 5-fold Spatial-GroupKFold ensemble (`HistGradientBoosting + Ridge`) to output `submission_mumbai_aq_v1.csv`.
