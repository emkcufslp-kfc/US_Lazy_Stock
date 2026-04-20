# TODO / Running Log

## 2026-04-20
- Read existing repo docs and app implementation (`README.md`, `docs/README_us_stock_monitor_app.md`, `app/us_stock_monitor_app.py`).
- Created missing project control docs: `SPEC`, `ACCEPTANCE`, `RISKS`, `TODO`, `CHANGELOG`.
- Added offline-first screen path in Streamlit app:
  - Added sample data loader for `data/full_screen_latest.csv`.
  - Added sidebar toggle `Use offline sample data (no live API)` defaulting to `True`.
  - Updated run flow to load bundled sample results in offline mode.
  - Changed live fundamentals autofill default to `False`.
- Next: finalize README updates and run validation loop.
- Updated `README.md` with offline-first startup, structure, and validation instructions.
- Validation loop results:
  - PASS: `python -m py_compile app/us_stock_monitor_app.py`.
  - BLOCKED: runtime smoke tests requiring `streamlit`/`pandas` imports in this sandbox environment (imports resolve as namespace stubs with missing symbols).
- Remaining task: run full Streamlit smoke (`streamlit run app/us_stock_monitor_app.py`) once dependency-import environment is healthy.
- Updated `.gitignore` to exclude local dependency/smoke artifacts generated during validation.
- Added reference-only fallback ingestion from verified GitHub datasets when primary free sources fail.
- Added `scripts/validate_reference_sources.py` and generated `docs/reference_validation_report.json` (`overall_ok=true`).
- Fixed production bug from screenshot: `KeyError` in passed-stocks grid caused by hardcoded missing columns; now renders only available columns.
