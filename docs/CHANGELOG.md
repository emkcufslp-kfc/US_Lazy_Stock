# Changelog

## 2026-04-20
### Added
- `docs/SPEC.md` with current product scope and requirements.
- `docs/ACCEPTANCE.md` with executable acceptance criteria.
- `docs/RISKS.md` with operational/data/product risk register.
- `docs/TODO.md` running implementation log.

### Changed
- `app/us_stock_monitor_app.py`
  - Added `SAMPLE_SCREEN_PATH` and cached `load_sample_screen_data()`.
  - Added offline sample mode toggle in sidebar (default on).
  - Updated screen execution path to load bundled sample data in offline mode.
  - Set live fundamentals auto-fill default to off.
- `.gitignore`
  - Added local dependency/smoke artifacts (`deps/`, `.deps/`, `.vendor/`, `streamlit_smoke*.txt`).

### Notes
- Offline-first mode improves reproducibility in restricted/no-network environments.
