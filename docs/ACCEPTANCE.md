# Acceptance Criteria

## Dashboard Behavior
- App starts without syntax/runtime import errors in local validation.
- Sidebar exposes an offline sample mode.
- Offline sample mode loads bundled data and avoids live scan calls.
- Live mode still supports existing screen workflow.

## Data and File Behavior
- Save action writes passed stocks to `app/saved_watchlists/US stock dd-MMM-yyyy.csv`.
- Saved files tab lists available watchlists and row counts.

## Quality Checks
- `python -m py_compile app/us_stock_monitor_app.py` passes.
- Sample data loader returns a non-empty DataFrame and normalizes `pass` to boolean when present.

## Documentation
- `docs/SPEC.md`, `docs/ACCEPTANCE.md`, `docs/RISKS.md`, `docs/TODO.md`, `docs/CHANGELOG.md` exist and are updated.
- README documents offline-first usage and validation commands.
