# Risks

## Runtime/Environment Risks
- Some environments may not permit user-level site-packages inside sandboxed runs.
- Live-data dependencies can fail due to network restrictions, throttling, or source schema drift.

## Data Quality Risks
- Free data sources can be delayed, incomplete, or inconsistent across endpoints.
- Offline sample data can become stale and diverge from live behavior.

## Product Risks
- BUY/SELL/HOLD logic is currently placeholder-only.
- Large single-file app structure increases maintenance risk.

## Mitigations
- Default to offline sample mode for deterministic startup.
- Keep live mode optional and explicit.
- Maintain validation loop with compile checks and targeted logic checks.
- Track refactor candidates in `docs/TODO.md` before large structural changes.
