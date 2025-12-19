# Contributing

Thanks for your interest in improving bible-eval.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

- Tests: `pytest`
- Lint: `ruff check src tests`

## Data and licensing

Please only use public domain Bible texts (KJV 1769, WEB, ASV). Do not commit
copyrighted translations (NIV/ESV/etc.) to a public repo.

## Pull requests

- Keep changes focused and include tests for new behavior.
- Update `docs/CONNECTORS.md` and `docs/CONFIG.md` if you add a connector or new
  config options.
- If you change scoring behavior, document the rationale in the PR description.
