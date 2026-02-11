# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Before opening a PR

Run quality gates locally:

```bash
make lint
make test
```

If you changed formatting-sensitive code:

```bash
make format
```

## Pull request checklist

- Keep changes focused and atomic.
- Add or update tests for behavior changes.
- Update documentation (`README.md`, CLI help examples, or comments) when interfaces change.
- Avoid committing secrets (`.env`, Kaggle keys, provider API keys).

## Commit style

Use concise, imperative commit messages. Example:

- `improve geometry recheck stage`
- `add ci workflow for lint and tests`
