# European Values

Measuring European values of LLMs. This repo contains the full pipeline: data loading, processing, discriminative/generative training, survey optimisation, and a Gradio web app.

## Stack

- **Python 3.11**
- **uv** for package management (not pip)
- **Gradio** for the web app
- **Hydra** for config management
- **PyTorch, scikit-learn** for ML
- **Ruff + Mypy** for quality checks

## Layout

| Path | Purpose | Agent notes |
|------|---------|-------------|
| `src/european_values/` | Core library (imported) | Use relative imports within |
| `src/scripts/` | Entry points (run with `uv run`) | Use absolute imports |
| `tests/` | pytest suite | Run via `uv run pytest` |
| `config/` | Hydra configs | Edited by agents |
| `data/` | Datasets (raw/processed/final) | Large files, gitignored |
| `docs/` | MkDocs source | Published to GitHub Pages |

## Running it

```bash
# Install everything
make install

# Run quality checks (format, lint, type-check)
make check

# Run tests
uv run pytest

# Run the Gradio app
uv run src/scripts/run_app.py
```

## Testing

- Framework: pytest
- Location: `tests/`
- Run: `uv run pytest` (or `make test`)
- CI runs tests on every PR

## Conventions

- **British English** in code comments, docstrings, and docs (organise, colour)
- **Line width:** 88 characters for code and Markdown
- **Type hints:** Python 3.11+ style (`list[T]`, `X | None`)
- **Logging:** Use `logging` module, never `print()`
- **Paths:** Use `pathlib.Path`, not strings
- **Dependencies:** `uv add <pkg>` (not manual pyproject.toml edits)
- **Commits:** See [CONTRIBUTING.md](CONTRIBUTING.md)

## Gotchas

- **No `print()`:** Always use a logger
- **Relative vs absolute imports:** Relative in `src/european_values/`, absolute in `src/scripts/`
- **Data files:** Large datasets in `data/` are gitignored — don't commit them
- **Config:** Hydra configs in `config/` are editable; check schema before adding keys
- **CI:** Pre-commit hooks run on PR — run `make check` before pushing
- **Markdown:** Wrap at 88 chars; use `prettier --write` to fix
