# Contributing to My-LM

Thanks for your interest in contributing! My-LM is an open-source local-AI playground and we welcome bug reports, features, docs improvements, and code.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

## Quick links

- 🐛 **Bug?** [Open a bug report](https://github.com/lavely/my-lm/issues/new?template=bug_report.yml)
- 💡 **Idea?** [Open a feature request](https://github.com/lavely/my-lm/issues/new?template=feature_request.yml)
- 🔒 **Security issue?** See [SECURITY.md](SECURITY.md) — please don't open a public issue.
- 💬 **Question?** Use [GitHub Discussions](https://github.com/lavely/my-lm/discussions).

## Development setup

```bash
git clone https://github.com/lavely/my-lm.git
cd my-lm

# Linux / macOS
./setup.sh

# Windows
setup.bat
```

This creates `.venv`, installs Python deps, and builds the Electron UI. Then:

```bash
cd ui
npm run dev   # main + renderer watch + electron
```

For Python work without rebuilding the UI:

```bash
source .venv/bin/activate           # or .venv\Scripts\activate on Windows
python scripts/qwen_inference.py    # smoke-test chat
```

## Coding conventions

### Python

- Target Python 3.10+
- Format with `black`, lint with `ruff` (configured in `pyproject.toml`)
- Type hints on public functions
- Long-lived bridges (`*_bridge.py`) communicate via newline-delimited JSON over stdin/stdout — never print non-JSON to stdout

```bash
ruff check scripts
black --check scripts
```

### TypeScript

- Strict mode (already on in `tsconfig.*.json`)
- Two-space indent
- Format with Prettier, lint with ESLint (when configured in Phase 5)

```bash
cd ui
npm run build   # type-check both main and renderer
```

### Commits

- Use clear, imperative subject lines: "Add face-detailer padding option" not "added padding"
- Keep commits focused — one logical change per commit
- Reference issues: `Fixes #123` in the body

## Pull requests

1. Fork and create a feature branch off `main`
2. Make your change with a focused diff
3. Run lint/build locally before pushing
4. Open a PR using the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
5. Be patient — maintainer reviews are best-effort

We squash-merge by default. The PR title becomes the squash commit subject — please make it descriptive.

## What we're looking for

**Yes please:**
- Bug fixes (especially VRAM-related)
- New model support in `scripts/modelCatalog` and `ui/src/main/modelCatalog.ts`
- Docs improvements
- Cross-platform fixes (macOS / Linux feedback especially welcome)
- Test coverage

**Probably not:**
- Cloud inference backends (this project is local-first by design)
- Pulling in heavy frameworks (React, Vue) for the renderer — vanilla TS is intentional
- Breaking changes to the bridge JSON protocol without an upgrade path

## License

By contributing, you agree your contributions are licensed under the [MIT License](LICENSE).
