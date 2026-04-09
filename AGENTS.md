# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/gen_modeling/`. Use `datasets/` for data loaders and preprocessing, `modules/` for reusable network blocks, and `utils/` for math, optimization, and running statistics helpers. Top-level training and demo entry points live in `examples/` and are grouped by method or dataset, for example `examples/FM_mnist.py` and `examples/EqM_lafan1.py`. Utility scripts live in `scripts/`, and transient outputs should stay under local data or asset directories, not inside `src/`.

## Build, Test, and Development Commands
Install dependencies with `uv sync`.
Run the package stub with `uv run python main.py`.
Run an experiment with `uv run python examples/FM_mnist.py` or another file in `examples/`.
Run dataset utilities with commands such as `uv run python scripts/compute_image_stats.py --dataset mnist`.
Type-check before submitting changes with `uv run pyright`.

## Coding Style & Naming Conventions
Target Python 3.11+ and keep imports, type hints, and dataclasses consistent with the existing codebase. Use 4-space indentation, `snake_case` for functions, variables, and module names, and `PascalCase` for classes like `LinearFlow` or `MNISTDataset`. Prefer explicit tensor shapes and small helper functions over deeply nested training logic. Keep example filenames method-oriented and dataset-oriented: `METHOD_dataset.py`.

## Testing Guidelines
There is no dedicated `tests/` suite yet, so contributors should validate changes with the smallest relevant runnable example and a static check. At minimum, run `uv run pyright` and one affected script or example. When adding tests, place them under `tests/`, name files `test_<module>.py`, and prefer fast unit coverage for math utilities, dataset transforms, and model wrappers.

## Commit & Pull Request Guidelines
Recent commits use short, plain subjects such as `shared config` and `fix lafan1 data processing`. Keep commit titles brief, imperative, and focused on one change. Pull requests should include a concise description, the commands used for verification, any dataset or hardware assumptions, and screenshots or sample outputs when changing visualizations or generated assets.

## Data & Configuration Tips
Large datasets and downloaded assets should remain in local `data/` directories and stay out of git. Avoid hard-coded machine-specific paths in committed code; pass them as arguments or keep them in local-only edits.
