uv tool install pre-commit

cd "$(dirname "$0")/.."
pre-commit install
pre-commit install-hooks
