#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
pre-commit run --all-files "$@"
