#!/bin/bash
PS4="\n\033[1;33m>>>\033[0m "; set -x
set -e
set -o pipefail # abort if a command fails

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR="${SCRIPT_DIR}/.."

if [ "$1" = "--fix" ];
then
    black "$PROJECT_DIR"
else
    black "$PROJECT_DIR" --check --diff
fi;
