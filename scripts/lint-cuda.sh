#!/bin/bash
PS4="\n\033[1;33m>>>\033[0m "; set -x
set -e
set -o pipefail # abort if a command fails

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR="${SCRIPT_DIR}/.."

CLANG_TIDY_FIX_PARAM=""
CLANG_FORMAT_FIX_PARAM="--dry-run"
if [ "$1" = "--fix" ];
then
    CLANG_TIDY_FIX_PARAM="--fix-errors"
    CLANG_FORMAT_FIX_PARAM="-i"
fi;

find "$PROJECT_DIR" -regex '.*\.\(cu\)' | grep -Ev "venv|cupc|build" | xargs -t -I{} clang-tidy $CLANG_TIDY_FIX_PARAM {} -- --cuda-host-only

find "$PROJECT_DIR" -regex '.*\.\(cu\)' | grep -Ev "venv|cupc|build" | xargs -t -I{} clang-format $CLANG_FORMAT_FIX_PARAM --Werror {}
