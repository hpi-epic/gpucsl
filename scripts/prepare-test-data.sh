#!/bin/bash


SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR="${SCRIPT_DIR}/.."

cd "$SCRIPT_DIR"


echo "Start preparing link, munin, alarm test data"
python3 encode-discrete-data.py alarm
python3 encode-discrete-data.py link
python3 encode-discrete-data.py munin

echo "Execute alarm"
Rscript use_bnlearn_discrete.R alarm 1
Rscript use_bnlearn_discrete.R alarm 3
Rscript use_bnlearn_discrete.R alarm 8
Rscript use_bnlearn_discrete.R alarm 11

echo "Execute link"
Rscript use_bnlearn_discrete.R link 1
Rscript use_bnlearn_discrete.R link 3

echo "Execute munin"
Rscript use_bnlearn_discrete.R munin 1
Rscript use_bnlearn_discrete.R munin 3

echo "Prepared test data for link, munin, alarm"
