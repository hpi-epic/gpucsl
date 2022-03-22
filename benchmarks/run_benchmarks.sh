#!/bin/bash
#PS4="\n\033[1;33m>>>\033[0m "; set -x

if [ "$#" != 0 ] && [ "$#" != 1 ] && [ "$#" != 3 ]; then
  echo "Usage: $0 [LIBRARIES (optional)] [GAUSSIAN_BENCHMARKS DISCRETE_BENCHMARKS (two strings) (optional)]" >&2
  echo "Example: ./benchmarks/run_benchmarks.sh \"gpucsl,bnlearn,pcalg,cupc\" \"BR51,NCI-60,MCC,DREAM5-Insilico,Saureus,Scerevisiae\" \"alarm,link,munin\""
  exit 1
fi

GAUSSIAN_BENCHMARKS="BR51,NCI-60,MCC,DREAM5-Insilico,Saureus,Scerevisiae"
DISCRETE_BENCHMARKS="alarm,link,munin"
LIBRARIES="gpucsl,bnlearn,pcalg,cupc"

if [ "$#" -gt 0 ]; then
  LIBRARIES=$1
fi

if [ "$#" == 3 ]; then
  GAUSSIAN_BENCHMARKS=$2
  DISCRETE_BENCHMARKS=$3
fi



HOSTNAME=$(hostname)
COMMIT=$(git rev-parse --short HEAD)
DATE=$(date +%y-%m-%d)
TIME=$(date +%H:%M:%S)
MAX_LEVEL="3"
OUTPUT_DIR="output/output_${DATE}_${TIME}_${COMMIT}_${HOSTNAME}_max-level-${MAX_LEVEL}"
SCRIPT_DIR=$(dirname "$0")

echo "running BENCHMARKS: gaussian = ${GAUSSIAN_BENCHMARKS}, discrete = ${DISCRETE_BENCHMARKS}"
echo "output will be stored in: ${OUTPUT_DIR}"
mkdir -p "${SCRIPT_DIR}/${OUTPUT_DIR}"

gpucsl_return=0
if [[ "$LIBRARIES" == *"gpucsl"* ]]; then
  echo "running GPUCSL..."
  python3 "${SCRIPT_DIR}/benchmark_gpucsl.py" -dg \'${GAUSSIAN_BENCHMARKS}\' -dd \'$DISCRETE_BENCHMARKS\' -o $OUTPUT_DIR -l $MAX_LEVEL
  # possible additional arguments: --devices "1"
  gpucsl_return=$?
fi

cupc_return=0
if [[ "$LIBRARIES" == *"cupc"* ]] || [[ "$LIBRARIES" == *"pcalg"* ]] || [[ "$LIBRARIES" == *"bnlearn"* ]]; then
  echo "running cupc/pcalg/bnlearn..."
  "${SCRIPT_DIR}/benchmark_cupc_pcalg_bnlearn.sh" \'$GAUSSIAN_BENCHMARKS\' \'$DISCRETE_BENCHMARKS\' $OUTPUT_DIR $MAX_LEVEL $LIBRARIES
  cupc_return=$?
fi

echo "output written to: ./benchmarks/${OUTPUT_DIR}"
echo "total runtime of this script: ${SECONDS} seconds"


if [ $gpucsl_return != 0 ]; then
  echo "GPUCSL FAILED"
else
  echo "GPUCSL SUCCESSFUL"
fi

if [ $cupc_return != 0 ]; then
  echo "cupc/pcalg/bnlearn FAILED"
else
  echo "cupc/pcalg/bnlearn SUCCESSFUL"
fi

if [ $gpucsl_return == 0 ] && [ $cupc_return == 0 ]; then
  echo "Benchmark run SUCCESSFUL"
fi

exit 0

