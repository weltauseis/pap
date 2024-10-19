#!/usr/bin/env bash

if [[ -z ${EASYPAPDIR+x} ]]; then
    echo "Error: EASYPAPDIR variable is not defined (easypap-common.bash)" >&2
fi

TRACEDIR=${TRACEDIR:-${EASYPAPDIR}/data/traces}
TRACEFILE=${TRACEDIR}/ezv_trace_current.evt
OLDTRACEFILE=${TRACEDIR}/ezv_trace_previous.evt

HASHDIR=${EASYPAPDIR}/data/hash
DUMPDIR=${EASYPAPDIR}/data/dump

SIMU=${EASYPAPDIR}/bin/easypap
VIEW=${EASYPAPDIR}/traces/bin/easyview

KERNEL_PREFIX=("" "mipp_" "cuda_")
