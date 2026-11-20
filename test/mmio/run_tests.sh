#!/bin/sh
# The MMIO handlers registered as the hardware is set up and found by address. RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 120 $RUN"
$RUN ./mmio_test${EXE:-}
