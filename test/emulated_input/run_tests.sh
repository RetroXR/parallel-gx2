#!/bin/sh
# An emulated controller's mappings read together. RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 120 $RUN"
$RUN ./emulated_input_test${EXE:-}
