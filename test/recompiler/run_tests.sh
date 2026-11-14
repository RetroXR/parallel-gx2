#!/bin/sh
# The PPC recompiler against the interpreter. RUNNER prefixes the test (wine
# for a MinGW build); JOBS compiles that many files at once.
set -e
cd "$(dirname "$0")"
make -s -j"${JOBS:-2}"
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 300 $RUN"
$RUN ./recompiler_test${EXE:-}
