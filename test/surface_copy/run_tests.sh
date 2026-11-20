#!/bin/sh
# The surface copies' framebuffers and descriptor sets by pipeline, texture, slice and mip. RUNNER prefixes the test (wine for a MinGW build). A broken chain loops rather than fails, so the time limit is short.
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 60 $RUN"
$RUN ./surface_copy_test${EXE:-}
