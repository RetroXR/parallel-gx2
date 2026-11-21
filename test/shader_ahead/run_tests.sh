#!/bin/sh
# Pixel shaders decompiled ahead of their draw: the command buffer read
# ahead, the registers compared, the hand-over (Makefile). RUNNER prefixes
# the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 300 $RUN"
$RUN ./shader_ahead_test${EXE:-}
