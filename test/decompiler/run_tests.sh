#!/bin/sh
# The shader decompiler's two Vulkan outputs, GLSL through rslc and SPIR-V
# written directly, drawn against each other (decompiler_test.cpp). RUNNER
# prefixes the test (wine for a MinGW build); DECOMPILER_COUNT sets how
# many random shaders. The direct modules go through spirv-val where that
# is installed. VN_VALIDATE runs it under the Khronos validation layer,
# where any error fails.
set -e
cd "$(dirname "$0")"
make -s
# the driver's threads' allocations (lsan.supp)
export LSAN_OPTIONS="suppressions=$(pwd)/lsan.supp${LSAN_OPTIONS:+:$LSAN_OPTIONS}"
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 900 $RUN"
COUNT=${DECOMPILER_COUNT:-600}
OUT=modules
rm -rf "$OUT"
mkdir -p "$OUT"
if [ -n "$VN_VALIDATE" ]; then
	LAYER=VK_LAYER_KHRONOS_validation
	found=
	for d in /usr/share/vulkan/explicit_layer.d /usr/local/share/vulkan/explicit_layer.d /etc/vulkan/explicit_layer.d; do
		[ -f "$d/VkLayer_khronos_validation.json" ] && found=1
	done
	if [ -z "$found" ]; then
		echo "FAIL: VN_VALIDATE is set but the validation layer is not installed"
		exit 1
	fi
	out=$(VK_INSTANCE_LAYERS=$LAYER $RUN ./decompiler_test${EXE:-} "$OUT" "$COUNT" 2>&1) || { echo "$out"; exit 1; }
	echo "$out" | grep -v "^VUID\|Validation" || true
	errors=$(echo "$out" | grep -c "Validation Error" || true)
	echo "validation errors: $errors"
	[ "$errors" = 0 ]
else
	$RUN ./decompiler_test${EXE:-} "$OUT" "$COUNT"
fi
if command -v spirv-val >/dev/null 2>&1; then
	n=0
	for f in "$OUT"/direct_*.spv; do
		[ -f "$f" ] || continue
		spirv-val --target-env vulkan1.0 "$f" || { echo "FAIL: $f is not valid SPIR-V"; exit 1; }
		n=$((n + 1))
	done
	echo "spirv-val: $n direct modules valid"
elif [ -n "$VN_REQUIRE_DEVICE" ]; then
	echo "FAIL: VN_REQUIRE_DEVICE is set but spirv-val is not installed"
	exit 1
fi
rm -rf "$OUT"
