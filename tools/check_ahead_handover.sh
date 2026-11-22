#!/bin/sh
# Fails if a pipeline a read made ahead of its draw no longer reaches it
# whole: the read's slot hands it over (PipelineCompiler::HandOverAhead),
# the draw takes it before it would link or compile one itself (TakeAhead,
# first in draw_createGraphicsPipeline), and those no draw took are given
# up as the title stops and as the renderer goes (DropAhead). Without the
# hand-over the draw linked or compiled again what the read had made, on
# the GPU thread; without the drop the pipelines would outlive the device.
cd "$(dirname "$0")/.." || exit 2
core=src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRendererCore.cpp
wrong=$(awk '
	/^PipelineInfo\* VulkanRenderer::draw_createGraphicsPipeline\(/ { inside = 1 }
	inside && /TakeAhead\(/ && !compiled { taken = 1 }
	inside && /->Compile\(/ { compiled = 1 }
	inside && /^}/ { inside = 0 }
	END { if (!taken) print "draw_createGraphicsPipeline: no pipeline made ahead taken before its compile" }' "$core")
grep -q 'PipelineCompiler::HandOverAhead(' src/Cafe/HW/Latte/Core/LatteShaderAhead.cpp ||
	wrong="$wrong
LatteShaderAhead.cpp: the pipelines made ahead not handed over"
for f in src/Cafe/HW/Latte/Core/LatteShaderAhead.cpp src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.cpp; do
	grep -q 'PipelineCompiler::DropAhead()' "$f" || wrong="$wrong
$f: the pipelines made ahead no draw took not given up"
done
if [ -n "$wrong" ]; then
	echo "a pipeline made ahead does not reach its draw whole:"
	printf '%s\n' "$wrong"
	exit 1
fi
echo "a pipeline made ahead reaches its draw whole"
