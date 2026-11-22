#!/bin/sh
# Fails if work only a later draw may need - parts and pipelines made ahead,
# optimized links - is queued for the compile threads in the lane of what a
# draw needs now (VulkanPipelineCompiler.cpp, the three lanes): there it
# went before the compile of a pipeline whose draws were being skipped, each
# shader's ahead parts up to 64 builds long, and the draws stayed skipped
# for all of it.
cd "$(dirname "$0")/.." || exit 2
wrong=$(awk '
	FNR == 1 { inside = 0 }
	# (PipelineAhead in a namespace, a tab in)
	/^(void|bool) PipelineCompiler::PrepareAheadParts(For)?\(/ { inside = 1; name = $0; queued = 0; end = "}" }
	/^	void PipelineAhead\(/ { inside = 1; name = $0; queued = 0; end = "	}" }
	inside && /CompileThreadPool_QueuePrepared(Urgent)?\(/ { print FILENAME ":" FNR ": queued with what a draw needs now: " $0 }
	inside && /CompileThreadPool_QueuePreparedAhead\(/ { queued = 1 }
	inside && $0 == end { if (!queued) print FILENAME ": " name ": not queued ahead"; inside = 0 }
	/^void PipelineCompiler::CompileThreadPool_QueueCompilation\(/ { links = 1 }
	links && /s_aheadCompileRequests\.push\(v\)/ { linksAhead = 1 }
	links && /^}/ { links = 0 }
	END { if (!linksAhead) print "CompileThreadPool_QueueCompilation (optimized links): not in the lane made ahead" }' \
	src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp \
	src/Cafe/HW/Latte/Core/LatteShaderAhead.cpp)
if [ -n "$wrong" ]; then
	echo "work made ahead queued before what a draw needs now:"
	printf '%s\n' "$wrong"
	exit 1
fi
echo "work made ahead is queued after what a draw needs now"
