#!/bin/sh
# Fails if a way a new pipeline's first draw goes, or a draw skipped for a
# pipeline not made yet, is not counted (LatteDrawPathCounts.h): the
# "[ShaderAhead] new pipelines at their first draw" line a tester's log has
# at title stop is how M2 (docs/SHADER-STUTTER.md) is judged on real games,
# and an outcome left out of it reads as one that never happens.
cd "$(dirname "$0")/.." || exit 2
f=src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRendererCore.cpp
missing=$(awk '
	/^PipelineInfo\* VulkanRenderer::draw_createGraphicsPipeline\(/ { create = 1 }
	create && /NotePipeline\(LatteDrawPathCounts::Ready,/ { ready = 1 }
	create && /NotePipeline\(LatteDrawPathCounts::Skipped,/ { skipped = 1 }
	create && /NotePipeline\(LatteDrawPathCounts::Waited,/ { waited = 1 }
	create && /^}/ { create = 0 }
	/^bool VulkanRenderer::draw_awaitAsyncPipeline\(/ { await = 1; found = 1 }
	await && /NoteAwait\(/ { awaited = 1 }
	# every way out that skips the draw counts it on the line before
	await && /return false;/ && prev !~ /NoteSkippedDraw\(\)/ { print FILENAME ":" FNR ": a skipped draw not counted" }
	await && /^}/ { await = 0 }
	await && NF { prev = $0 }
	END {
		if (!ready) print "draw_createGraphicsPipeline: no Ready pipeline counted"
		if (!skipped) print "draw_createGraphicsPipeline: no Skipped pipeline counted"
		if (!waited) print "draw_createGraphicsPipeline: no Waited pipeline counted"
		if (!found) print "draw_awaitAsyncPipeline: not found"
		if (!awaited) print "draw_awaitAsyncPipeline: no wait for a compile thread counted"
	}' "$f")
if [ -n "$missing" ]; then
	echo "a new pipeline's draw or a skipped draw not counted:"
	printf '%s\n' "$missing"
	exit 1
fi
echo "every new pipeline's first draw and every skipped draw is counted"
