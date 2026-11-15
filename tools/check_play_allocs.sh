#!/bin/sh
# Fails if a function on the list below - code that runs while a game plays,
# per frame, per draw, per new state or per guest call (AGENTS.md: no
# allocations mid-game) - declares a container or string of its own, formats
# a string, makes something with make_unique/make_shared/malloc, or news an
# object of a class that does not come from a pool (PoolNew.h). What it
# keeps static, or borrows by reference, it may. Each function is checked
# by its body, so what it calls is checked where it is listed itself. The
# list only grows: a function on it has been made allocation-free, and an
# allocation must not come back.
cd "$(dirname "$0")/.." || exit 2

list='
src/Cafe/HW/Latte/Core/LatteTexture.cpp LatteTexture_Invalidate
src/Cafe/HW/Latte/Core/LatteTexture.cpp LatteTC_ForEachTextureAtPhysAddr
src/Cafe/HW/Latte/Core/LatteRenderTarget.cpp LatteRenderTarget_itHLEClearColorDepthStencil
src/Cafe/HW/Latte/Core/LatteTextureReadback.cpp LatteTextureReadback_StartTransfer
src/Cafe/HW/Latte/Core/LatteTextureReadback.cpp LatteTextureReadback_Update
src/Cafe/HW/Latte/Core/LatteTextureReadback.cpp LatteTextureReadback_UpdateFinishedTransfers
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.cpp VulkanRenderer::texture_createReadback
src/Cafe/HW/Latte/Core/LatteQuery.cpp LatteQuery_BeginOcclusionQuery
src/Cafe/HW/Latte/Core/LatteQuery.cpp LatteQuery_EndOcclusionQuery
src/Cafe/HW/Latte/Core/LatteQuery.cpp LatteQuery_UpdateFinishedQueries
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanQuery.cpp VulkanRenderer::occlusionQuery_create
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanQuery.cpp VulkanRenderer::occlusionQuery_destroy
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.cpp VKRObjectRenderPass::VKRObjectRenderPass
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanRendererCore.cpp VulkanRenderer::draw_createGraphicsPipeline
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp PipelineCompiler::CreateDescriptorSetLayout
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp PipelineCompiler::InitVertexInputState
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp PipelineCompiler::InitDynamicState
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp PipelineCompiler::PrepareAheadParts
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp PipelineCompiler::BuildAheadParts
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.cpp PipelineCompiler::CompileThreadPool_QueuePrepared
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineKey.cpp VulkanPipelineKey_DescriptorBindings
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineKey.cpp VulkanPipelineKey_DynamicStates
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineLibrary.cpp VulkanPipelineLibrary_Key
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineLibrary.cpp VulkanPipelineLibrary_Create
src/Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineLibrary.cpp VulkanPipelineLibrary_FromTemplate
src/Cafe/HW/Latte/Common/ShaderSerializer.cpp DeserializeShaderProgram
src/Cafe/Filesystem/fsc.cpp fsc_lookupPath
src/Cafe/Filesystem/fsc.cpp fsc_open
src/Cafe/Filesystem/fsc.cpp PopulateIterationList
src/Cafe/HW/Latte/Core/LatteBufferCache.cpp BufferCacheNode
src/Cafe/HW/Latte/Core/LatteBufferCache.cpp Create
src/Cafe/Filesystem/fscDeviceHostFS.cpp FSCVirtualFile_Host::OpenFile
src/Cafe/IOSU/fsa/iosu_fsa.cpp __FSATranslatePath
src/Cafe/IOSU/fsa/iosu_fsa.cpp __FSAOpenNode
src/Cafe/IOSU/fsa/iosu_fsa.cpp __FSAOpenFile
src/Cafe/IOSU/fsa/iosu_fsa.cpp __FSA_GetFileStat
'

# the body of every definition of $2 in $1 (overloads too), its lines numbered
bodies() {
	awk -v name="$2" '
		BEGIN { n = split(name, parts, "::"); short = parts[n] }
		!inside {
			line = $0
			sub(/\/\/.*/, "", line)
			if (line ~ /;[[:space:]]*$/ || line ~ /^[[:space:]]*(return|else|if|while|for)[[:space:](]/)
				next
			i = index(line, name "(")
			if (i == 0)
				next
			before = substr(line, i - 1, 1)
			if (i > 1 && before ~ /[A-Za-z0-9_:>]/ && substr(line, i - 2, 2) != "::")
				next
			inside = 1; depth = 0; started = 0
		}
		inside {
			print FILENAME ":" FNR ":" $0
			t = $0
			sub(/\/\/.*/, "", t)
			gsub(/'"'"'[{}]'"'"'/, "", t)
			o = gsub(/\{/, "{", t); c = gsub(/\}/, "}", t)
			depth += o - c
			if (o > 0)
				started = 1
			if (started && depth <= 0)
				inside = 0
		}
	' "$1"
}

pooled() {
	grep -rqE "(class|struct)[[:space:]]+$1[[:space:]]*(final[[:space:]]*)?:[^{;]*PoolNew<$1[,>]" src --include='*.h' --include='*.cpp'
}

out=$(echo "$list" | while read -r file fn; do
	[ -z "$file" ] && continue
	if [ ! -f "$file" ]; then
		echo "$file: not there (take $fn off the list in $0 only when it moved)"
		echo "#fail"
		continue
	fi
	body=$(bodies "$file" "$fn")
	if [ -z "$body" ]; then
		echo "$file: no definition of $fn found"
		echo "#fail"
		continue
	fi
	# code only: comment lines, logging and assertions out
	code=$(printf '%s\n' "$body" | grep -v -e ':[0-9]*:[[:space:]]*\(//\|/\*\|\*\)' | grep -v -e 'cemuLog_' -e 'UnrecoverableError' -e 'cemu_assert' -e 'PSO_TRACE')
	bad=$(printf '%s\n' "$code" | grep -E \
		-e 'std::(vector|string|wstring|u8string|deque|queue|list|map|multimap|set|unordered_map|unordered_set|function)(<[^;()&]*>)?[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*([;=({]|$)' \
		-e 'fs::path[[:space:]]+[A-Za-z_]' -e 'fmt::format\(' -e 'std::to_string\(' -e 'make_(unique|shared)<' \
		-e '(^|[^A-Za-z0-9_])(malloc|calloc|realloc|strdup)\(' | grep -v -e ':[[:space:]]*static[[:space:]]' -e 'thread_local')
	news=$(printf '%s\n' "$code" | grep -oE 'new[[:space:]]+[A-Za-z_][A-Za-z0-9_:]*[[:space:]]*[({]' | sed -E 's/new[[:space:]]+//; s/[[:space:]]*[({]$//' | sort -u)
	for t in $news; do
		if ! pooled "${t##*::}"; then
			bad="$bad
$(printf '%s\n' "$code" | grep -E "new[[:space:]]+$t[[:space:]]*[({]") (not from a pool)"
		fi
	done
	bad=$(printf '%s\n' "$bad" | grep -v '^$')
	if [ -n "$bad" ]; then
		echo "an allocation in $fn, which runs while a game plays:"
		printf '%s\n' "$bad"
		echo "#fail"
	fi
	echo "#checked"
done)
printf '%s\n' "$out" | grep -v '^#'
n=$(printf '%s\n' "$out" | grep -c '^#checked')
if printf '%s\n' "$out" | grep -q '^#fail'; then
	exit 1
fi
echo "no allocation in the $n functions listed that run while a game plays"
