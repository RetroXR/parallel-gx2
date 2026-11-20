#pragma once

#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCompiler.h"
#include "util/helpers/MpmcQueue.h"
#include "util/helpers/RecycledBuffers.h"
#include "util/helpers/MpscList.h"
#include "util/helpers/Serializer.h"
#include "util/math/vector2.h"

class MetalPipelineCache
{
public:
	static MetalPipelineCache& GetInstance();

    MetalPipelineCache(class MetalRenderer* metalRenderer);
    ~MetalPipelineCache();

    PipelineObject* GetRenderPipelineState(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, Vector2i extend, uint32 indexCount, const LatteContextRegister& lcr);

    // Cache loading
	uint32 BeginLoading(uint64 cacheTitleId); // returns count of pipelines stored in cache
	bool UpdateLoading(uint32& pipelinesLoadedTotal, uint32& pipelinesMissingShaders);
	void EndLoading();
	void LoadPipelineFromCache(std::span<uint8> fileData);
       void Close(); // called on title exit

    // Debug
    size_t GetPipelineCacheSize() const { return m_pipelineCache.size(); }

private:
    class MetalRenderer* m_mtlr;

    std::map<uint64, PipelineObject*> m_pipelineCache; // the GPU thread's alone
    // loaded by the compiler threads, for the GPU thread to put in the map
    MpscList<std::pair<uint64, PipelineObject*>> m_loadedPipelines;
    void TakeLoadedPipelines();

	std::thread* m_pipelineCacheStoreThread;
	// the writer thread's, emptied for each pipeline and grown only past the
	// largest it has written: made with the cache, not when the thread starts
	MemStreamWriter m_writeBuffer{ 1024 * 4 };

	class FileCache* s_cache;

	std::atomic_uint32_t m_numCompilationThreads{ 0 };
	MpmcQueue<std::vector<uint8>> m_compilationQueue; // the loader to every compiler thread: no lock
	// the entries' buffers, back from the compiler threads to the loader: a
	// load allocates for no more entries than are queued at once
	RecycledBuffers m_buffers;
	std::vector<uint8> m_loadBuffer;
	std::atomic_uint32_t m_compilationCount;

    static uint64 CalculatePipelineHash(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr);

    void AddCurrentStateToCache(uint64 pipelineStateHash, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo);
    void StartCacheStoreThread();

	// pipeline serialization for file
	bool SerializePipeline(class MemStreamWriter& memWriter, struct MetalCachedPipeline& cachedPipeline);
	bool DeserializePipeline(class MemStreamReader& memReader, struct MetalCachedPipeline& cachedPipeline);

    int CompilerThread();
	template<typename TPred>
	void WaitForCompileProgress(TPred progressed);
	void WorkerThread();
};
