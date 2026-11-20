#pragma once
#include "util/helpers/FixedQueue.h"
#include "util/helpers/FixedQueue.h"
#include "util/helpers/MpmcQueue.h"
#include "util/helpers/RecycledBuffers.h"
#include "util/helpers/U64Table.h"
#include "util/helpers/Serializer.h"

struct VulkanPipelineHash
{
	VulkanPipelineHash() = default;
	VulkanPipelineHash(uint64 h0, uint64 h1) : h0(h0), h1(h1) {};

	uint64 h0;
	uint64 h1;
};

class VulkanPipelineStableCache
{
	struct PipelineHash 
	{
		PipelineHash(uint64 h0, uint64 h1) : h0(h0), h1(h1) {};

		uint64 h0;
		uint64 h1;

		bool operator==(const PipelineHash& r) const
		{
			return h0 == r.h0 && h1 == r.h1;
		}
	};

public:
	static VulkanPipelineStableCache& GetInstance();

	uint32 BeginLoading(uint64 cacheTitleId); // returns count of pipelines stored in cache
	bool UpdateLoading(uint32& pipelinesLoadedTotal, uint32& pipelinesMissingShaders);
	void EndLoading();
	void LoadPipelineFromCache(std::span<uint8> fileData);
    void Close(); // called on title exit
	// libretro: closes the cache file Close() keeps open, once nothing is left
	// that could still reach it; returns false and keeps it when something is
	bool ReleaseCacheFile();

	bool HasPipelineCached(uint64 baseHash, uint64 pipelineStateHash);
	void AddCurrentStateToCache(uint64 baseHash, uint64 pipelineStateHash);

	// pipeline serialization for file
	bool SerializePipeline(class MemStreamWriter& memWriter, struct CachedPipeline& cachedPipeline);
	bool DeserializePipeline(class MemStreamReader& memReader, struct CachedPipeline& cachedPipeline);

private:
	int CompilerThread(); // runs on: this cache's own plCacheCompiler threads
	template<typename TPred>
	void WaitForCompileProgress(TPred progressed);
	void WorkerThread();  // runs on: this cache's own plCacheWriter thread

	std::thread* m_pipelineCacheStoreThread;
	// the writer thread's, emptied for each pipeline and grown only past the
	// largest it has written: made with the cache, not when the thread starts
	MemStreamWriter m_writeBuffer{ 1024 * 4 };
	// Upstream detaches both kinds of thread and never waits for them, which
	// costs nothing in a process that is about to end. A core outlives the
	// title: the compiler threads build pipelines that are registered with the
	// renderer, and the writer holds the cache file, so both have to be gone
	// before the renderer is - and a handle is what lets Close() join them
	// instead of waiting on a flag. The loops themselves are upstream's.
	std::vector<std::thread> m_compilerThreads;
	std::atomic_bool m_stopCacheStoreThread{ false };

	// the pipelines in the cache file or queued for it, by CachedKey: the GPU
	// thread's alone, sized at BeginLoading
	U64Table<char> m_pipelineIsCached;
	static uint64 CachedKey(uint64 baseHash, uint64 pipelineStateHash);
	void MarkCached(uint64 baseHash, uint64 pipelineStateHash);
	void StartCacheStoreThread();
	FixedList<PipelineHash, 4096> m_pipelinesLoaded; // from the compile threads, see TakeLoadedHashes
	void TakeLoadedHashes();
	class FileCache* s_cache;

	std::atomic_uint32_t m_numCompilationThreads{ 0 };
	MpmcQueue<std::vector<uint8>> m_compilationQueue; // the loader to every compiler thread: no lock
	// the entries' buffers, back from the compiler threads to the loader: a
	// load allocates for no more entries than are queued at once
	RecycledBuffers m_buffers;
	std::vector<uint8> m_loadBuffer;
	std::atomic_uint32_t m_compilationCount;
};