#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "util/helpers/obj_pool.h"

// Blocks of a few sizes - 256 octets up to 64 KiB, each four times the last -
// from pools of their own (obj_pool.h), for any thread with no lock: what is
// put back is handed out again, so once as many as are ever out at once have
// been made, getting one allocates nothing. A block larger than the largest
// comes from the heap. The sizes up to 4 KiB have their first chunk made at
// load. Each block is 16-aligned.
class SizedBlocks
{
public:
	static constexpr unsigned kClasses = 5;
	static constexpr size_t kLargest = (size_t)256 << (2 * (kClasses - 1));
	static constexpr uint8_t kHeap = 0xFF;

	SizedBlocks()
	{
		for (unsigned c = 0; c < kClasses; c++)
		{
			obj_pool_init(&m_pools[c], m_rings[c], (unsigned)ClassSize(c), PerChunk(c));
			if (ClassSize(c) <= 4096)
				obj_pool_reserve(&m_pools[c], 1);
		}
	}
	~SizedBlocks()
	{
		for (unsigned c = 0; c < kClasses; c++)
			obj_pool_free_all(&m_pools[c]);
	}
	SizedBlocks(const SizedBlocks&) = delete;
	SizedBlocks& operator=(const SizedBlocks&) = delete;

	// size octets (not 0), its class in *cls for Put; nullptr for no memory
	void* Get(size_t size, uint8_t* cls)
	{
		for (unsigned c = 0; c < kClasses; c++)
		{
			if (size <= ClassSize(c))
			{
				*cls = (uint8_t)c;
				return obj_pool_get(&m_pools[c]);
			}
		}
		*cls = kHeap;
		return std::malloc(size);
	}

	void Put(void* block, uint8_t cls)
	{
		if (!block)
			return;
		if (cls == kHeap)
			std::free(block);
		else
			obj_pool_put(&m_pools[cls], block);
	}

	// blocks of a class out now (kClasses: those from the heap are not counted)
	int Out(unsigned cls) { return obj_pool_out(&m_pools[cls]); }

	static constexpr size_t ClassSize(unsigned c) { return (size_t)256 << (2 * c); }

private:
	// 64 a chunk up to 4 KiB, then 16 and 4
	static constexpr unsigned PerChunk(unsigned c) { return c <= 2 ? 64u : c == 3 ? 16u : 4u; }

	obj_pool_t m_pools[kClasses]{};
	alignas(8) unsigned char m_rings[kClasses][OBJ_POOL_RING_SIZE(64)]{};
};
