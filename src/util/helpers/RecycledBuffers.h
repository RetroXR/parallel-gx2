#pragma once

#include <cstdint>
#include <vector>

#include "util/helpers/MpmcQueue.h"

// Byte buffers handed from a loader to the threads that use them and back,
// with no lock: a buffer taken here is one a user gave back, its capacity
// kept, so once as many as are ever out at once have grown to the largest
// entry a cache load reads its entries without allocating. The pipeline
// caches' loaders (VulkanPipelineStableCache, MetalPipelineCache) read each
// entry into one and queue it for a compile thread, which gives it back.
class RecycledBuffers
{
public:
	// a given-back buffer in out (empty, its capacity kept) when out has none
	// of its own and one was given back; out as it was if not
	void Take(std::vector<uint8_t>& out)
	{
		if (out.capacity() != 0)
			return;
		std::vector<uint8_t> got;
		if (m_free.peek2(got))
			out.swap(got);
	}

	// buf back, for a later Take; buf left empty
	void Give(std::vector<uint8_t>& buf)
	{
		if (buf.capacity() == 0)
			return;
		buf.clear();
		m_free.push(std::move(buf));
		buf = std::vector<uint8_t>();
	}

	size_t Count() const { return m_free.size(); }

private:
	MpmcQueue<std::vector<uint8_t>> m_free;
};
