#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "util/helpers/MpmcQueue.h"

// Buffers handed from a maker to the threads that use them and back,
// with no lock: a buffer taken here is one a user gave back, its capacity
// kept, so once as many as are ever out at once have grown to the largest
// entry a cache load reads its entries without allocating. The pipeline
// caches' loaders (VulkanPipelineStableCache, MetalPipelineCache) read each
// entry into one and queue it for a compile thread, which gives it back; the
// decompiler writes a shader's SPIR-V into one, which the Vulkan shader gives
// back once its module is made; and a shader's GLSL or MSL, given back by the
// renderer's shader once compiled.
template<typename C>
class Recycled
{
public:
	// a given-back one in out (empty, its capacity kept) when out has no room
	// of its own and one was given back; out as it was if not
	void Take(C& out)
	{
		if (out.capacity() > C().capacity())
			return;
		C got;
		if (m_free.peek2(got))
			out.swap(got);
	}

	// buf back, for a later Take; buf left empty
	void Give(C& buf)
	{
		if (buf.capacity() <= C().capacity())
		{
			buf.clear();
			return;
		}
		buf.clear();
		m_free.push(std::move(buf));
		buf = C();
	}

	size_t Count() const { return m_free.size(); }

private:
	MpmcQueue<C> m_free;
};

template<typename T>
using RecycledVectors = Recycled<std::vector<T>>;
using RecycledBuffers = RecycledVectors<uint8_t>;
using RecycledStrings = Recycled<std::string>;
