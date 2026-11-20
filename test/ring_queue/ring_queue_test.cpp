// The fixed first-in first-out queue (RingQueue.h) the Vulkan and Metal
// upload allocators keep their sync points in: against a reference deque
// for random pushes and pops across many wraps, a full queue refusing a
// push (and its back still the last item queued, which the allocators then
// extend), batches made in place keeping their room (as the heap
// allocator's release queue reuses them), and none of it allocating.
#include "util/helpers/RingQueue.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <new>
#include <random>
#include <vector>

#if defined(__SANITIZE_ADDRESS__)
#define COUNT_ALLOCS 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define COUNT_ALLOCS 0
#endif
#endif
#ifndef COUNT_ALLOCS
#define COUNT_ALLOCS 1
#endif
static unsigned long long s_allocs = 0;
#if COUNT_ALLOCS
void* operator new(size_t size)
{
	s_allocs++;
	if (void* p = std::malloc(size ? size : 1))
		return p;
	throw std::bad_alloc();
}
void* operator new[](size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
#endif

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct SyncPoint
{
	uint64_t commandBufferId;
	uint32_t offset;
};

int main()
{
	static RingQueue<SyncPoint, 16> q;
	std::deque<SyncPoint> ref;
	std::mt19937 rng(4);
	uint64_t id = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (rng() % 3 != 0)
		{
			const SyncPoint p{ ++id, (uint32_t)rng() };
			const bool pushed = q.push(p);
			CHECK(pushed == (ref.size() < 16));
			if (pushed)
				ref.push_back(p);
			else
				CHECK(q.full() && q.back().commandBufferId == ref.back().commandBufferId);
		}
		else if (!ref.empty())
		{
			CHECK(!q.empty() && q.front().commandBufferId == ref.front().commandBufferId && q.front().offset == ref.front().offset);
			q.pop();
			ref.pop_front();
		}
		CHECK(q.size() == ref.size() && q.empty() == ref.empty());
		if (!ref.empty())
			CHECK(q.back().commandBufferId == ref.back().commandBufferId);
	}
	// as the allocators use it: pushed, popped as command buffers finish
	const unsigned long long before = s_allocs;
	RingQueue<SyncPoint, 256> sync;
	for (uint64_t cb = 1; cb < 100000; cb++)
	{
		if (!sync.push({ cb, (uint32_t)cb }))
			sync.back().commandBufferId = cb;
		while (!sync.empty() && sync.front().commandBufferId + 8 < cb)
			sync.pop();
	}
	std::printf("ring_queue: %llu allocations for 100000 sync points\n", s_allocs - before);
	CHECK(s_allocs == before);
	// batches in place: once each place has held its largest, none allocates
	struct Batch
	{
		uint64_t id;
		std::vector<uint32_t> items;
	};
	static RingQueue<Batch, 8> batches;
	auto cycle = [&](uint64_t base) {
		uint32_t released = 0;
		for (uint64_t cb = base; cb < base + 64; cb++)
		{
			Batch& b = batches.push_slot();
			b.id = cb;
			b.items.clear();
			for (uint32_t i = 0; i < 20; i++)
				b.items.push_back(i);
			while (batches.size() > 4)
			{
				released += (uint32_t)batches.front().items.size();
				batches.front().items.clear();
				batches.pop();
			}
		}
		return released;
	};
	cycle(1);
	const unsigned long long beforeBatches = s_allocs;
	const uint32_t released = cycle(1000);
	std::printf("ring_queue: %llu allocations for 64 batches made in place\n", s_allocs - beforeBatches);
	CHECK(s_allocs == beforeBatches && released == 64 * 20);
	if (s_failures)
	{
		std::printf("ring_queue_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("ring_queue_test: all passed\n");
	return 0;
}
