// Pipelines made ahead of the draw or copy that needs them
// (Cafe/HW/Latte/Renderer/Vulkan/PreparedPipeline.h): queued as tickets for
// compile threads, taken by the GPU thread. The surface copies' pipelines go
// this way, queued when a texture that can be drawn to is created. Builds are
// counted, not compiled:
//  - made ahead: once the compile threads are through, every take finds its
//    pipeline built and builds nothing;
//  - a build is told whether it runs on a compile thread or on the thread
//    taking it (the pipeline compiler links differently on each);
//  - takes racing the compile threads: each built exactly once, by one side,
//    and every take gets that one;
//  - retired before a compile thread got to it: never built, and the ticket
//    still in the queue finds nothing; retired while one builds it: waits for
//    it, and what it built is there to take;
//  - tickets dropped with the queue, unrun: nothing leaks (LeakSanitizer);
//  - tickets come from a pool (ClaimTicket.h, PoolNew.h): once as many have
//    been out as a game keeps, queuing another allocates nothing;
//  - an urgent lane (MpmcQueue_PushUrgent, _PopPreferring: the reads ahead of
//    the GPU thread): queued after a backlog, taken before all of it; one
//    pushed to an idle pool wakes a thread for it; the wake left behind
//    when another thread took it is passed over; and three lanes (the
//    compile threads': reads, what a draw needs now, what is made ahead):
//    each lane's items in order, every earlier lane's before a later one's,
//    whatever order they were pushed in, the wakes passed over, and a
//    thread waiting on the last lane woken by a push to the middle one;
//  - buffers given back and taken again (RecycledBuffers.h: the decompiler's
//    SPIR-V and source): a taker that knows the room it needs gets one with
//    that room wherever it waits among those given back, nothing allocated
//    - where it took the first, smaller one and grew it; with none large
//    enough, one of them, the others kept;
//  - built together (BuildTogether, a waiting draw's missing pipeline parts):
//    each built once, the queued ones on the compile threads while the
//    others are built here when threads are free - the builds overlapping -
//    and here, one after the other, when none is.
#include "Cafe/HW/Latte/Renderer/Vulkan/PreparedPipeline.h"
#include "util/helpers/MpmcQueue.h"
#include "util/helpers/RecycledBuffers.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <memory>
#include <thread>
#include <vector>

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

// every operator new counted, for the tickets' allocations
static std::atomic<uint64_t> s_allocs{0};
void* operator new(size_t n)
{
	s_allocs.fetch_add(1, std::memory_order_relaxed);
	if (void* p = std::malloc(n ? n : 1))
		return p;
	throw std::bad_alloc();
}
void* operator new[](size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

struct Entry
{
	std::atomic<int> builds{0};
	std::atomic<bool> hold{false}; // a build waits while set
	std::atomic<bool> started{false};
	std::atomic<int> onCompileThread{-1}; // what the build was told
	uint64_t id = 0;
	// last: destroyed first, before what its build reads
	PreparedPipeline prepared{[](void* self, bool onCompileThread) {
		Entry* e = (Entry*)self;
		e->started = true;
		e->onCompileThread = onCompileThread;
		while (e->hold.load())
			std::this_thread::yield();
		e->builds++;
		return (VkPipeline)(uintptr_t)(e->id + 1);
	}, this};
};

// the compile thread pool's shape (VulkanPipelineCompiler.cpp): nullptr stops a thread
struct Pool
{
	MpmcQueue<PreparedPipeline::Ticket*> queue;
	std::vector<std::thread> threads;
	void Start(int n)
	{
		for (int i = 0; i < n; i++)
			threads.emplace_back([this] {
				for (;;)
				{
					PreparedPipeline::Ticket* t = queue.pop();
					if (!t)
						return;
					PreparedPipeline::Run(t);
				}
			});
	}
	void Stop()
	{
		for (size_t i = 0; i < threads.size(); i++)
			queue.push(nullptr);
		for (auto& t : threads)
			t.join();
		threads.clear();
		PreparedPipeline::Ticket* t;
		while (queue.peek2(t))
			if (t)
				PreparedPipeline::Drop(t);
	}
};

static void MadeAhead()
{
	std::vector<std::unique_ptr<Entry>> entries;
	Pool pool;
	for (int i = 0; i < 64; i++)
	{
		entries.emplace_back(new Entry());
		entries.back()->id = i;
		pool.queue.push(entries.back()->prepared.Queue());
	}
	pool.Start(4);
	pool.Stop(); // every ticket run
	int builtHere = 0;
	for (auto& e : entries)
	{
		bool here = false;
		const VkPipeline p = e->prepared.Take(&here);
		builtHere += here;
		CHECK(p == (VkPipeline)(uintptr_t)(e->id + 1));
		CHECK(e->builds == 1);
		CHECK(e->onCompileThread == 1);
	}
	CHECK(builtHere == 0);
}

static void Racing()
{
	Pool pool;
	pool.Start(4);
	int builtHere = 0;
	for (int round = 0; round < 50; round++)
	{
		std::vector<std::unique_ptr<Entry>> entries;
		for (int i = 0; i < 40; i++)
		{
			entries.emplace_back(new Entry());
			entries.back()->id = round * 100 + i;
			pool.queue.push(entries.back()->prepared.Queue());
		}
		// from the back: some still queued, some being built, some built
		for (int i = 39; i >= 0; i--)
		{
			bool here = false;
			const VkPipeline p = entries[i]->prepared.Take(&here);
			builtHere += here;
			CHECK(p == (VkPipeline)(uintptr_t)(entries[i]->id + 1));
			CHECK(entries[i]->prepared.Take() == p);
			// the build knew which side it was on
			CHECK(entries[i]->onCompileThread == (here ? 0 : 1));
		}
		for (auto& e : entries)
			CHECK(e->builds == 1);
	}
	pool.Stop();
	std::printf("racing: %d of 2000 taken before a compile thread built them\n", builtHere);
}

static void Retired()
{
	// before a compile thread got to it
	{
		Pool pool;
		Entry* e = new Entry();
		pool.queue.push(e->prepared.Queue());
		delete e;
		pool.Start(2); // the ticket finds nothing
		pool.Stop();
	}
	// while one builds it
	{
		Pool pool;
		auto e = std::make_unique<Entry>();
		e->hold = true;
		pool.queue.push(e->prepared.Queue());
		pool.Start(1);
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (!e->started && std::chrono::steady_clock::now() < deadline)
			std::this_thread::yield();
		CHECK(e->started); // a compile thread builds it
		if (!e->started)
		{
			e->prepared.Retire();
			pool.Stop();
			return;
		}
		std::thread release([&] {
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			e->hold = false;
		});
		e->prepared.Retire(); // waits for the build
		CHECK(e->prepared.IsBuilt());
		CHECK(e->builds == 1);
		CHECK(e->prepared.Take() == (VkPipeline)(uintptr_t)1);
		release.join();
		pool.Stop();
	}
}

static void Dropped()
{
	Pool pool; // never started: every ticket dropped with the queue
	std::vector<std::unique_ptr<Entry>> entries;
	for (int i = 0; i < 8; i++)
	{
		entries.emplace_back(new Entry());
		pool.queue.push(entries.back()->prepared.Queue());
	}
	pool.Stop();
	for (auto& e : entries)
		CHECK(e->builds == 0);
	entries.clear();
}

static void Pooled()
{
	// the rounds a game's new pipelines come in: the first fills the pool,
	// the rest take their tickets from it
	const int perRound = 300;
	uint64_t laterAllocs = 0;
	for (int round = 0; round < 4; round++)
	{
		std::vector<std::unique_ptr<Entry>> entries;
		for (int i = 0; i < perRound; i++)
			entries.emplace_back(new Entry());
		Pool pool;
		pool.Start(2);
		std::vector<PreparedPipeline::Ticket*> tickets;
		tickets.reserve(perRound);
		const uint64_t before = s_allocs.load();
		for (auto& e : entries)
			tickets.push_back(e->prepared.Queue());
		const uint64_t allocs = s_allocs.load() - before;
		for (PreparedPipeline::Ticket* t : tickets)
			pool.queue.push(t);
		for (auto& e : entries)
			CHECK(e->prepared.Take() == (VkPipeline)(uintptr_t)(e->id + 1));
		pool.Stop();
		entries.clear();
		if (round)
			laterAllocs += allocs;
		else
			std::printf("tickets: %d queued first, %llu allocations\n", perRound, (unsigned long long)allocs);
	}
	std::printf("tickets: %d queued later, %llu allocations\n", perRound * 3, (unsigned long long)laterAllocs);
	CHECK(laterAllocs == 0);
}

// a build that takes a while and counts how many run at once
static std::atomic<int> s_running{0};
static std::atomic<int> s_mostRunning{0};
struct SlowEntry
{
	std::atomic<int> builds{0};
	std::atomic<int> onCompileThread{-1};
	PreparedPipeline prepared{[](void* self, bool onCompileThread) {
		SlowEntry* e = (SlowEntry*)self;
		const int now = ++s_running;
		int most = s_mostRunning.load();
		while (now > most && !s_mostRunning.compare_exchange_weak(most, now))
			;
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		--s_running;
		e->onCompileThread = onCompileThread;
		e->builds++;
		return (VkPipeline)(uintptr_t)1;
	}, this};
};

static void Together()
{
	// compile threads free: the queued ones built there, overlapping the
	// ones built here
	{
		Pool pool;
		pool.Start(2);
		SlowEntry e[4];
		PreparedPipeline* here[2] = { &e[0].prepared, &e[1].prepared };
		PreparedPipeline* queued[2] = { &e[2].prepared, &e[3].prepared };
		s_mostRunning = 0;
		const auto t0 = std::chrono::steady_clock::now();
		PreparedPipeline::BuildTogether(here, 2, queued, 2, [&](PreparedPipeline& item) { pool.queue.push(item.Queue()); });
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		for (auto& x : e)
			CHECK(x.builds == 1 && x.prepared.IsBuilt());
		CHECK(e[0].onCompileThread == 0 && e[1].onCompileThread == 0);
		CHECK(s_mostRunning >= 2);
		std::printf("together: 4 builds of 20 ms in %.0f ms, %d at once at most\n", ms, s_mostRunning.load());
		pool.Stop();
	}
	// no compile thread: all here, one after the other, none waited for
	{
		Pool pool; // never started
		SlowEntry e[3];
		PreparedPipeline* here[1] = { &e[0].prepared };
		PreparedPipeline* queued[2] = { &e[1].prepared, &e[2].prepared };
		s_mostRunning = 0;
		PreparedPipeline::BuildTogether(here, 1, queued, 2, [&](PreparedPipeline& item) { pool.queue.push(item.Queue()); });
		for (auto& x : e)
			CHECK(x.builds == 1 && x.onCompileThread == 0);
		CHECK(s_mostRunning == 1);
		pool.Stop(); // the tickets left in the queue find nothing to build
		for (auto& x : e)
			CHECK(x.builds == 1);
	}
}

static void Urgent()
{
	struct Item
	{
		int id;
	};
	MpmcQueue<Item*> urgent, rest;
	Item items[12];
	for (int i = 0; i < 12; i++)
		items[i].id = i;
	// a backlog, then an urgent one: the urgent one first, the backlog in order
	for (int i = 0; i < 10; i++)
		rest.push(&items[i]);
	MpmcQueue_PushUrgent(urgent, rest, &items[10]);
	std::vector<int> order;
	while (order.size() < 11)
		if (Item* it = MpmcQueue_PopPreferring(urgent, rest))
			order.push_back(it->id);
	CHECK(order.size() == 11 && order[0] == 10);
	bool inOrder = true;
	for (int i = 1; i < 11; i++)
		inOrder &= order[i] == i - 1;
	CHECK(inOrder);
	// its wake, left in rest, passed over: nothing else there
	// (read without waiting, so that a wrong order fails rather than hangs)
	CHECK(rest.size() == 1);
	Item* wake = &items[0];
	CHECK(rest.peek2(wake) && wake == nullptr && rest.empty() && urgent.empty());
	// an idle pool: the thread waiting on rest wakes for an urgent one
	std::atomic<int> got{-1};
	std::thread worker([&] {
		while (true)
		{
			Item* it = MpmcQueue_PopPreferring(urgent, rest);
			if (it)
			{
				got = it->id;
				return;
			}
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(20)); // asleep on rest by now
	MpmcQueue_PushUrgent(urgent, rest, &items[11]);
	worker.join();
	CHECK(got == 11);
}

static void Lanes()
{
	struct Item
	{
		int id;
	};
	MpmcQueue<Item*> urgent, soon, later;
	Item items[31];
	for (int i = 0; i < 31; i++)
		items[i].id = i;
	// pushed interleaved: later 0-9, soon 10-19, urgent 20-29
	for (int i = 0; i < 10; i++)
	{
		later.push(&items[i]);
		MpmcQueue_PushUrgent(soon, later, &items[10 + i]);
		MpmcQueue_PushUrgent(urgent, later, &items[20 + i]);
	}
	std::vector<int> order;
	int wakes = 0;
	while (order.size() < 30)
	{
		if (Item* it = MpmcQueue_PopPreferring(urgent, soon, later))
			order.push_back(it->id);
		else
			wakes++;
	}
	bool inOrder = order.size() == 30;
	for (int i = 0; inOrder && i < 10; i++)
		inOrder = order[i] == 20 + i && order[10 + i] == 10 + i && order[20 + i] == i;
	CHECK(inOrder);
	// every wake taken came after the items it was for, nothing left
	CHECK(urgent.empty() && soon.empty());
	CHECK(wakes + (int)later.size() == 20);
	while (!later.empty())
	{
		Item* wake = &items[0];
		CHECK(later.peek2(wake) && wake == nullptr);
	}
	// an idle pool: the thread waiting on later wakes for one pushed to soon
	std::atomic<int> got{-1};
	std::thread worker([&] {
		while (true)
		{
			if (Item* it = MpmcQueue_PopPreferring(urgent, soon, later))
			{
				got = it->id;
				return;
			}
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(20)); // asleep on later by now
	MpmcQueue_PushUrgent(soon, later, &items[30]);
	worker.join();
	CHECK(got == 30);
}

static void GivenBack()
{
	RecycledVectors<uint32_t> buffers;
	for (size_t room : {16, 256, 4096})
	{
		std::vector<uint32_t> b;
		b.reserve(room);
		b.push_back(1);
		buffers.Give(b);
		CHECK(b.empty() && b.capacity() == 0);
	}
	CHECK(buffers.Count() == 3);
	const uint64_t before = s_allocs.load();
	// the one with room, behind two smaller ones; the module written into it
	std::vector<uint32_t> big;
	buffers.Take(big, 1000);
	CHECK(big.capacity() >= 1000 && big.empty() && buffers.Count() == 2);
	big.assign(1000, 7);
	// the first with room, past the one too small
	std::vector<uint32_t> mid;
	buffers.Take(mid, 256); // its room exactly
	CHECK(mid.capacity() >= 256 && mid.capacity() < 1000 && buffers.Count() == 1);
	CHECK(s_allocs.load() == before); // nothing allocated taking them
	// none large enough: one of them all the same, as before
	std::vector<uint32_t> small;
	buffers.Take(small, 100000);
	CHECK(small.capacity() >= 16 && small.capacity() < 100 && buffers.Count() == 0);
	// nothing given back: out as it was
	std::vector<uint32_t> none;
	buffers.Take(none, 10);
	CHECK(none.capacity() == 0);
	// one with room of its own kept
	buffers.Give(mid);
	buffers.Take(big, 1);
	CHECK(big.size() == 1000 && buffers.Count() == 1);
	// without a size: the first given back
	buffers.Give(small);
	std::vector<uint32_t> first;
	buffers.Take(first);
	CHECK(first.capacity() >= 100 && first.capacity() < 1000);
	// room exactly what is needed is room enough, the smaller ones behind it
	// left
	RecycledVectors<uint32_t> exact;
	for (size_t room : {512, 16, 8})
	{
		std::vector<uint32_t> b;
		b.reserve(room);
		exact.Give(b);
	}
	std::vector<uint32_t> fits;
	exact.Take(fits, fits.capacity() + 512);
	CHECK(fits.capacity() >= 512 && exact.Count() == 2);
	// strings too (a shader's source), past their own small room
	RecycledStrings sources;
	for (size_t room : {64, 8192})
	{
		std::string str;
		str.reserve(room);
		str = "x";
		sources.Give(str);
	}
	std::string src;
	sources.Take(src, 5000);
	CHECK(src.capacity() >= 5000 && sources.Count() == 1);
}

int main()
{
	MadeAhead();
	Racing();
	Retired();
	Dropped();
	Pooled();
	Together();
	Urgent();
	Lanes();
	GivenBack();
	if (s_failures)
	{
		std::printf("%d failed\n", s_failures);
		return 1;
	}
	std::printf("prepared_pipeline: all passed\n");
	return 0;
}
