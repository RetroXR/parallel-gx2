// What a thread keeps for itself, freed as it ends with no thread_local
// destructor (src/util/helpers/thread_end.c, ThreadOwned.h): MinGW frees a
// thread_local's storage before it runs its destructor, so the core's
// thread_local buffers, decoders and hazard slots read freed memory as a
// thread ended. Each check below runs on threads that end:
//  - objects made with ThreadOwned on many threads, each with a canary its
//    destructor checks, every one destroyed once, intact, as its thread ends
//    (as the thread_local objects were not, under MinGW);
//  - a function registered with an argument run once per thread, with it;
//  - thread_end_run: run now, newest first, the thread's pointers emptied,
//    and made again on the next use;
//  - at most THREAD_END_MAX a thread, the rest refused (kept, never freed);
//  - thread_end_deinit: the caller's run, nothing kept after, and kept
//    again once thread_end_init starts it over.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "util/helpers/ThreadOwned.h"

static int s_failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); s_failures++; } } while (0)

static std::atomic<int> s_live{0};
static std::atomic<int> s_destroyed{0};
static std::atomic<int> s_torn{0}; // destroyed with its canary gone: freed storage

// large enough that freed storage is written over, as the decompiler's writer was
struct Owned
{
	static constexpr unsigned kCanary = 0x5EEDF00Du;
	unsigned canary = kCanary;
	unsigned words[60]{};
	std::vector<int> held;
	explicit Owned(int n = 16) : held((size_t)n, 7) { s_live++; }
	~Owned()
	{
		if (canary != kCanary || held.size() == 0 || held[0] != 7)
			s_torn++;
		canary = 0;
		s_live--;
		s_destroyed++;
	}
};

static Owned& OwnedA()
{
	thread_local Owned* t_a;
	return ThreadOwned(t_a);
}
static Owned& OwnedB()
{
	thread_local Owned* t_b;
	return ThreadOwned(t_b, 1000);
}
static std::vector<int>& Scratch()
{
	thread_local std::vector<int>* t_v;
	return ThreadOwned(t_v);
}

static void ManyThreads()
{
	s_destroyed = 0;
	s_torn = 0;
	const int kThreads = 64;
	std::vector<std::thread> threads;
	for (int t = 0; t < kThreads; t++)
	{
		threads.emplace_back([t] {
			// the same object on every use, made once
			Owned* a = &OwnedA();
			CHECK(&OwnedA() == a);
			OwnedB().held[0] = 7;
			Scratch().assign(4096, t);
			CHECK(thread_end_count() == 3);
		});
	}
	for (auto& th : threads)
		th.join();
	CHECK(s_live == 0);
	CHECK(s_destroyed == kThreads * 2);
	CHECK(s_torn == 0);
	std::printf("thread_end: %d threads ended, %d objects destroyed as they ended, %d torn\n", kThreads, s_destroyed.load(), s_torn.load());
}

static std::atomic<int> s_ran{0};
static std::atomic<long> s_argSum{0};

static void Ran(void* arg)
{
	s_ran++;
	s_argSum += (long)(intptr_t)arg;
}

static void RegisteredFunctions()
{
	s_ran = 0;
	s_argSum = 0;
	std::vector<std::thread> threads;
	for (int t = 1; t <= 32; t++)
		threads.emplace_back([t] { CHECK(thread_end_register(Ran, (void*)(intptr_t)t, nullptr)); });
	for (auto& th : threads)
		th.join();
	CHECK(s_ran == 32);
	CHECK(s_argSum == 32 * 33 / 2);
}

static std::vector<int> s_order;
static void Order(void* arg)
{
	s_order.push_back((int)(intptr_t)arg);
}

static void RunNow()
{
	std::thread([] {
		Owned* first = &OwnedA();
		s_order.clear();
		CHECK(thread_end_register(Order, (void*)(intptr_t)1, nullptr));
		CHECK(thread_end_register(Order, (void*)(intptr_t)2, nullptr));
		const int destroyed = s_destroyed;
		thread_end_run();
		// newest first, the object gone, the list empty
		CHECK(s_order.size() == 2 && s_order[0] == 2 && s_order[1] == 1);
		CHECK(s_destroyed == destroyed + 1);
		CHECK(thread_end_count() == 0);
		// made again, and freed as the thread ends
		Owned* again = &OwnedA();
		CHECK(again->canary == Owned::kCanary);
		(void)first;
		CHECK(thread_end_count() == 1);
	}).join();
	CHECK(s_live == 0);
}

static void Full()
{
	std::thread([] {
		int accepted = 0;
		for (int i = 0; i < THREAD_END_MAX + 8; i++)
			accepted += thread_end_register(Ran, nullptr, nullptr) ? 1 : 0;
		CHECK(accepted == THREAD_END_MAX);
	}).join();
}

static void DeinitAndInit()
{
	// this thread's run at the end, nothing kept after it
	Owned* mine = &OwnedA();
	(void)mine;
	CHECK(s_live == 1);
	thread_end_deinit();
	CHECK(s_live == 0);
	CHECK(!thread_end_register(Ran, nullptr, nullptr));
	std::thread([] { CHECK(!thread_end_register(Ran, nullptr, nullptr)); }).join();
	// started over: kept and run again
	CHECK(thread_end_init());
	s_ran = 0;
	std::thread([] { CHECK(thread_end_register(Ran, nullptr, nullptr)); }).join();
	CHECK(s_ran == 1);
	OwnedA();
	thread_end_run();
	CHECK(s_live == 0);
}

int main()
{
	// no thread_end_init: made at the first registration
	ManyThreads();
	RegisteredFunctions();
	RunNow();
	Full();
	DeinitAndInit();
	if (s_failures)
	{
		std::printf("thread_end_test: %d failure(s)\n", s_failures);
		return 1;
	}
	std::printf("thread_end_test: all passed\n");
	return 0;
}
