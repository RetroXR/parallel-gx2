// The fibers the emulated cores run guest threads on (src/util/Fiber), as the
// pool of host threads uses them: a fiber that ran a thread to wherever it
// stopped is started over for the next one (Fiber::Restart) - from its entry,
// as many times as threads come and go, its stack used to the bottom each
// time - and the running fiber's data is found, or nothing on a thread that
// prepared none (TryGetFiberPrivateData). Built on each backend this machine
// can run: ucontext, libco (Android's) and Windows' own fibers under wine.
#include "util/Fiber/Fiber.h"

#include <cstdio>
#include <cstring>
#include <thread>

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct State
{
	Fiber* main = nullptr;
	Fiber* fiber = nullptr;
	int generation = 0;    // set by the test before each start
	int entries = 0;       // times the entry ran from the top
	int seenGeneration = -1;
	long switches = 0;
	bool privateDataSeen = false;
	unsigned depthSum = 0;
};

static State s_state;

// uses the stack to some depth, so a restart over a used stack shows
static unsigned Deep(unsigned n)
{
	volatile unsigned char buf[4096];
	std::memset((void*)buf, (int)n, sizeof(buf));
	if (n == 0)
		return buf[17];
	return Deep(n - 1) + buf[4095];
}

static void Entry(void* param)
{
	State* st = (State*)param;
	st->entries++;
	st->seenGeneration = st->generation;
	st->privateDataSeen = Fiber::TryGetFiberPrivateData() == st && Fiber::GetFiberPrivateData() == st;
	st->depthSum += Deep(64); // 256 KB
	for (;;)
	{
		st->switches++;
		Fiber::Switch(*st->main);
	}
}

int main()
{
	// a thread with no fiber prepared has no fiber data
	std::thread([] { CHECK(Fiber::TryGetFiberPrivateData() == nullptr); }).join();

	s_state.main = Fiber::PrepareCurrentThread();
	CHECK(Fiber::TryGetFiberPrivateData() == nullptr);
	s_state.fiber = new Fiber(Entry, &s_state, &s_state);
	for (int i = 0; i < 3; i++)
		Fiber::Switch(*s_state.fiber);
	CHECK(s_state.entries == 1);
	CHECK(s_state.switches == 3);
	CHECK(s_state.privateDataSeen);

	// started over, again and again: each time from the top
	for (int round = 1; round <= 2000; round++)
	{
		s_state.generation = round;
		s_state.privateDataSeen = false;
		s_state.fiber->Restart(Entry, &s_state);
		const long before = s_state.switches;
		Fiber::Switch(*s_state.fiber);
		Fiber::Switch(*s_state.fiber);
		CHECK(s_state.entries == round + 1);
		CHECK(s_state.seenGeneration == round);
		CHECK(s_state.switches == before + 2);
		CHECK(s_state.privateDataSeen);
	}
	CHECK(Fiber::TryGetFiberPrivateData() == nullptr);
	delete s_state.fiber;

	if (s_failures)
	{
		std::printf("fiber: %d failure(s)\n", s_failures);
		return 1;
	}
	std::printf("fiber: ok (%d starts)\n", s_state.entries);
	return 0;
}
