// The guest's sockets (nsysnet.cpp): every handle there is made, the one past
// them refused rather than trapped or written before the table, each closed
// and made again in its place; and a game opening and closing sockets while
// it plays allocating nothing for them.
#include "Cafe/OS/libs/nsysnet/nsysnet.h"
#include "Cafe/HW/Espresso/PPCState.h"

#include <atomic>
#if !BOOST_OS_WINDOWS
#include <netinet/in.h>
#endif
#include <cstdio>
#include <cstdlib>

WUSOCKET nsysnet_createVirtualSocket(sint32 family, sint32 type, sint32 protocol);
void nsysnetExport_socketclose(PPCInterpreter_t* hCPU);

static std::atomic<unsigned long long> s_mallocs{0};
extern "C" void* __real_malloc(size_t);
extern "C" void __real_free(void*);
extern "C" void* __wrap_malloc(size_t n)
{
	s_mallocs++;
	return __real_malloc(n);
}
extern "C" void __wrap_free(void* p) { __real_free(p); }

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

static void Close(WUSOCKET s)
{
	PPCInterpreter_t cpu{};
	cpu.gpr[3] = (uint32)s;
	nsysnetExport_socketclose(&cpu);
}

int main()
{
#if BOOST_OS_WINDOWS
	WSADATA wsa;
	WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
	constexpr int kLimit = 32;
	WUSOCKET handles[kLimit];
	for (int i = 0; i < kLimit; i++)
	{
		handles[i] = nsysnet_createVirtualSocket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
		CHECK(handles[i] == i + 1);
	}
	// none left: refused
	CHECK(nsysnet_createVirtualSocket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) == -1);
	// one closed, its handle made again
	Close(handles[9]);
	CHECK(nsysnet_createVirtualSocket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) == 10);
	for (int i = 0; i < kLimit; i++)
		Close(handles[i]);
	// sockets opened and closed: nothing of nsysnet's from the heap
	const unsigned long long before = s_mallocs;
	for (int round = 0; round < 200; round++)
	{
		WUSOCKET s[4];
		for (WUSOCKET& h : s)
		{
			h = nsysnet_createVirtualSocket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
			CHECK(h > 0);
		}
		for (WUSOCKET h : s)
			Close(h);
	}
	std::printf("nsysnet: %llu allocations for 800 sockets opened and closed\n", s_mallocs - before);
	CHECK(s_mallocs == before);
	if (s_failures)
	{
		std::printf("nsysnet_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("nsysnet_test: all passed\n");
	return 0;
}
