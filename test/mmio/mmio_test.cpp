// The MMIO handlers (MMU.cpp), found by address in a table in place: those
// registered by static initializers, as the hardware's files register theirs,
// before the table's own file is initialized; each access size apart; the
// first one for an address kept; an unregistered address read as 0 and its
// writes dropped; many registered in any order all found; and none of it
// allocating.
#include "Cafe/HW/MMU/MMU.h"

#include <cstdio>
#include <cstdlib>
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

static uint32 s_written = 0;
static PAddr s_writtenAt = 0;
static uint32 R32a(PAddr) { return 0x11111111; }
static uint32 R32b(PAddr) { return 0x22222222; }
static uint16 R16a(PAddr) { return 0x3333; }
static void W32a(PAddr a, uint32 v) { s_written = v; s_writtenAt = a; }
static void W16a(PAddr a, uint16 v) { s_written = v | 0x10000; s_writtenAt = a; }
static uint32 R32Echo(PAddr a) { return a ^ 0x5A5A5A5A; }

// as ACR.cpp and SI.cpp register theirs: from a static initializer
static struct EarlyRegistration
{
	EarlyRegistration()
	{
		MMU::RegisterMMIO_R32(MMU::MMIOInterface::INTERFACE_0D000000, 0x21C, R32a);
		MMU::RegisterMMIO_W32(MMU::MMIOInterface::INTERFACE_0D000000, 0x21C, W32a);
	}
} s_early;

int main()
{
	using MMU::MMIOInterface;
	CHECK(MMU::ReadMMIO_32(0x0D00021C) == 0x11111111);
	// the first stays
	MMU::RegisterMMIO_R32(MMIOInterface::INTERFACE_0D000000, 0x21C, R32b);
	CHECK(MMU::ReadMMIO_32(0x0D00021C) == 0x11111111);
	MMU::WriteMMIO_32(0x0D00021C, 0xCAFEF00D);
	CHECK(s_written == 0xCAFEF00D && s_writtenAt == 0x0D00021C);
	// sizes apart, interfaces apart
	MMU::RegisterMMIO_R16(MMIOInterface::INTERFACE_0C000000, 0x21C, R16a);
	MMU::RegisterMMIO_W16(MMIOInterface::INTERFACE_0C000000, 0x21C, W16a);
	CHECK(MMU::ReadMMIO_16(0x0C00021C) == 0x3333);
	CHECK(MMU::ReadMMIO_16(0x0D00021C) == 0);
	CHECK(MMU::ReadMMIO_32(0x0C00021C) == 0);
	MMU::WriteMMIO_16(0x0C00021C, 0x1234);
	CHECK(s_written == 0x11234 && s_writtenAt == 0x0C00021C);
	// unregistered: 0, the write dropped
	s_written = 0;
	CHECK(MMU::ReadMMIO_32(0x0D000400) == 0);
	MMU::WriteMMIO_32(0x0D000400, 7);
	CHECK(s_written == 0);
	// many, in any order: all found
	std::mt19937 rng(5);
	std::vector<uint32> offsets;
	for (uint32 i = 0; i < 40; i++)
		offsets.push_back(0x1000 + i * 4);
	std::shuffle(offsets.begin(), offsets.end(), rng);
	for (uint32 off : offsets)
		MMU::RegisterMMIO_R32(MMIOInterface::INTERFACE_0C000000, off, R32Echo);
	const unsigned long long before = s_allocs;
	for (int round = 0; round < 1000; round++)
		for (uint32 i = 0; i < 40; i++)
		{
			const PAddr a = 0x0C000000 + 0x1000 + i * 4;
			CHECK(MMU::ReadMMIO_32(a) == (a ^ 0x5A5A5A5A));
		}
	CHECK(MMU::ReadMMIO_32(0x0C000000 + 0x1000 + 40 * 4) == 0);
	CHECK(MMU::ReadMMIO_32(0x0D00021C) == 0x11111111);
	std::printf("mmio: %llu allocations for 40000 reads\n", s_allocs - before);
	CHECK(s_allocs == before);
	if (s_failures)
	{
		std::printf("mmio_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("mmio_test: all passed\n");
	return 0;
}
