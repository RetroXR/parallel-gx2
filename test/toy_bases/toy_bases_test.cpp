// The toy bases taking figures from a frontend (FigureMail.h, hid_portal's
// host commands): a figure put on is reported to the title as upstream's
// LoadSkylander / LoadFigure report it (the portal's status ADDED then READY,
// the Infinity Base's and the Toy Pad's "added" messages), a title's write to
// a figure comes back to save, a figure taken off is reported gone, a
// returning Skylander takes its old slot, and figures put on and taken off
// from one thread while another sends the title's commands all land, the
// state touched by one runner at a time.

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#include "Cafe/OS/libs/nsyshid/Skylander.h"
#include "Cafe/OS/libs/nsyshid/Infinity.h"
#include "Cafe/OS/libs/nsyshid/Dimensions.h"

static int s_failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); s_failures++; } } while (0)

using namespace nsyshid;

static uint32 SlotStatus(const std::array<uint8, 64>& r, int slot)
{
	uint32 word;
	memcpy(&word, &r[1], 4);
	return (word >> (slot * 2)) & 3;
}

static std::array<uint8, SKY_FIGURE_SIZE> Skylander(uint32 serial, uint8 fill)
{
	std::array<uint8, SKY_FIGURE_SIZE> f{};
	f.fill(fill);
	memcpy(f.data(), &serial, 4);
	return f;
}

static void TestPortal()
{
	SkylanderUSB& p = g_skyportal;
	p.Start();
	const auto a = Skylander(0x11223344, 0xA1);
	CHECK(p.PlaceFigure(0, a.data()));
	// status reports: ADDED once, then READY
	auto r = p.GetStatus();
	CHECK(r[0] == 0x53 && SlotStatus(r, 0) == SkylanderUSB::Skylander::ADDED);
	r = p.GetStatus();
	CHECK(SlotStatus(r, 0) == SkylanderUSB::Skylander::READY);
	// the title reads block 2 and gets the figure's bytes
	const uint8 query[] = {'Q', 0x10, 2};
	p.Submit(query, sizeof(query));
	r = p.GetStatus();
	CHECK(r[0] == 'Q' && r[1] == 0x10 && r[2] == 2 && r[3] == 0xA1);
	// the title writes block 5: it comes back to save, once
	uint8 write[19] = {'W', 0x10, 5};
	memset(write + 3, 0x5E, 16);
	p.Submit(write, sizeof(write));
	(void)p.GetStatus(); // the write's answer
	std::array<uint8, SKY_FIGURE_SIZE> saved{};
	CHECK(p.TakeSave(0, saved.data()));
	CHECK(saved[5 * 16] == 0x5E && saved[5 * 16 + 15] == 0x5E && saved[4 * 16] == 0xA1);
	CHECK(!p.TakeSave(0, saved.data()));
	// a second figure on another spot takes the next slot
	const auto b = Skylander(0x55667788, 0xB2);
	CHECK(p.PlaceFigure(1, b.data()));
	r = p.GetStatus();
	CHECK(SlotStatus(r, 1) == SkylanderUSB::Skylander::ADDED && SlotStatus(r, 0) == SkylanderUSB::Skylander::READY);
	// taken off: REMOVING once, then REMOVED
	CHECK(p.RemoveFigure(0));
	r = p.GetStatus();
	CHECK(SlotStatus(r, 0) == SkylanderUSB::Skylander::REMOVING);
	r = p.GetStatus();
	CHECK(SlotStatus(r, 0) == SkylanderUSB::Skylander::REMOVED);
	// a query for it now finds nothing
	p.Submit(query, sizeof(query));
	r = p.GetStatus();
	CHECK(r[0] == 'Q' && r[1] == 0x00);
	// b sits in slot 1. Taken off, slots 0 and 1 are both free; put back (from
	// another spot), b takes slot 1, its own, not the first free one
	CHECK(p.RemoveFigure(1));
	(void)p.GetStatus();
	(void)p.GetStatus();
	CHECK(p.PlaceFigure(5, b.data()));
	r = p.GetStatus();
	CHECK(SlotStatus(r, 1) == SkylanderUSB::Skylander::ADDED && SlotStatus(r, 0) == SkylanderUSB::Skylander::REMOVED);
	p.Close();
	p.Stop();
}

static void TestPortalBeforeStart()
{
	SkylanderUSB& p = g_skyportal;
	p.DropPending();
	const auto a = Skylander(0x0BADF00D, 0x77);
	CHECK(p.PlaceFigure(4, a.data()));  // the title has not started the portal
	p.Start();
	const auto r = p.GetStatus();
	CHECK(SlotStatus(r, 0) == SkylanderUSB::Skylander::ADDED);
	p.Close();
	p.Stop();
}

// figures put on and off from this thread while another sends the title's
// queries; every figure placed is seen, nothing torn
static void TestPortalUnderLoad()
{
	SkylanderUSB& p = g_skyportal;
	p.DropPending();
	p.Start();
	std::atomic<bool> stop{false};
	std::thread title([&] {
		const uint8 q[] = {'Q', 0x10, 0};
		while (!stop.load())
		{
			p.Submit(q, sizeof(q));
			(void)p.GetStatus();
		}
	});
	int placed = 0;
	for (int i = 0; i < 2000; i++)
	{
		const auto f = Skylander(0x1000 + i, uint8(i));
		if (p.PlaceFigure(uint8(i % 8), f.data()))
			placed++;
		if (i % 3 == 0)
			p.RemoveFigure(uint8((i + 4) % 8));
	}
	stop = true;
	title.join();
	CHECK(placed == 2000);
	p.Close();
	p.Stop();
}

static void TestInfinity()
{
	InfinityUSB& b = g_infinitybase;
	b.Start();
	std::array<uint8, INF_FIGURE_SIZE> f{};
	f.fill(0x42);
	CHECK(b.PlaceFigure(4, f.data()));   // player one's first ability
	std::array<uint8, 32> r{};
	CHECK(b.GetStatus(r));
	// added: pad 2 (player one), order 0, way 0
	CHECK(r[0] == 0xab && r[1] == 0x04 && r[2] == 2 && r[3] == 0x09 && r[4] == 0 && r[5] == 0x00);
	CHECK(b.PlaceFigure(0, f.data()));   // the hexagon
	CHECK(b.GetStatus(r));
	CHECK(r[2] == 1 && r[4] == 1);
	CHECK(b.RemoveFigure(4));
	CHECK(b.GetStatus(r));
	CHECK(r[0] == 0xab && r[2] == 2 && r[4] == 0 && r[5] == 0x01);
	b.Close();
	b.Stop();
}

static void TestToyPad()
{
	DimensionsUSB& t = g_dimensionstoypad;
	t.Start();
	std::array<uint8, DIM_FIGURE_SIZE> f{};
	for (size_t i = 0; i < f.size(); i++)
		f[i] = uint8(i * 7);
	CHECK(t.PlaceFigure(5, f.data()));   // the right pad
	std::array<uint8, 32> r{};
	CHECK(t.GetStatus(r));
	// added: pad 3, index 6, way 0, and the UID (bytes 0-2, 4-7)
	CHECK(r[0] == 0x56 && r[1] == 0x0b && r[2] == 3 && r[4] == 6 && r[5] == 0x00);
	CHECK(r[6] == f[0] && r[7] == f[1] && r[8] == f[2] && r[9] == f[4] && r[12] == f[7]);
	CHECK(t.RemoveFigure(5));
	CHECK(t.GetStatus(r));
	CHECK(r[0] == 0x56 && r[2] == 3 && r[4] == 6 && r[5] == 0x01);
	CHECK(DimensionsUSB::PadOf(1) == 1 && DimensionsUSB::PadOf(0) == 2 && DimensionsUSB::PadOf(6) == 3);
	t.Close();
	t.Stop();
}

int main()
{
	TestPortal();
	TestPortalBeforeStart();
	TestPortalUnderLoad();
	TestInfinity();
	TestToyPad();
	if (s_failures)
	{
		fprintf(stderr, "toy_bases: %d failure(s)\n", s_failures);
		return 1;
	}
	printf("toy_bases: ok\n");
	return 0;
}
