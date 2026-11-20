// NFC version 0 tags (TagV0.cpp): tags with random blocks locked - the UID,
// the reserved and the lock byte blocks kept as they are, the locked blocks'
// data apart, the rest the capability container and an NDEF TLV filling it -
// read and written back the same, byte for byte, the locked area and the
// NDEF data as written; and a tag with no NDEF TLV refused.
#include "Cafe/OS/libs/nfc/TagV0.h"

#include <cstdio>
#include <random>
#include <vector>

uint64 s_loggingFlagMask = 0;
bool cemuLog_log(LogType, std::string_view) { return false; }

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { if (s_failures < 20) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

static bool Reserved(int b) { return b == 0 || b == 0xd || b == 0xe || b == 0xf; }

int main()
{
	std::mt19937 rng(17);
	int tested = 0;
	for (int round = 0; round < 3000; round++)
	{
		std::vector<std::byte> tag(512);
		for (auto& b : tag)
			b = std::byte(rng());
		// lock bits: block i locked by bit i, 16 in block 0xe, 48 in 0xf
		uint64 locks = 0;
		for (int b = 0x10; b < 64; b++)
			if (rng() % 4 == 0)
				locks |= 1ull << b;
		if (round % 5 == 0)
			locks |= 0xFull << 1; // blocks 1 to 4 too, below the reserved ones
		locks |= 1ull << 0 | 1ull << 0xd | 1ull << 0xe | 1ull << 0xf; // their bits as a tag has them
		for (int i = 0; i < 2; i++)
			tag[0xe * 8 + i] = std::byte(locks >> (8 * i));
		for (int i = 0; i < 6; i++)
			tag[0xf * 8 + 2 + i] = std::byte(locks >> (16 + 8 * i));
		// the data area: the blocks neither locked nor reserved, in order
		std::vector<int> dataBlocks;
		std::vector<std::byte> locked;
		for (int b = 0; b < 64; b++)
		{
			if (Reserved(b))
				continue;
			if (locks >> b & 1)
				locked.insert(locked.end(), tag.begin() + b * 8, tag.begin() + b * 8 + 8);
			else
				dataBlocks.push_back(b);
		}
		const size_t area = dataBlocks.size() * 8;
		if (area < 16)
			continue;
		std::vector<std::byte> data(area);
		data[0] = std::byte(0xe1);
		data[1] = std::byte(0x10);
		data[2] = std::byte(0x3f);
		data[3] = std::byte(0x00);
		// the NDEF TLV's length in one byte under 0xff, as it is written back,
		// else 0xff and two bytes
		const bool shortLength = area - 4 - 2 < 0xff;
		const size_t header = shortLength ? 6 : 8;
		const size_t payload = area - header;
		data[4] = std::byte(0x03); // NDEF
		if (shortLength)
			data[5] = std::byte(payload);
		else
		{
			data[5] = std::byte(0xff);
			data[6] = std::byte(payload >> 8);
			data[7] = std::byte(payload);
		}
		for (size_t i = header; i < area; i++)
			data[i] = std::byte(rng());
		for (size_t i = 0; i < dataBlocks.size(); i++)
			std::copy_n(data.begin() + i * 8, 8, tag.begin() + dataBlocks[i] * 8);

		auto parsed = TagV0::FromBytes(tag);
		CHECK(parsed != nullptr);
		if (!parsed)
			continue;
		tested++;
		{
			const auto out = parsed->ToBytes();
			if (out != tag && s_failures == 0)
				for (size_t i = 0; i < 512; i++)
					if (out[i] != tag[i]) { std::printf("round %d byte %zu block %zu: %02x vs %02x, area %zu locks %016llx\n", round, i, i / 8, (int)out[i], (int)tag[i], area, (unsigned long long)locks); break; }
			CHECK(out == tag);
		}
		CHECK(parsed->GetLockedArea() == locked);
		CHECK(parsed->GetNDEFData() == std::vector<std::byte>(data.begin() + header, data.end()));
		CHECK(std::equal(parsed->GetUIDBlock().begin(), parsed->GetUIDBlock().end(), tag.begin()));
	}
	// no NDEF TLV: refused
	{
		std::vector<std::byte> tag(512);
		const int blocks[] = { 1, 2 };
		std::byte data[16] = { std::byte(0xe1), std::byte(0x10), std::byte(0x3f), std::byte(0), std::byte(0xfe) };
		for (int i = 0; i < 2; i++)
			std::copy_n(data + i * 8, 8, tag.begin() + blocks[i] * 8);
		CHECK(TagV0::FromBytes(tag) == nullptr);
	}
	std::printf("nfc_tag: %d tags read and written back\n", tested);
	CHECK(tested > 2500);
	if (s_failures)
	{
		std::printf("nfc_tag_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("nfc_tag_test: all passed\n");
	return 0;
}
