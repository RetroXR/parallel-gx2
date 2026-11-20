#pragma once

#include <memory>
#include <span>
#include <array>

#include "TLV.h"

class TagV0
{
public:
	using Block = std::array<std::byte, 0x8>;

public:
	TagV0();
	virtual ~TagV0();

	static std::shared_ptr<TagV0> FromBytes(const std::span<const std::byte>& data);
	std::vector<std::byte> ToBytes() const;

	const Block& GetUIDBlock() const;
	const std::vector<std::byte>& GetNDEFData() const;
	const std::vector<std::byte>& GetLockedArea() const;

	void SetNDEFData(const std::span<const std::byte>& data);

private:
	bool ParseLockedArea(const std::span<const std::byte>& data);
	bool IsBlockLocked(uint8 blockIdx) const;
	bool ParseDataArea(const std::span<const std::byte>& data, std::vector<std::byte>& dataArea);
	bool ValidateCapabilityContainer();

	// the tag's 64 blocks: which are locked (or locked and reserved), a bit
	// each, and their data in place
	static constexpr std::size_t kBlocks = 64;
	void SetBlock(uint64& mask, std::array<Block, kBlocks>& blocks, uint8 index, const Block& block);
	uint64 mLockedOrReservedMask = 0;
	std::array<Block, kBlocks> mLockedOrReservedBlocks{};
	uint64 mLockedMask = 0;
	std::array<Block, kBlocks> mLockedBlocks{};
	std::array<uint8, 0x4> mCapabilityContainer;
	std::vector<TLV> mTLVs;
	std::size_t mNdefTlvIdx;
	std::vector<std::byte> mLockedArea;
};
