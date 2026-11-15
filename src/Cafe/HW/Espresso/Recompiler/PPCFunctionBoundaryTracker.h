#pragma once
#include "Cafe/HW/Espresso/EspressoISA.h"
#include "Cafe/HW/MMU/MMU.h"
#include "Cafe/HW/Espresso/Recompiler/IML/IMLScratch.h"

#include <algorithm>

bool GamePatch_IsNonReturnFunction(uint32 hleIndex);

// utility class to determine shape of a function. Its ranges and branch
// targets are sorted vectors, their storage the compiling thread's from one
// function to the next (IMLScratch.h), where sets allocated a node apiece.
class PPCFunctionBoundaryTracker
{
public:
	struct PPCRange_t
	{
		PPCRange_t() = default;
		PPCRange_t(uint32 _startAddress) : startAddress(_startAddress) {};

		uint32 startAddress{};
		uint32 length{};
		//bool isProcessed{false};

		uint32 getEndAddress() const { return startAddress + length; };
	};

public:
	PPCFunctionBoundaryTracker()
		: m_ranges(IMLScratchVector<PPCRange_t>::Take()),
		  m_queuedBranchTargets(IMLScratchVector<uint32, 1>::Take()),
		  m_branchTargetsAll(IMLScratchVector<uint32, 2>::Take())
	{
	}

	~PPCFunctionBoundaryTracker()
	{
		IMLScratchVector<PPCRange_t>::Give(std::move(m_ranges));
		IMLScratchVector<uint32, 1>::Give(std::move(m_queuedBranchTargets));
		IMLScratchVector<uint32, 2>::Give(std::move(m_branchTargetsAll));
	}

	PPCFunctionBoundaryTracker(const PPCFunctionBoundaryTracker&) = delete;
	PPCFunctionBoundaryTracker& operator=(const PPCFunctionBoundaryTracker&) = delete;

	void trackStartPoint(MPTR startAddress)
	{
		processRange(startAddress, -1, -1);
		processBranchTargets();
	}

	bool getRangeForAddress(uint32 address, PPCRange_t& range)
	{
		for (const PPCRange_t& r : m_ranges)
		{
			if (address >= r.startAddress && address < (r.startAddress + r.length))
			{
				range = r;
				return true;
			}
		}
		return false;
	}

	// sorted by start address
	const std::vector<PPCRange_t>& GetRanges() const
	{
		return m_ranges;
	}

	bool ContainsAddress(uint32 addr) const
	{
		for (const PPCRange_t& r : m_ranges)
		{
			if (addr >= r.startAddress && addr < r.getEndAddress())
				return true;
		}
		return false;
	}

	bool IsBranchTarget(uint32 addr) const
	{
		return std::binary_search(m_branchTargetsAll.begin(), m_branchTargetsAll.end(), addr);
	}

private:
	void addBranchDestination(MPTR address)
	{
		// data that gets scanned as code can decode as a branch to unmapped memory
		if (!memory_isAddressRangeAccessible(address, 4))
			return;
		InsertSorted(m_queuedBranchTargets, address);
		InsertSorted(m_branchTargetsAll, address);
	}

	// process flow of instruction
	// returns false if the IP cannot increment past the current instruction
	bool processInstruction(MPTR address)
	{
		// parse instructions
		uint32 opcode = memory_readU32(address);
		switch (Espresso::GetPrimaryOpcode(opcode))
		{
		case Espresso::PrimaryOpcode::ZERO:
		{
			if (opcode == 0)
				return false; // invalid instruction
			break;
		}
		case Espresso::PrimaryOpcode::VIRTUAL_HLE:
		{
			// end of function
			// is there a jump to a instruction after this one?
			uint32 hleFuncId = opcode & 0xFFFF;
			if (hleFuncId >= 0x1000 && hleFuncId < 0x4000)
			{
				if (GamePatch_IsNonReturnFunction(hleFuncId - 0x1000) == false)
				{
					return true;
				}
			}
			return false;
		}
		case Espresso::PrimaryOpcode::BC:
		{
			uint32 BD, BI;
			Espresso::BOField BO;
			bool AA, LK;
			Espresso::decodeOp_BC(opcode, BD, BO, BI, AA, LK);
			uint32 branchTarget = AA ? BD : BD + address;
			if (!LK)
				addBranchDestination(branchTarget);
			break;
		}
		case Espresso::PrimaryOpcode::B:	
		{
			uint32 LI;
			bool AA, LK;
			Espresso::decodeOp_B(opcode, LI, AA, LK);
			uint32 branchTarget = AA ? LI : LI + address;
			if (!LK)
			{
				addBranchDestination(branchTarget);
				// if the next two or previous two instructions are branch instructions, we assume that they are destinations of a jump table
				// todo - can we make this more reliable by checking for BCTR or similar instructions first?
				// example: The Swapper 0x01B1FC04
				if (PPCRecompilerCalcFuncSize_isUnconditionalBranchInstruction(address + 4) && PPCRecompilerCalcFuncSize_isUnconditionalBranchInstruction(address + 8) ||
					PPCRecompilerCalcFuncSize_isUnconditionalBranchInstruction(address - 8) && PPCRecompilerCalcFuncSize_isUnconditionalBranchInstruction(address - 4))
				{
					return true;
				}
				return false; // current flow ends at unconditional branch instruction
			}
			break;
		}
		case Espresso::PrimaryOpcode::GROUP_19:
			switch (Espresso::GetGroup19Opcode(opcode))
			{
			case Espresso::Opcode19::BCLR:
			{
				Espresso::BOField BO;
				uint32 BI;
				bool LK;
				Espresso::decodeOp_BCSPR(opcode, BO, BI, LK);
				if (BO.branchAlways() && !LK)
				{
					// unconditional BLR
					return false;
				}
				break;
			}
			case Espresso::Opcode19::BCCTR:
				if (opcode == 0x4E800420)
				{
					// unconditional BCTR
					// this instruction is often used for switch statements, therefore we should be wary of ending the function here
					// It's better to overestimate function size than to predict sizes that are too short

					// Currently we only end the function if the BCTR is followed by a NOP (alignment) or invalid instruction
					// todo: improve robustness, find better ways to handle false positives
					uint32 nextOpcode = memory_isAddressRangeAccessible(address + 4, 4) ? memory_readU32(address + 4) : 0;

					if (nextOpcode == 0x60000000 || PPCRecompilerCalcFuncSize_isValidInstruction(nextOpcode) == false)
					{
						return false;
					}
					return true;
				}
				// conditional BCTR
				return true;
			default:
				break;
			}
			break;
		default:
			break;
		}
		return true;
	}

	void checkForCollisions()
	{
#ifdef CEMU_DEBUG_ASSERT
		uint32 endOfPrevious = 0;
		for (const PPCRange_t& r : m_ranges)
		{
			if (endOfPrevious > r.startAddress)
			{
				cemu_assert_debug(false);
			}
			endOfPrevious = r.startAddress + r.length;
		}
#endif
	}

	// nextRange is the index of the closest range after startAddress, or -1
	// if there is none; previousRange that of the one it extends, or -1
	void processRange(MPTR startAddress, sint32 previousRange, sint32 nextRange)
	{
		checkForCollisions();
		cemu_assert_debug(previousRange < 0 || (startAddress == m_ranges[previousRange].getEndAddress()));
		sint32 cur;
		if (previousRange >= 0 && m_ranges[previousRange].getEndAddress() == startAddress)
		{
			cur = previousRange;
		}
		else
		{
			cemu_assert_debug(previousRange < 0);
			auto it = std::lower_bound(m_ranges.begin(), m_ranges.end(), startAddress, [](const PPCRange_t& r, uint32 a) { return r.startAddress < a; });
			cur = (sint32)(it - m_ranges.begin());
			m_ranges.insert(it, PPCRange_t(startAddress));
			if (nextRange >= cur)
				nextRange++;
		}
		// process instruction flow until it is interrupted by a non-conditional branch
		MPTR currentAddress = startAddress;
		MPTR endAddress = 0xFFFFFFFF;
		if (nextRange >= 0)
			endAddress = m_ranges[nextRange].startAddress;
		// don't scan past the end of the mapped region
		if (MMURange* mmuRange = memory_getMMURangeByAddress(startAddress); mmuRange && mmuRange->getEnd() < endAddress)
			endAddress = mmuRange->getEnd();
		while (currentAddress < endAddress)
		{
			if (!processInstruction(currentAddress))
			{
				currentAddress += 4;
				break;
			}
			currentAddress += 4;
		}
		m_ranges[cur].length = currentAddress - m_ranges[cur].startAddress;

		if (nextRange >= 0 && currentAddress >= m_ranges[nextRange].startAddress)
		{
			// merge with next range
			m_ranges[cur].length = m_ranges[nextRange].getEndAddress() - m_ranges[cur].startAddress;
			m_ranges.erase(m_ranges.begin() + nextRange);
			checkForCollisions();
			return;
		}
		checkForCollisions();
	}

	// find first unvisited branch target and start a new range there
	// return true if method should be called again
	bool processBranchTargetsSinglePass()
	{
		cemu_assert_debug(!m_ranges.empty());
		size_t rangeIdx = 0;
		sint32 previousRange = -1;
		for (size_t targetIdx = 0; targetIdx < m_queuedBranchTargets.size(); )
		{
			const uint32 target = m_queuedBranchTargets[targetIdx];
			while (rangeIdx < m_ranges.size() && m_ranges[rangeIdx].getEndAddress() <= target)
			{
				previousRange = (sint32)rangeIdx;
				rangeIdx++;
				if (rangeIdx == m_ranges.size())
				{
					// last range reached
					if (m_ranges[previousRange].getEndAddress() == target)
						processRange(target, previousRange, -1);
					else
						processRange(target, -1, -1);
					return true;
				}
			}

			if (target >= m_ranges[rangeIdx].startAddress && target < m_ranges[rangeIdx].getEndAddress())
			{
				// delete visited targets
				m_queuedBranchTargets.erase(m_queuedBranchTargets.begin() + targetIdx);
				continue;
			}

			cemu_assert_debug(m_ranges[rangeIdx].startAddress > target);
			if (previousRange >= 0 && m_ranges[previousRange].getEndAddress() == target)
				processRange(target, previousRange, (sint32)rangeIdx); // extend previousRange
			else
				processRange(target, -1, (sint32)rangeIdx);
			return true;
		}
		return false;
	}

	void processBranchTargets()
	{
		while (processBranchTargetsSinglePass());
	}

	static void InsertSorted(std::vector<uint32>& v, uint32 value)
	{
		auto it = std::lower_bound(v.begin(), v.end(), value);
		if (it == v.end() || *it != value)
			v.insert(it, value);
	}

	private:
	bool PPCRecompilerCalcFuncSize_isUnconditionalBranchInstruction(MPTR address)
	{
		// neighbours can be unmapped
		if (!memory_isAddressRangeAccessible(address, 4))
			return false;
		uint32 opcode = memory_readU32(address);
		if (Espresso::GetPrimaryOpcode(opcode) == Espresso::PrimaryOpcode::B)
		{
			uint32 LI;
			bool AA, LK;
			Espresso::decodeOp_B(opcode, LI, AA, LK);
			if (!LK)
				return true;
		}
		return false;
	}

	bool PPCRecompilerCalcFuncSize_isValidInstruction(uint32 opcode)
	{
		if ((opcode >> 26) == 0)
			return false;
		return true;
	}

private:
	std::vector<PPCRange_t> m_ranges; // sorted by start, not overlapping
	std::vector<uint32> m_queuedBranchTargets; // sorted
	std::vector<uint32> m_branchTargetsAll; // sorted
};