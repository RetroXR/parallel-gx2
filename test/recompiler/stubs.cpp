// What the recompiler and the interpreter call in the rest of the core, for
// recompiler_test: guest memory is the test's own 4 GB reservation, and the
// OS, logging and settings answer as a title with the recompiler on would.
#include "Cafe/HW/Espresso/PPCState.h"
#include "Cafe/HW/MMU/MMU.h"
#include "Cafe/OS/libs/coreinit/coreinit_CodeGen.h"
#include "Cafe/OS/libs/coreinit/coreinit_Time.h"
#include "config/ActiveSettings.h"
#include "Common/FileStream.h"

#include <cstdlib>

uint8* memory_base;
uint64 s_loggingFlagMask = 0;

MMURange::MMURange(const uint32 baseAddress, const uint32 size, MMU_MEM_AREA_ID areaId, const std::string_view name, MFLAG flags)
	: baseAddress(baseAddress), initSize(size), name(name), flags(flags), areaId(areaId), size(size)
{
}

MMURange mmuRange_TRAMPOLINE_AREA{0x00E00000, 0x00200000, MMU_MEM_AREA_ID::CODE_TRAMPOLINE, "TRAMPOLINE_AREA"};
MMURange mmuRange_CODECAVE{0x01800000, 0x00400000, MMU_MEM_AREA_ID::CODE_CAVE, "CODECAVE"};
static MMURange s_testText{0x02000000, 0x0C000000, MMU_MEM_AREA_ID::CODE_MAIN, "TEXT_AREA"};

MMURange* memory_getMMURangeByAddress(MPTR address)
{
	return s_testText.containsAddress(address) ? &s_testText : nullptr;
}

uint8* memory_getPointerFromVirtualOffset(uint32 offset)
{
	return memory_base + offset;
}

bool memory_isAddressRangeAccessible(MPTR addr, uint32 size)
{
	return ((uint64)addr + size <= 0x10000ull) || (addr >= 0x02000000 && (uint64)addr + size <= 0x10100000ull);
}

uint32 memory_readU32(uint32 address)
{
	return _swapEndianU32(*(uint32*)(memory_base + address));
}

bool cemuLog_log(LogType, std::string_view)
{
	return true;
}

CPUMode ActiveSettings::GetCPUMode()
{
	return CPUMode::MulticoreRecompiler;
}

bool ActiveSettings::DumpRecompilerFunctionsEnabled()
{
	return false;
}

FileStream* FileStream::createFile2(const fs::path&)
{
	return nullptr;
}

sint32 FileStream::writeData(const void*, sint32)
{
	return 0;
}

FileStream::~FileStream()
{
}

bool GamePatch_IsNonReturnFunction(uint32)
{
	return false;
}

void LatteBufferCache_notifyDCFlush(MPTR, uint32)
{
}

uint64 PPCTimer_getFromRDTSC()
{
	return 0;
}

void SetThreadName(const char*)
{
}

uint32 RPLLoader_MakePPCCallable(void (*)(PPCInterpreter_t*))
{
	return 0;
}

namespace coreinit
{
	void OSGetCodegenVirtAddrRangeInternal(uint32& rangeStart, uint32& rangeSize)
	{
		rangeStart = 0;
		rangeSize = 0;
	}

	void codeGenHandleICBI(uint32)
	{
	}

	bool codeGenShouldAvoid()
	{
		return false;
	}

	uint64 OSGetSystemTime()
	{
		return 0;
	}
}
