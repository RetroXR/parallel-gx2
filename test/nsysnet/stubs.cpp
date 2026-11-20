// What nsysnet.cpp takes from the rest of the core: none of it reached by
// making and closing sockets
#include "Cafe/OS/common/OSCommon.h"
#include "Cafe/HW/Espresso/PPCState.h"
#include "Cafe/HW/MMU/MMU.h"
#include "Common/SysAllocator.h"
#include "Cafe/OS/libs/coreinit/coreinit_Thread.h"
#include "Cafe/OS/libs/coreinit/coreinit_GHS.h"
#include "Cafe/IOSU/legacy/iosu_crypto.h"

uint64 s_loggingFlagMask = 0;
uint8* memory_base = nullptr;
uint32 g_lastReturn = 0;

bool cemuLog_log(LogType type, std::string_view text) { return false; }
bool cemuLog_advancedPPCLoggingEnabled() { return false; }
void osLib_returnFromFunction(PPCInterpreter_t* hCPU, uint32 returnValue) { g_lastReturn = returnValue; }
void osLib_addFunctionInternal(const char*, const char*, void (*)(PPCInterpreter_t*)) {}
void PPCCore_switchToScheduler() {}
SysAllocatorBase::SysAllocatorBase() {}
MPTR OSAllocFromSystem(uint32, uint32) { return 0; }
MPTR coreinit_allocFromSysArea(uint32, uint32) { return 0; }
uint8* memory_getPointerFromVirtualOffset(uint32) { return nullptr; }
uint8* memory_getPointerFromVirtualOffsetAllowNull(uint32) { return nullptr; }
uint32 memory_getVirtualOffsetFromPointer(void*) { return 0; }
uint32 memory_readU32(uint32) { return 0; }
void memory_writeU32(uint32, uint32) {}
uint8* iosuCrypto_getCertificateDataById(int, int*) { return nullptr; }
uint8* iosuCrypto_getCertificatePrivateKeyById(int, int*) { return nullptr; }
namespace coreinit
{
	OSThread_t* OSGetCurrentThread() { return nullptr; }
	void OSSleepTicks(uint64) {}
	sint32 __gh_get_errno() { return 0; }
	void __gh_set_errno(sint32) {}
}
#if !BOOST_OS_WINDOWS
uint32_t GetTickCount() { return 0; }
#endif
