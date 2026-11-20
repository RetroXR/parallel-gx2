// What MMU.cpp takes from the rest of the core: none of it reached by the
// MMIO handlers
#include "Cafe/HW/MMU/MMU.h"
#include "Cafe/GraphicPack/GraphicPack2.h"
#include "util/MemMapper/MemMapper.h"
#include "WindowSystem.h"
#include "Common/FileStream.h"

uint64 s_loggingFlagMask = 0;
bool cemuLog_log(LogType, std::string_view) { return false; }
std::vector<std::pair<MPTR, MPTR>> GraphicPack2::GetActiveRAMMappings() { return {}; }
namespace MemMapper
{
	size_t GetPageSize() { return 4096; }
	void* ReserveMemory(void*, size_t, PAGE_PERMISSION) { return nullptr; }
	void* AllocateMemory(void*, size_t, PAGE_PERMISSION, bool) { return nullptr; }
	void FreeMemory(void*, size_t, bool) {}
}
void WindowSystem::ShowErrorDialog(std::string_view, std::string_view, std::optional<WindowSystem::ErrorCategory>) {}
FileStream* FileStream::createFile2(const fs::path&) { return nullptr; }
bool FileStream::CreateDirectories(const fs::path&) { return false; }
sint32 FileStream::writeData(const void*, sint32) { return 0; }
FileStream::~FileStream() {}
