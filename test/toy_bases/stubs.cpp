// What the toy bases call outside themselves, stubbed for the test: the log
// (printed, so a run shows each placement), the portal's speaker and the
// generic USB device the bases' devices derive from.
#include <cstdio>

#include "Cemu/Logging/CemuLogging.h"
#include "audio/IAudioAPI.h"
#include "Cafe/OS/libs/nsyshid/Backend.h"

uint64 s_loggingFlagMask = ~0ull;

bool cemuLog_log(LogType, std::string_view text)
{
	printf("  log: %.*s\n", (int)text.size(), text.data());
	return true;
}

AudioAPIPtr g_portalAudio;

std::unique_ptr<IAudioAPI> IAudioAPI::CreateDeviceFromConfig(AudioType, sint32, sint32, sint32)
{
	return nullptr;
}

namespace nsyshid
{
	Device::Device(uint16 vendorId, uint16 productId, uint8 interfaceIndex, uint8 interfaceSubClass, uint8 protocol)
	{
		(void)vendorId; (void)productId; (void)interfaceIndex; (void)interfaceSubClass; (void)protocol;
	}

	void Device::AssignHID(HID_t*)
	{
	}
}
