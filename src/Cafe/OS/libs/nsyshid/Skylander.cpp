#include "Skylander.h"

#include "nsyshid.h"
#include "Backend.h"

#include "audio/IAudioAPI.h"
#include "config/CemuConfig.h"

namespace nsyshid
{
	SkylanderUSB g_skyportal;

	SkylanderPortalDevice::SkylanderPortalDevice()
		: Device(0x1430, 0x0150, 1, 2, 0)
	{
		m_IsOpened = false;
	}

	bool SkylanderPortalDevice::Open()
	{
		if (!IsOpened())
		{
			m_IsOpened = true;
		}
		return true;
	}

	void SkylanderPortalDevice::Close()
	{
		if (IsOpened())
		{
			m_IsOpened = false;
		}
	}

	bool SkylanderPortalDevice::IsOpened()
	{
		return m_IsOpened;
	}

	Device::ReadResult SkylanderPortalDevice::Read(ReadMessage* message)
	{
		const std::array<uint8, 64> status = g_skyportal.GetStatus();
		memcpy(message->data, status.data(), std::min<uint32>(message->length, 64));
		message->bytesRead = message->length;
		// the portal answers an interrupt read every 10 ms
		message->completeAfterUs = 10000;
		return Device::ReadResult::Success;
	}

	Device::WriteResult SkylanderPortalDevice::Write(WriteMessage* message)
	{
		if (message->length != 64) {
			cemu_assert_error();
		}

		bool haveAudio;
		{
			AudioDevicesUse devices;
			haveAudio = devices && g_portalAudio;
		}
		if (!haveAudio)
		{
			// Portal audio is mono channel, 16 bit audio.
			// Audio is unsigned 16 bit, supplied as 64 bytes which is 32 samples per block
			AudioDevicesChange change;
			if (!g_portalAudio)
				g_portalAudio = IAudioAPI::CreateDeviceFromConfig(IAudioAPI::AudioType::Portal, 8000, 32, 16);
		}
		std::array<sint16, 32> mono_samples;
		for (unsigned int i = 0; i < mono_samples.size(); ++i)
		{
			sint16 sample = static_cast<uint16>(message->data[i * 2 + 1]) << 8 | static_cast<uint16>(message->data[i * 2]);
			mono_samples[i] = sample;
		}
		{
			AudioDevicesUse devices;
			if (devices && g_portalAudio)
				g_portalAudio->FeedBlock(mono_samples.data());
		}
		message->bytesWritten = message->length;
		return Device::WriteResult::Success;
	}

	bool SkylanderPortalDevice::GetDescriptor(uint8 descType,
											  uint8 descIndex,
											  uint16 lang,
											  uint8* output,
											  uint32 outputMaxLength)
	{
		uint8 configurationDescriptor[0x29];

		uint8* currentWritePtr;

		// configuration descriptor
		currentWritePtr = configurationDescriptor + 0;
		*(uint8*)(currentWritePtr + 0) = 9;			// bLength
		*(uint8*)(currentWritePtr + 1) = 2;			// bDescriptorType
		*(uint16be*)(currentWritePtr + 2) = 0x0029; // wTotalLength
		*(uint8*)(currentWritePtr + 4) = 1;			// bNumInterfaces
		*(uint8*)(currentWritePtr + 5) = 1;			// bConfigurationValue
		*(uint8*)(currentWritePtr + 6) = 0;			// iConfiguration
		*(uint8*)(currentWritePtr + 7) = 0x80;		// bmAttributes
		*(uint8*)(currentWritePtr + 8) = 0xFA;		// MaxPower
		currentWritePtr = currentWritePtr + 9;
		// interface descriptor
		*(uint8*)(currentWritePtr + 0) = 9;	   // bLength
		*(uint8*)(currentWritePtr + 1) = 0x04; // bDescriptorType
		*(uint8*)(currentWritePtr + 2) = 0;	   // bInterfaceNumber
		*(uint8*)(currentWritePtr + 3) = 0;	   // bAlternateSetting
		*(uint8*)(currentWritePtr + 4) = 2;	   // bNumEndpoints
		*(uint8*)(currentWritePtr + 5) = 3;	   // bInterfaceClass
		*(uint8*)(currentWritePtr + 6) = 0;	   // bInterfaceSubClass
		*(uint8*)(currentWritePtr + 7) = 0;	   // bInterfaceProtocol
		*(uint8*)(currentWritePtr + 8) = 0;	   // iInterface
		currentWritePtr = currentWritePtr + 9;
		// HID descriptor
		*(uint8*)(currentWritePtr + 0) = 9;			// bLength
		*(uint8*)(currentWritePtr + 1) = 0x21;		// bDescriptorType
		*(uint16be*)(currentWritePtr + 2) = 0x0111; // bcdHID
		*(uint8*)(currentWritePtr + 4) = 0x00;		// bCountryCode
		*(uint8*)(currentWritePtr + 5) = 0x01;		// bNumDescriptors
		*(uint8*)(currentWritePtr + 6) = 0x22;		// bDescriptorType
		*(uint16be*)(currentWritePtr + 7) = 0x001D; // wDescriptorLength
		currentWritePtr = currentWritePtr + 9;
		// endpoint descriptor 1
		*(uint8*)(currentWritePtr + 0) = 7;			// bLength
		*(uint8*)(currentWritePtr + 1) = 0x05;		// bDescriptorType
		*(uint8*)(currentWritePtr + 2) = 0x81;		// bEndpointAddress
		*(uint8*)(currentWritePtr + 3) = 0x03;		// bmAttributes
		*(uint16be*)(currentWritePtr + 4) = 0x0040; // wMaxPacketSize
		*(uint8*)(currentWritePtr + 6) = 0x01;		// bInterval
		currentWritePtr = currentWritePtr + 7;
		// endpoint descriptor 2
		*(uint8*)(currentWritePtr + 0) = 7;			// bLength
		*(uint8*)(currentWritePtr + 1) = 0x05;		// bDescriptorType
		*(uint8*)(currentWritePtr + 2) = 0x02;		// bEndpointAddress
		*(uint8*)(currentWritePtr + 3) = 0x03;		// bmAttributes
		*(uint16be*)(currentWritePtr + 4) = 0x0040; // wMaxPacketSize
		*(uint8*)(currentWritePtr + 6) = 0x01;		// bInterval
		currentWritePtr = currentWritePtr + 7;

		cemu_assert_debug((currentWritePtr - configurationDescriptor) == 0x29);

		memcpy(output, configurationDescriptor,
			   std::min<uint32>(outputMaxLength, sizeof(configurationDescriptor)));
		return true;
	}

	bool SkylanderPortalDevice::SetIdle(uint8 ifIndex,
										uint8 reportId,
										uint8 duration)
	{
		return true;
	}

	bool SkylanderPortalDevice::SetProtocol(uint8 ifIndex, uint8 protocol)
	{
		return true;
	}

	bool SkylanderPortalDevice::SetReport(ReportMessage* message)
	{
		g_skyportal.Submit(message->data, message->length);
		message->completeAfterUs = 1000;
		return true;
	}

	void SkylanderUSB::Start()
	{
		// a title loaded again without its unload in between
		Stop();
		for (auto& s : m_skylanders)
			s = {};
		m_spotOfSlot.fill(0xFF);
		m_slotOfSpot.fill(0xFF);
		m_lastSerial.fill(0);
		m_dirty.fill(false);
		m_activated = true;
		m_colorRight = {};
		m_colorLeft = {};
		m_colorTrap = {};
		retro_atomic_int_init(&m_statusWord, 0);
		retro_atomic_int_init(&m_active, 1);
		retro_atomic_int_init(&m_interruptCounter, 0);
		m_started = hid_portal_init(&m_portal, &SkylanderUSB::RunCommand, this);
		// figures the frontend put on before the title started; no transfer
		// runs yet, so this thread is the runner
		ApplyMail();
	}

	void SkylanderUSB::Close()
	{
		if (m_started)
			hid_portal_close(&m_portal);
	}

	void SkylanderUSB::Stop()
	{
		if (m_started)
			hid_portal_free(&m_portal);
		m_started = false;
	}

	void SkylanderUSB::Submit(const uint8* buf, uint32 length)
	{
		if (m_started && !hid_portal_submit(&m_portal, buf, length))
			cemuLog_logDebug(LogType::Force, "Skylander portal: command dropped, {} queued", HID_PORTAL_SLOTS);
	}

	void SkylanderUSB::RunCommand(void* ctx, const unsigned char* cmd, unsigned len)
	{
		SkylanderUSB* self = static_cast<SkylanderUSB*>(ctx);
		// every command takes what the frontend put on or took off first
		self->ApplyMail();
		if (len == HID_PORTAL_HOST)
		{
			self->PublishStatus();
			return;
		}
		uint8 buf[HID_PORTAL_REPORT] = {};
		memcpy(buf, cmd, len);
		self->ControlTransfer(buf, len);
		self->FlushSaves();
	}

	bool SkylanderUSB::PlaceFigure(uint8 spot, const uint8* figure)
	{
		if (spot >= MAX_SKYLANDERS || !m_mail[spot].Put(FigureMail<SKY_FIGURE_SIZE>::PLACE, figure))
			return false;
		Kick();
		return true;
	}

	bool SkylanderUSB::RemoveFigure(uint8 spot)
	{
		if (spot >= MAX_SKYLANDERS || !m_mail[spot].Put(FigureMail<SKY_FIGURE_SIZE>::REMOVE, nullptr))
			return false;
		Kick();
		return true;
	}

	bool SkylanderUSB::TakeSave(uint8 spot, uint8* figure)
	{
		return spot < MAX_SKYLANDERS && m_mail[spot].TakeSave(figure);
	}

	void SkylanderUSB::DropPending()
	{
		for (auto& mail : m_mail)
			mail.Put(FigureMail<SKY_FIGURE_SIZE>::NONE, nullptr);
	}

	// a command of our own, so the change is applied now, not at the title's next
	void SkylanderUSB::Kick()
	{
		if (m_started)
			hid_portal_submit_host(&m_portal, 0);
	}

	void SkylanderUSB::ApplyMail()
	{
		for (uint8 spot = 0; spot < MAX_SKYLANDERS; spot++)
		{
			m_mail[spot].Take([&](uint8 op, const uint8* figure) {
				if (op == FigureMail<SKY_FIGURE_SIZE>::PLACE)
					Place(spot, figure);
				else if (op == FigureMail<SKY_FIGURE_SIZE>::REMOVE)
					Remove(spot);
			});
		}
	}

	// LoadSkylander upstream: the slot this figure had if it is free, else the
	// first free one; reported ADDED, then READY
	void SkylanderUSB::Place(uint8 spot, const uint8* figure)
	{
		Remove(spot);
		uint32 serial = 0;
		for (int i = 3; i >= 0; i--)
			serial = (serial << 8) | figure[i];
		uint8 slot = 0xFF;
		for (uint8 i = 0; i < MAX_SKYLANDERS; i++)
		{
			if (m_skylanders[i].status & 1)
				continue;
			if (m_lastSerial[i] == serial)
			{
				slot = i;
				break;
			}
			if (slot == 0xFF)
				slot = i;
		}
		if (slot == 0xFF)
		{
			cemuLog_log(LogType::Force, "Skylander portal: full, figure {:08x} not placed", serial);
			return;
		}
		auto& sky = m_skylanders[slot];
		memcpy(sky.data.data(), figure, sky.data.size());
		sky.status = Skylander::ADDED;
		m_lastSerial[slot] = serial;
		m_dirty[slot] = false;
		m_spotOfSlot[slot] = spot;
		m_slotOfSpot[spot] = slot;
		retro_atomic_fetch_or_int(&m_statusWord, (int)((uint32)Skylander::ADDED << (slot * 2)));
		cemuLog_log(LogType::Force, "Skylander portal: figure {:08x} placed in slot {}", serial, slot);
	}

	// RemoveSkylander upstream: reported REMOVING, then REMOVED
	void SkylanderUSB::Remove(uint8 spot)
	{
		const uint8 slot = m_slotOfSpot[spot];
		if (slot >= MAX_SKYLANDERS)
			return;
		FlushSaves();
		m_skylanders[slot].status = Skylander::REMOVED;
		m_dirty[slot] = false;
		m_spotOfSlot[slot] = 0xFF;
		m_slotOfSpot[spot] = 0xFF;
		// ADDED (3) or READY (1) to REMOVING (2): clear the low bit, and set the
		// high one where a status report had already moved ADDED on to READY
		const int shift = slot * 2;
		const uint32 old = (uint32)retro_atomic_fetch_and_int(&m_statusWord, (int)~(1u << shift));
		if (((old >> shift) & 3) == Skylander::READY)
			retro_atomic_fetch_or_int(&m_statusWord, (int)((uint32)Skylander::REMOVING << shift));
		cemuLog_log(LogType::Force, "Skylander portal: slot {} emptied", slot);
	}

	// figures the title wrote to, back to the frontend; one still being taken
	// stays dirty for the next command
	void SkylanderUSB::FlushSaves()
	{
		for (uint8 slot = 0; slot < MAX_SKYLANDERS; slot++)
		{
			if (!m_dirty[slot] || m_spotOfSlot[slot] >= MAX_SKYLANDERS)
				continue;
			if (m_mail[m_spotOfSlot[slot]].Saved(m_skylanders[slot].data.data()))
				m_dirty[slot] = false;
		}
	}

	void SkylanderUSB::ControlTransfer(uint8* buf, uint32 length)
	{
		std::array<uint8, 64> interruptResponse = {};
		switch (buf[0])
		{
		case 'A':
		{
			interruptResponse = {buf[0], buf[1], 0xFF, 0x77};
			Activate();
			break;
		}
		case 'C':
		{
			SetLeds(0x01, buf[1], buf[2], buf[3]);
			break;
		}
		case 'J':
		{
			SetLeds(buf[1], buf[2], buf[3], buf[4]);
			interruptResponse = {buf[0]};
			break;
		}
		case 'L':
		{
			uint8 side = buf[1];
			if (side == 0x02)
			{
				side = 0x04;
			}
			SetLeds(side, buf[2], buf[3], buf[4]);
			break;
		}
		case 'M':
		{
			interruptResponse = {buf[0], buf[1], 0x00, 0x19};
			break;
		}
		case 'Q':
		{
			const uint8 skyNum = buf[1] & 0xF;
			const uint8 block = buf[2];
			QueryBlock(skyNum, block, interruptResponse.data());
			break;
		}
		case 'R':
		{
			interruptResponse = {buf[0], 0x02, 0x1b};
			break;
		}
		case 'S':
		case 'V':
		{
			// No response needed
			break;
		}
		case 'W':
		{
			const uint8 skyNum = buf[1] & 0xF;
			const uint8 block = buf[2];
			WriteBlock(skyNum, block, &buf[3], interruptResponse.data());
			break;
		}
		default:
			cemuLog_logDebug(LogType::Force, "Skylander portal: unknown command {:02x}", buf[0]);
			break;
		}
		if (interruptResponse[0] != 0)
			hid_portal_report(&m_portal, interruptResponse.data());
		PublishStatus();
	}

	// what an interrupt read with no answer queued reports besides the
	// figures' status, which Place, Remove and GetStatus keep in m_statusWord
	void SkylanderUSB::PublishStatus()
	{
		retro_atomic_store_release_int(&m_active, m_activated ? 1 : 0);
	}

	void SkylanderUSB::Activate()
	{
		m_activated = true;
	}

	void SkylanderUSB::SetLeds(uint8 side, uint8 r, uint8 g, uint8 b)
	{
		if (side == 0x00)
		{
			m_colorRight.red = r;
			m_colorRight.green = g;
			m_colorRight.blue = b;
		}
		else if (side == 0x01)
		{
			m_colorRight.red = r;
			m_colorRight.green = g;
			m_colorRight.blue = b;

			m_colorLeft.red = r;
			m_colorLeft.green = g;
			m_colorLeft.blue = b;
		}
		else if (side == 0x02)
		{
			m_colorLeft.red = r;
			m_colorLeft.green = g;
			m_colorLeft.blue = b;
		}
		else if (side == 0x03)
		{
			m_colorTrap.red = r;
			m_colorTrap.green = g;
			m_colorTrap.blue = b;
		}
	}

	void SkylanderUSB::QueryBlock(uint8 skyNum, uint8 block, uint8* replyBuf)
	{
		const auto& skylander = m_skylanders[skyNum];

		replyBuf[0] = 'Q';
		replyBuf[2] = block;
		if ((skylander.status & 1) && block < SKY_BLOCK_COUNT)
		{
			replyBuf[1] = (0x10 | skyNum);
			memcpy(replyBuf + 3, skylander.data.data() + (16 * block), 16);
		}
		else
		{
			replyBuf[1] = skyNum;
		}
	}

	void SkylanderUSB::WriteBlock(uint8 skyNum, uint8 block,
								  const uint8* toWriteBuf, uint8* replyBuf)
	{
		auto& skylander = m_skylanders[skyNum];

		replyBuf[0] = 'W';
		replyBuf[2] = block;

		if ((skylander.status & 1) && block < SKY_BLOCK_COUNT)
		{
			replyBuf[1] = (0x10 | skyNum);
			memcpy(skylander.data.data() + (block * 16), toWriteBuf, 16);
			m_dirty[skyNum] = true;
		}
		else
		{
			replyBuf[1] = skyNum;
		}
	}

	std::array<uint8, 64> SkylanderUSB::GetStatus()
	{
		std::array<uint8, 64> interruptResponse = {};
		if (m_started && hid_portal_try_read(&m_portal, interruptResponse.data()))
			return interruptResponse;

		// this report's status; ADDED moves on to READY and REMOVING to
		// REMOVED for the next (every odd bit cleared, in one step)
		const uint32 status = (uint32)retro_atomic_fetch_and_int(&m_statusWord, 0x55555555);
		const uint8 active = retro_atomic_load_acquire_int(&m_active) ? 0x01 : 0x00;
		const uint8 counter = (uint8)retro_atomic_fetch_add_int(&m_interruptCounter, 1);
		interruptResponse[0] = 0x53;
		memcpy(&interruptResponse[1], &status, sizeof(status));
		interruptResponse[5] = counter;
		interruptResponse[6] = active;
		return interruptResponse;
	}

} // namespace nsyshid
