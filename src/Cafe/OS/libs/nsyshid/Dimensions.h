#include <array>
#include <optional>
#include <span>

#include "nsyshid.h"
#include "Backend.h"

#include "hid_portal.h"
#include "FigureMail.h"

namespace nsyshid
{
	class DimensionsToypadDevice final : public Device
	{
	  public:
		DimensionsToypadDevice();
		~DimensionsToypadDevice() = default;

		bool Open() override;

		void Close() override;

		bool IsOpened() override;

		ReadResult Read(ReadMessage* message) override;

		WriteResult Write(WriteMessage* message) override;

		bool GetDescriptor(uint8 descType,
						   uint8 descIndex,
						   uint16 lang,
						   uint8* output,
						   uint32 outputMaxLength) override;

		bool SetIdle(uint8 ifIndex,
					 uint8 reportId,
					 uint8 duration) override;

		bool SetProtocol(uint8 ifIndex, uint8 protocol) override;

		bool SetReport(ReportMessage* message) override;

	  private:
		bool m_IsOpened;
	};

	constexpr uint16 DIM_FIGURE_SIZE = 0x2D * 0x04;
	constexpr uint8 DIM_SPOTS = 7;

	// The emulated toypad. Commands run one at a time on whichever transfer
	// worker hid_portal hands them to, so the state below is the runner's
	// alone; each answer is queued for an interrupt read, which waits for one.
	// A frontend puts figures on its seven spots through FigureMail (1 the
	// centre pad, 0, 3 and 4 the left, 2, 5 and 6 the right), and takes a
	// title's writes to a figure back the same way.
	class DimensionsUSB
	{
	  public:
		struct DimensionsMini final
		{
			std::array<uint8, 0x2D * 0x04> data{};
			uint8 index = 255;
			uint8 pad = 255;
			uint32 id = 0;
		};

		// for each title, before its transfers start
		void Start();
		// wakes the reads still waiting, before the transfer workers stop
		void Close();
		// once they have
		void Stop();

		void Submit(const uint8* buf, uint32 length);
		// false once closed
		bool GetStatus(std::array<uint8, 32>& response);

		// The frontend's thread. `spot` is a position on the base (see the
		// class comment); a figure is its dump. False only while that spot is
		// being applied.
		bool PlaceFigure(uint8 spot, const uint8* figure);
		bool RemoveFigure(uint8 spot);
		// a figure on the spot the title wrote to since the last take, to save
		bool TakeSave(uint8 spot, uint8* figure);
		// what is waiting and not applied yet, dropped (at unload)
		void DropPending();
		// the pad a spot is on: 1 the centre, 2 the left, 3 the right
		static uint8 PadOf(uint8 spot);

	  private:
		static void RunCommand(void* ctx, const unsigned char* cmd, unsigned len);
		// the runner: what the frontend put on or took off, and saves handed back
		void ApplyMail();
		void Place(uint8 spot, const uint8* figure);
		void Remove(uint8 spot);
		void FlushSaves();
		void Kick();
		void FigureChanged(const DimensionsMini& figure, bool removed);
		uint32 GetFigureId(const std::array<uint8, DIM_FIGURE_SIZE>& buf);
		std::array<uint8, 16> GenerateFigureKey(const std::array<uint8, DIM_FIGURE_SIZE>& buf);
		void SendCommand(std::span<const uint8, 32> buf);
		void GenerateRandomNumber(std::span<const uint8, 8> buf, uint8 sequence,
								  std::array<uint8, 32>& replyBuf);
		void InitializeRNG(uint32 seed);
		void GetChallengeResponse(std::span<const uint8, 8> buf, uint8 sequence,
								  std::array<uint8, 32>& replyBuf);
		void QueryBlock(uint8 index, uint8 page, std::array<uint8, 32>& replyBuf,
						uint8 sequence);
		void WriteBlock(uint8 index, uint8 page, std::span<const uint8, 4> toWriteBuf, std::array<uint8, 32>& replyBuf,
						uint8 sequence);
		void GetModel(std::span<const uint8, 8> buf, uint8 sequence,
					  std::array<uint8, 32>& replyBuf);
		uint8 GenerateChecksum(const std::array<uint8, 32>& data,
							   int numOfBytes) const;
		std::array<uint8, 8> Decrypt(std::span<const uint8, 8> buf, std::optional<std::array<uint8, 16>> key);
		std::array<uint8, 8> Encrypt(std::span<const uint8, 8> buf, std::optional<std::array<uint8, 16>> key);
		uint32 GetNext();
		DimensionsMini& GetFigureByIndex(uint8 index);

		hid_portal_t m_portal;
		bool m_started = false;
		std::array<DimensionsMini, 7> m_figures{};
		std::array<bool, DIM_SPOTS> m_dirty{};
		// the frontend's mailboxes, by spot; outlive a title
		std::array<FigureMail<DIM_FIGURE_SIZE>, DIM_SPOTS> m_mail;
		uint32 m_randomA = 0;
		uint32 m_randomB = 0;
		uint32 m_randomC = 0;
		uint32 m_randomD = 0;
	};
	extern DimensionsUSB g_dimensionstoypad;

} // namespace nsyshid