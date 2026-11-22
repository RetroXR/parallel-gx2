#pragma once

#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "util/helpers/mailbox.h"

namespace nsyshid
{
	// A figure handed between the frontend's thread and a toy base's command
	// runner, one per spot on the base, with no lock and nothing allocated:
	//
	//   in   the frontend puts a figure on the spot (its bytes) or takes it
	//        off; the runner applies it in order with the title's commands
	//        (it takes what is waiting at every command it runs, and
	//        hid_portal_submit_host makes one run). A put replaces one not
	//        yet applied, so only the latest counts.
	//   out  a title's write to the figure, handed back for the frontend to
	//        save to its dump; again the latest only.
	//
	// The base's state is still only ever touched by its runner: the
	// frontend's thread only fills a mailbox, and a host command run on it
	// runs as the runner (hid_portal.h).
	template<size_t N>
	struct FigureMail
	{
		enum : uint8
		{
			NONE = 0,
			PLACE = 1,
			REMOVE = 2
		};

		FigureMail()
		{
			mailbox_init(&in);
			mailbox_init(&out);
		}

		// the frontend's thread. The runner holds a spot only while it applies
		// it (a copy, a few fields and a line of log), so a put that finds it
		// held waits for it, by the clock: false only if it is still held a
		// second later
		bool Put(uint8 operation, const uint8* bytes)
		{
			if (!mailbox_begin_put(&in))
			{
				const auto giveUp = std::chrono::steady_clock::now() + std::chrono::seconds(1);
				do
				{
					if (std::chrono::steady_clock::now() > giveUp)
						return false;
					std::this_thread::yield();
				} while (!mailbox_begin_put(&in));
			}
			op = operation;
			if (bytes)
				memcpy(data.data(), bytes, N);
			mailbox_end_put(&in);
			return true;
		}

		// the runner: apply(op, bytes) for what is waiting; false when nothing is
		template<typename TApply>
		bool Take(TApply&& apply)
		{
			if (!mailbox_begin_take(&in))
				return false;
			apply(op, data.data());
			mailbox_end_take(&in);
			return true;
		}

		// the runner: a figure the title wrote to; false while the frontend is
		// taking the last one (the runner keeps it dirty and tries again)
		bool Saved(const uint8* bytes)
		{
			if (!mailbox_begin_put(&out))
				return false;
			memcpy(saved.data(), bytes, N);
			mailbox_end_put(&out);
			return true;
		}

		// the frontend's thread: the latest write, to save; false when none
		bool TakeSave(uint8* bytes)
		{
			if (!mailbox_begin_take(&out))
				return false;
			memcpy(bytes, saved.data(), N);
			mailbox_end_take(&out);
			return true;
		}

	  private:
		mailbox_t in;
		uint8 op = NONE;
		std::array<uint8, N> data{};
		mailbox_t out;
		std::array<uint8, N> saved{};
	};
}
