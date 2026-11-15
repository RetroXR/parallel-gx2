#pragma once
#include "Cafe/OS/libs/coreinit/coreinit_Thread.h"
#include "Cafe/OS/libs/coreinit/alarm_table.h"

namespace coreinit
{
	// A host alarm its caller keeps (on its stack: OSSleepTicks,
	// OSWaitEventWithTimeout), zeroed before its first start; with the
	// scheduler lock held. Nothing is allocated.
	void OSHostAlarmStart(os_host_alarm_t* alarm, uint64 nextFire, uint64 period, os_alarm_cb_t callbackFunc, void* context);
	void OSHostAlarmStop(os_host_alarm_t* alarm);

	struct OSAlarm_t;

	struct OSAlarmQueue
	{
		/* +0x00 */ uint32be magic;
		/* +0x04 */ MEMPTR<const char> name;
		/* +0x08 */ uint32be ukn08;
		/* +0x0C */ OSThreadQueue threadQueue;
		/* +0x1C */ MEMPTR<OSAlarm_t> head;
		/* +0x20 */ MEMPTR<OSAlarm_t> tail;
	};

	static_assert(sizeof(OSAlarmQueue) == 0x24);

	struct OSAlarm_t
	{
		/* +0x00 */ betype<uint32>  magic;
		/* +0x04 */ MEMPTR<const char> name;
		/* +0x08 */ uint32	        ukn08;
		/* +0x0C */ MPTR			handler;
		/* +0x10 */ uint32be        group;
		/* +0x14 */ uint32			padding14;
		/* +0x18 */ uint64	        nextTime; // next fire time
		/* +0x20 */ MEMPTR<OSAlarm_t> prev;
		/* +0x24 */ MEMPTR<OSAlarm_t> next;
		/* +0x28 */ uint64          period; // period (zero for non-periodic timer)
		/* +0x30 */ uint64          startTime; // period start
		/* +0x38 */ uint32be        userData;
		/* +0x3C */ uint32		    ukn3C;
		/* +0x40 */ OSThreadQueue   uknThreadQueue;
		/* +0x50 */ MEMPTR<OSAlarmQueue> alarmQueue;
		/* +0x54 */ MEMPTR<void>    context;

		void setMagic()
		{
			magic = (uint32)'aLrM';
		}

		bool checkMagic()
		{
			return magic == (uint32)'aLrM';
		}
	};

	static_assert(sizeof(OSAlarm_t) == 0x58);

	void OSCreateAlarm(OSAlarm_t* alarm);
	bool OSCancelAlarm(OSAlarm_t* alarm);
	bool OSSetAlarm(OSAlarm_t* alarm, uint64 time, MPTR handlerFunc);
	void OSSetAlarmUserData(OSAlarm_t* alarm, uint32 userData);
	bool OSSetPeriodicAlarm(OSAlarm_t* OSAlarm, uint64 startTick, uint64 periodTick, MPTR OSAlarmHandler);

	void OSAlarm_Shutdown();

	void alarm_update();
	// When the soonest alarm is due, in OSGetTime's ticks; ~0 for none
	uint64 alarm_soonest();

	void MapAlarmExports();
	void InitializeAlarm();
}