#include "Cafe/OS/common/OSCommon.h"
#include "Cafe/HW/Espresso/PPCCallback.h"
#include "Cafe/OS/libs/coreinit/coreinit_Thread.h"
#include "Cafe/OS/libs/coreinit/coreinit_Time.h"
#include "Cafe/OS/libs/coreinit/coreinit_Alarm.h"
#include "Cafe/HW/Espresso/Recompiler/PPCRecompiler.h"
#include "Cafe/OS/RPL/rpl.h"
#include "Cafe/OS/libs/coreinit/alarm_table.h"

// #define ALARM_LOGGING

namespace coreinit
{
	SysAllocator<OSEvent> g_alarmEvent;

	SysAllocator<OSThread_t> g_alarmThread;
	SysAllocator<uint8, 1024 * 128> _g_alarmThreadStack;
	SysAllocator<char, 32> _g_alarmThreadName;


	// The host alarms, sorted by when they fire, and the title's alarms paired
	// with theirs (alarm_table.h): fixed, nothing allocated as alarms are set,
	// fired and cancelled. All with the scheduler lock held.
	static os_alarm_list_t s_hostAlarms;
	static os_guest_alarms_t s_guestAlarms;
	static std::atomic_uint64_t s_soonestAlarm{ std::numeric_limits<uint64>::max() };

	static void UpdateSoonestAlarm()
	{
		s_soonestAlarm.store(os_alarm_list_soonest(&s_hostAlarms), std::memory_order_relaxed);
	}

	void OSHostAlarmStart(os_host_alarm_t* alarm, uint64 nextFire, uint64 period, os_alarm_cb_t callbackFunc, void* context)
	{
		cemu_assert_debug(__OSHasSchedulerLock());
		os_alarm_list_start(&s_hostAlarms, alarm, nextFire, period, callbackFunc, context);
		UpdateSoonestAlarm();
	}

	void OSHostAlarmStop(os_host_alarm_t* alarm)
	{
		cemu_assert_debug(__OSHasSchedulerLock());
		os_alarm_list_stop(&s_hostAlarms, alarm);
		UpdateSoonestAlarm();
	}

	uint64 alarm_soonest()
	{
		return s_soonestAlarm.load(std::memory_order_relaxed);
	}

	void alarm_update()
	{	
		cemu_assert_debug(!__OSHasSchedulerLock());
		uint64 currentTick = coreinit::OSGetTime();
		// without the lock: nothing due yet
		if (currentTick < s_soonestAlarm.load(std::memory_order_relaxed))
			return;
		__OSLockScheduler();
		os_alarm_list_fire(&s_hostAlarms, currentTick);
		UpdateSoonestAlarm();
		__OSUnlockScheduler();
	}

	/* alarm API */

	void OSInitAlarmQueue(OSAlarmQueue* queue)
	{
		queue->magic = (uint32)'aLmQ';
		queue->name = nullptr;
		queue->ukn08 = 0;
		OSInitThreadQueueEx(&queue->threadQueue, queue);
		queue->head = nullptr;
		queue->tail = nullptr;
	}

	void OSInitAlarmQueueEx(OSAlarmQueue* queue, const char* name)
	{
		OSInitAlarmQueue(queue);
		queue->name = name;
	}

	void OSCreateAlarm(OSAlarm_t* alarm)
	{
		memset(alarm, 0, sizeof(OSAlarm_t));
		alarm->setMagic();
	}

	void OSCreateAlarmEx(OSAlarm_t* alarm, const char* alarmName)
	{
		memset(alarm, 0, sizeof(OSAlarm_t));
		alarm->setMagic();
		alarm->name = alarmName;
	}

	// the title's alarms that are set: their slot, or -1
	static int __OSFindActiveAlarm(OSAlarm_t* alarm)
	{
		return os_guest_alarms_find(&s_guestAlarms, MEMPTR<OSAlarm_t>(alarm).GetMPTR());
	}

	static void __OSEndActiveAlarm(int slot)
	{
		OSHostAlarmStop(&s_guestAlarms.host[slot]);
		os_guest_alarms_remove(&s_guestAlarms, slot);
	}

	// fired alarms wait in their alarm queue until they are taken out of it
	static bool __OSIsAlarmInQueue(OSAlarm_t* alarm)
	{
		OSAlarmQueue* queue = alarm->alarmQueue.GetPtr();
		return queue && (queue->head.GetPtr() == alarm || alarm->prev);
	}

	static void __OSAppendAlarmToQueue(OSAlarm_t* alarm)
	{
		OSAlarmQueue* queue = alarm->alarmQueue.GetPtr();
		alarm->prev = queue->tail;
		alarm->next = nullptr;
		if (queue->tail)
			queue->tail->next = alarm;
		else
			queue->head = alarm;
		queue->tail = alarm;
		queue->threadQueue.wakeupEntireWaitQueue(false);
	}

	static void __OSRemoveAlarmFromQueue(OSAlarm_t* alarm)
	{
		OSAlarmQueue* queue = alarm->alarmQueue.GetPtr();
		if (alarm->next)
			alarm->next->prev = alarm->prev;
		else
			queue->tail = alarm->prev;
		if (alarm->prev)
			alarm->prev->next = alarm->next;
		else
			queue->head = alarm->next;
		alarm->prev = nullptr;
		alarm->next = nullptr;
	}

	bool OSCancelAlarm(OSAlarm_t* alarm)
	{
		__OSLockScheduler();
		bool alarmWasActive = false;
		const int slot = __OSFindActiveAlarm(alarm);
		if (slot >= 0)
		{
			__OSEndActiveAlarm(slot);
			alarmWasActive = true;
		}
		else if (__OSIsAlarmInQueue(alarm))
		{
			__OSRemoveAlarmFromQueue(alarm);
			alarmWasActive = true;
		}

		__OSUnlockScheduler();
		return alarmWasActive;
	}

	void OSCancelAlarms(uint32 group)
	{
		if (group == 0 || group == 0xFFFFFFFF)
			return;
		__OSLockScheduler();
		// from the end: a slot removed takes the last one's place
		for (unsigned i = os_guest_alarms_count(&s_guestAlarms); i-- > 0;)
		{
			const int slot = os_guest_alarms_at(&s_guestAlarms, i);
			OSAlarm_t* active = MEMPTR<OSAlarm_t>(s_guestAlarms.addr[slot]).GetPtr();
			if (active->group == group)
				__OSEndActiveAlarm(slot);
		}
		__OSUnlockScheduler();
	}

	void OSSetAlarmTag(OSAlarm_t* alarm, uint32 group)
	{
		if (group == 0xFFFFFFFF)
			return;
		__OSLockScheduler();
		alarm->group = group;
		__OSUnlockScheduler();
	}

	void __OSHostAlarmTriggered(uint64 currentTick, void* context)
	{
#ifdef ALARM_LOGGING
		cemuLog_log(LogType::Force, "[Alarm] Alarm ready and alarm thread signalled. Current tick: {}", currentTick);
#endif
		OSSignalEventInternal(g_alarmEvent.GetPtr());
	}

	void __OSInitiateAlarm(OSAlarm_t* alarm, uint64 startTime, uint64 period, MPTR handlerFunc, bool isPeriodic, OSAlarmQueue* alarmQueue)
	{
        cemu_assert_debug(MMU_IsInPPCMemorySpace(alarm));
		cemu_assert_debug(__OSHasSchedulerLock());

		// alarms which fired and still wait in their alarm queue cannot be set
		if (__OSIsAlarmInQueue(alarm))
			return;

		uint64 nextTime = startTime;

#ifdef ALARM_LOGGING
		double periodInMS = (double)period * 1000.0 / (double)EspressoTime::GetTimerClock();
		cemuLog_log(LogType::Force, "[Alarm] Start alarm 0x{:08x}. Func 0x{:08x}. Period: {}ms", MEMPTR(alarm).GetMPTR(), handlerFunc, periodInMS);
#endif

		if (isPeriodic)
		{
			if ((sint64)period <= 0)
			{
				cemuLog_log(LogType::APIErrors, "OSSetPeriodicAlarm(): Period must be positive, alarm is not set");
				return;
			}

			uint64 currentTime = OSGetTime();
			if (startTime < currentTime)
			{
				uint64 ticksSinceStart = currentTime - startTime;
				uint64 numPeriods = ticksSinceStart / period;
				nextTime = startTime + (numPeriods + 1ull) * period;
			}

			alarm->startTime = _swapEndianU64(startTime);
			alarm->nextTime = _swapEndianU64(nextTime);
			alarm->period = _swapEndianU64(period);
			alarm->handler = _swapEndianU32(handlerFunc);
		}
		else
		{
			alarm->nextTime = _swapEndianU64(startTime);
			alarm->period = 0;
			alarm->handler = _swapEndianU32(handlerFunc);
		}
		alarm->alarmQueue = alarmQueue;

		int slot = __OSFindActiveAlarm(alarm);
		if (slot >= 0)
			cemuLog_logDebug(LogType::Force, "__OSInitiateAlarm() called on alarm which was already active");
		else
		{
			slot = os_guest_alarms_add(&s_guestAlarms, MEMPTR<OSAlarm_t>(alarm).GetMPTR());
			if (slot < 0)
			{
				cemuLog_log(LogType::Force, "OSSetAlarm(): more than {} alarms set at once, alarm 0x{:08x} is not set", OS_GUEST_ALARMS, MEMPTR<OSAlarm_t>(alarm).GetMPTR());
				return;
			}
		}
		OSHostAlarmStart(&s_guestAlarms.host[slot], nextTime, period, __OSHostAlarmTriggered, nullptr);
	}

	bool OSSetAlarm(OSAlarm_t* alarm, uint64 delayInTicks, MPTR handlerFunc)
	{
		if ((sint64)delayInTicks <= 0)
		{
			cemuLog_log(LogType::APIErrors, "OSSetAlarm(): Delay must be positive, alarm is not set");
			return true;
		}
		__OSLockScheduler();
		__OSInitiateAlarm(alarm, OSGetTime() + delayInTicks, 0, handlerFunc, false, nullptr);
		__OSUnlockScheduler();
		return true;
	}

	bool OSSetPeriodicAlarm(OSAlarm_t* alarm, uint64 nextFire, uint64 period, MPTR handlerFunc)
	{
		__OSLockScheduler();
		__OSInitiateAlarm(alarm, nextFire, period, handlerFunc, true, nullptr);
		__OSUnlockScheduler();
		return true;
	}

	bool OSSetAlarmEx(OSAlarm_t* alarm, uint64 delayInTicks, MPTR handlerFunc, OSAlarmQueue* alarmQueue)
	{
		if ((sint64)delayInTicks <= 0)
		{
			cemuLog_log(LogType::APIErrors, "OSSetAlarmEx(): Delay must be positive, alarm is not set");
			return true;
		}
		__OSLockScheduler();
		__OSInitiateAlarm(alarm, OSGetTime() + delayInTicks, 0, handlerFunc, false, alarmQueue);
		__OSUnlockScheduler();
		return true;
	}

	bool OSSetPeriodicAlarmEx(OSAlarm_t* alarm, uint64 nextFire, uint64 period, MPTR handlerFunc, OSAlarmQueue* alarmQueue)
	{
		__OSLockScheduler();
		__OSInitiateAlarm(alarm, nextFire, period, handlerFunc, true, alarmQueue);
		__OSUnlockScheduler();
		return true;
	}

	bool OSGetAlarmFromQueue(OSAlarmQueue* queue, MEMPTR<OSAlarm_t>* alarmOut, MEMPTR<void>* contextOut, bool noWait)
	{
		__OSLockScheduler();
		while (!queue->head)
		{
			if (noWait)
			{
				*alarmOut = nullptr;
				*contextOut = nullptr;
				__OSUnlockScheduler();
				return false;
			}
			queue->threadQueue.queueAndWait(OSGetCurrentThread());
		}
		OSAlarm_t* alarm = queue->head.GetPtr();
		__OSRemoveAlarmFromQueue(alarm);
		// periodic alarms are only set again once they are taken out of their alarm queue
		if (alarm->period != 0)
			__OSInitiateAlarm(alarm, _swapEndianU64(alarm->startTime), _swapEndianU64(alarm->period), _swapEndianU32(alarm->handler), true, queue);
		*alarmOut = alarm;
		*contextOut = alarm->context;
		__OSUnlockScheduler();
		return true;
	}

	void OSSetAlarmUserData(OSAlarm_t* alarm, uint32 userData)
	{
		alarm->userData = userData;
	}

	uint32 OSGetAlarmUserData(OSAlarm_t* alarm)
	{
		return alarm->userData;
	}

	void OSAlarm_Shutdown()
	{
		__OSLockScheduler();
		os_guest_alarms_init(&s_guestAlarms);
		os_alarm_list_init(&s_hostAlarms);
		s_soonestAlarm.store(0, std::memory_order_relaxed);
		__OSUnlockScheduler();
	}

	void _OSAlarmThread(PPCInterpreter_t* hCPU)
	{
		while( true )
		{
			OSWaitEvent(g_alarmEvent.GetPtr());
			uint64 currentTick = OSGetTime();
			while (true)
			{
				// get alarm to fire
				OSAlarm_t* alarm = nullptr;
				__OSLockScheduler();
				// the one due soonest
				int dueSlot = -1;
				uint64 dueTime = 0;
				for (unsigned i = 0; i < os_guest_alarms_count(&s_guestAlarms); i++)
				{
					const int slot = os_guest_alarms_at(&s_guestAlarms, i);
					const uint64 nextTime = _swapEndianU64(MEMPTR<OSAlarm_t>(s_guestAlarms.addr[slot]).GetPtr()->nextTime);
					if (currentTick >= nextTime && (dueSlot < 0 || nextTime < dueTime))
					{
						dueSlot = slot;
						dueTime = nextTime;
					}
				}
				if (dueSlot >= 0)
				{
					alarm = MEMPTR<OSAlarm_t>(s_guestAlarms.addr[dueSlot]).GetPtr();
					if (alarm->alarmQueue || alarm->period == 0)
						__OSEndActiveAlarm(dueSlot); // queued, or done
					else
						alarm->nextTime = _swapEndianU64(_swapEndianU64(alarm->nextTime) + _swapEndianU64(alarm->period));
				}
				if (alarm && alarm->alarmQueue)
				{
					alarm->context = &g_alarmThread.GetPtr()->context;
					__OSAppendAlarmToQueue(alarm);
					__OSUnlockScheduler();
					continue;
				}
				__OSUnlockScheduler();
				if (!alarm)
					break;
				// do callback for alarm
#ifdef ALARM_LOGGING
				double periodInMS = (double)_swapEndianU64(alarm->period) * 1000.0 / (double)EspressoTime::GetTimerClock();
				cemuLog_log(LogType::Force, "[Alarm] Callback 0x{:08x} for alarm 0x{:08x}. Current tick: {}. Period: {}ms", _swapEndianU32(alarm->handler), MEMPTR(alarm).GetMPTR(), currentTick, periodInMS);
#endif
				PPCCoreCallback(_swapEndianU32(alarm->handler), alarm, &(g_alarmThread.GetPtr()->context));
			}
		}
	}

	void MapAlarmExports()
	{
		cafeExportRegister("coreinit", OSCreateAlarm, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSCreateAlarmEx, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSCancelAlarm, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSCancelAlarms, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSSetAlarmTag, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSSetAlarm, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSSetPeriodicAlarm, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSSetAlarmEx, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSSetPeriodicAlarmEx, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSInitAlarmQueue, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSInitAlarmQueueEx, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSGetAlarmFromQueue, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSSetAlarmUserData, LogType::CoreinitAlarm);
		cafeExportRegister("coreinit", OSGetAlarmUserData, LogType::CoreinitAlarm);
	}

	void InitializeAlarm()
	{
		__OSLockScheduler();
		os_guest_alarms_init(&s_guestAlarms);
		os_alarm_list_init(&s_hostAlarms);
		UpdateSoonestAlarm();
		__OSUnlockScheduler();

		// init event
		OSInitEvent(g_alarmEvent.GetPtr(), OSEvent::EVENT_STATE::STATE_NOT_SIGNALED, OSEvent::EVENT_MODE::MODE_AUTO);

		// create alarm callback handler thread
		coreinit::OSCreateThreadType(g_alarmThread.GetPtr(), RPLLoader_MakePPCCallable(_OSAlarmThread), 0, nullptr, _g_alarmThreadStack.GetPtr() + _g_alarmThreadStack.GetByteSize(), (sint32)_g_alarmThreadStack.GetByteSize(), 0, 0x7, OSThread_t::THREAD_TYPE::TYPE_IO);
		OSResumeThread(g_alarmThread.GetPtr());
		strcpy(_g_alarmThreadName.GetPtr(), "Alarm Thread");
		coreinit::OSSetThreadName(g_alarmThread.GetPtr(), _g_alarmThreadName.GetPtr());
	}
}
