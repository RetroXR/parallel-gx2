#ifndef COREINIT_ALARM_TABLE_H
#define COREINIT_ALARM_TABLE_H

/* coreinit's alarms, with nothing allocated while a title runs. All of it
 * is used with the scheduler lock held. C89.
 *
 * A host alarm is a node its owner keeps - on the stack for OSSleepTicks
 * and OSWaitEventWithTimeout, in the guest alarm table for OSSetAlarm - on
 * a list sorted by when it fires (then by address, so two due at once keep
 * an order). Firing pops what is due, calls it and puts a periodic one back
 * a period on.
 *
 * The guest alarm table pairs a title's OSAlarm with a host alarm: a fixed
 * pool of slots, found by the alarm's guest address through an open
 * addressed index, and the slots in use kept in a dense array to walk. */

#include <stdint.h>

#include <boolean.h>
#include <retro_common_api.h>

RETRO_BEGIN_DECLS

typedef void (*os_alarm_cb_t)(uint64_t now, void *ctx);

typedef struct os_host_alarm
{
   uint64_t fire;
   uint64_t period; /* 0: once */
   os_alarm_cb_t cb;
   void *ctx;
   struct os_host_alarm *prev;
   struct os_host_alarm *next;
   int active;
} os_host_alarm_t;

typedef struct os_alarm_list
{
   os_host_alarm_t *head;
} os_alarm_list_t;

/* empty; a node is zeroed before its first start */
void os_alarm_list_init(os_alarm_list_t *l);
/* a, set again if it was on the list */
void os_alarm_list_start(os_alarm_list_t *l, os_host_alarm_t *a,
      uint64_t fire, uint64_t period, os_alarm_cb_t cb, void *ctx);
/* off the list, if it was on it */
void os_alarm_list_stop(os_alarm_list_t *l, os_host_alarm_t *a);
/* when the first fires; UINT64_MAX when none is set */
uint64_t os_alarm_list_soonest(const os_alarm_list_t *l);
/* calls every alarm due by now, soonest first */
void os_alarm_list_fire(os_alarm_list_t *l, uint64_t now);

#define OS_GUEST_ALARMS 1024
#define OS_GUEST_ALARM_INDEX (OS_GUEST_ALARMS * 2)

typedef struct os_guest_alarms
{
   os_host_alarm_t host[OS_GUEST_ALARMS];
   uint32_t addr[OS_GUEST_ALARMS];             /* the guest alarm, in use */
   uint16_t index[OS_GUEST_ALARM_INDEX];       /* slot + 1, 0 empty */
   uint16_t dense[OS_GUEST_ALARMS];            /* the slots in use */
   uint16_t pos[OS_GUEST_ALARMS];              /* a slot's place in dense */
   uint16_t free_slots[OS_GUEST_ALARMS];
   unsigned count;
   unsigned free_count;
} os_guest_alarms_t;

/* empty: every host alarm must be off its list */
void os_guest_alarms_init(os_guest_alarms_t *g);
/* addr's slot, or -1 */
int os_guest_alarms_find(const os_guest_alarms_t *g, uint32_t addr);
/* a slot for addr (not in the table), its host alarm zeroed; -1 when full */
int os_guest_alarms_add(os_guest_alarms_t *g, uint32_t addr);
/* the slot freed; its host alarm must be off its list */
void os_guest_alarms_remove(os_guest_alarms_t *g, int slot);
unsigned os_guest_alarms_count(const os_guest_alarms_t *g);
/* the i-th slot in use, i below the count; removing it moves the last one
 * into its place */
int os_guest_alarms_at(const os_guest_alarms_t *g, unsigned i);

RETRO_END_DECLS

#endif
