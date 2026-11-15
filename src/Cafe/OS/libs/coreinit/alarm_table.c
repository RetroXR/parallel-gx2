/* coreinit's alarms, with nothing allocated (alarm_table.h) */

#include <string.h>

#include "alarm_table.h"

void os_alarm_list_init(os_alarm_list_t *l)
{
   l->head = NULL;
}

/* a before b: sooner, or as soon and lower in memory */
static bool alarm_before(const os_host_alarm_t *a, const os_host_alarm_t *b)
{
   if (a->fire != b->fire)
      return a->fire < b->fire;
   return (uintptr_t)a < (uintptr_t)b;
}

static void alarm_link(os_alarm_list_t *l, os_host_alarm_t *a)
{
   os_host_alarm_t *prev = NULL;
   os_host_alarm_t *cur  = l->head;
   while (cur && alarm_before(cur, a))
   {
      prev = cur;
      cur  = cur->next;
   }
   a->prev = prev;
   a->next = cur;
   if (cur)
      cur->prev = a;
   if (prev)
      prev->next = a;
   else
      l->head = a;
   a->active = 1;
}

static void alarm_unlink(os_alarm_list_t *l, os_host_alarm_t *a)
{
   if (a->prev)
      a->prev->next = a->next;
   else
      l->head = a->next;
   if (a->next)
      a->next->prev = a->prev;
   a->prev   = NULL;
   a->next   = NULL;
   a->active = 0;
}

void os_alarm_list_start(os_alarm_list_t *l, os_host_alarm_t *a,
      uint64_t fire, uint64_t period, os_alarm_cb_t cb, void *ctx)
{
   if (a->active)
      alarm_unlink(l, a);
   a->fire   = fire;
   a->period = period;
   a->cb     = cb;
   a->ctx    = ctx;
   alarm_link(l, a);
}

void os_alarm_list_stop(os_alarm_list_t *l, os_host_alarm_t *a)
{
   if (a->active)
      alarm_unlink(l, a);
}

uint64_t os_alarm_list_soonest(const os_alarm_list_t *l)
{
   return l->head ? l->head->fire : UINT64_MAX;
}

void os_alarm_list_fire(os_alarm_list_t *l, uint64_t now)
{
   while (l->head && now >= l->head->fire)
   {
      os_host_alarm_t *a = l->head;
      alarm_unlink(l, a);
      /* a periodic alarm is back on the list before its call, which may
       * stop it */
      if (a->period)
      {
         a->fire += a->period;
         alarm_link(l, a);
      }
      a->cb(now, a->ctx);
   }
}

static unsigned alarm_hash(uint32_t addr)
{
   return (unsigned)(((addr >> 2) * 2654435761u) >> 21) & (OS_GUEST_ALARM_INDEX - 1);
}

void os_guest_alarms_init(os_guest_alarms_t *g)
{
   unsigned i;
   memset(g, 0, sizeof(*g));
   for (i = 0; i < OS_GUEST_ALARMS; i++)
      g->free_slots[i] = (uint16_t)(OS_GUEST_ALARMS - 1 - i);
   g->free_count = OS_GUEST_ALARMS;
}

int os_guest_alarms_find(const os_guest_alarms_t *g, uint32_t addr)
{
   unsigned i = alarm_hash(addr);
   for (;;)
   {
      unsigned e = g->index[i];
      if (!e)
         return -1;
      if (g->addr[e - 1] == addr)
         return (int)(e - 1);
      i = (i + 1) & (OS_GUEST_ALARM_INDEX - 1);
   }
}

int os_guest_alarms_add(os_guest_alarms_t *g, uint32_t addr)
{
   unsigned i;
   int slot;
   if (!g->free_count)
      return -1;
   slot = g->free_slots[--g->free_count];
   memset(&g->host[slot], 0, sizeof(g->host[slot]));
   g->addr[slot] = addr;
   i = alarm_hash(addr);
   while (g->index[i])
      i = (i + 1) & (OS_GUEST_ALARM_INDEX - 1);
   g->index[i] = (uint16_t)(slot + 1);
   g->pos[slot] = (uint16_t)g->count;
   g->dense[g->count++] = (uint16_t)slot;
   return slot;
}

void os_guest_alarms_remove(os_guest_alarms_t *g, int slot)
{
   unsigned i = alarm_hash(g->addr[slot]);
   unsigned hole;
   unsigned last;
   while (g->index[i] != (uint16_t)(slot + 1))
      i = (i + 1) & (OS_GUEST_ALARM_INDEX - 1);
   /* backward shift: an entry after the hole that would not be found past
    * it moves into it */
   hole = i;
   for (;;)
   {
      unsigned home;
      unsigned e;
      i = (i + 1) & (OS_GUEST_ALARM_INDEX - 1);
      e = g->index[i];
      if (!e)
         break;
      home = alarm_hash(g->addr[e - 1]);
      /* home cyclically in (hole, i]: stays */
      if (hole <= i ? (home > hole && home <= i) : (home > hole || home <= i))
         continue;
      g->index[hole] = (uint16_t)e;
      hole = i;
   }
   g->index[hole] = 0;
   /* out of the dense array: the last one into its place */
   last = g->dense[--g->count];
   g->dense[g->pos[slot]] = (uint16_t)last;
   g->pos[last] = g->pos[slot];
   g->addr[slot] = 0;
   g->free_slots[g->free_count++] = (uint16_t)slot;
}

unsigned os_guest_alarms_count(const os_guest_alarms_t *g)
{
   return g->count;
}

int os_guest_alarms_at(const os_guest_alarms_t *g, unsigned i)
{
   return g->dense[i];
}
