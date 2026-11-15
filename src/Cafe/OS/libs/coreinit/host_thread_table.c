/* Which host thread runs which guest thread (host_thread_table.h) */

#include <string.h>

#include "host_thread_table.h"

static unsigned host_thread_hash(uint32_t thread)
{
   return (unsigned)(((thread >> 3) * 2654435761u) >> 22) & (HOST_THREAD_INDEX - 1);
}

void host_thread_table_init(host_thread_table_t *t)
{
   memset(t, 0, sizeof(*t));
}

int host_thread_table_find(const host_thread_table_t *t, uint32_t thread)
{
   unsigned i = host_thread_hash(thread);
   for (;;)
   {
      if (!t->index_thread[i])
         return -1;
      if (t->index_thread[i] == thread)
         return t->index_slot[i];
      i = (i + 1) & (HOST_THREAD_INDEX - 1);
   }
}

int host_thread_table_take(host_thread_table_t *t, uint32_t thread, int avoid, int *fresh)
{
   unsigned i;
   int slot = -1;
   *fresh = 0;
   /* newest given back first, passing over the one to avoid */
   for (i = t->free_count; i-- > 0;)
   {
      if ((int)t->free_slots[i] == avoid)
         continue;
      slot = t->free_slots[i];
      t->free_slots[i] = t->free_slots[--t->free_count];
      break;
   }
   if (slot < 0)
   {
      if (t->slot_count >= HOST_THREADS)
         return -1;
      slot = (int)t->slot_count++;
      *fresh = 1;
   }
   i = host_thread_hash(thread);
   while (t->index_thread[i])
      i = (i + 1) & (HOST_THREAD_INDEX - 1);
   t->index_thread[i] = thread;
   t->index_slot[i]   = (uint16_t)slot;
   return slot;
}

int host_thread_table_give(host_thread_table_t *t, uint32_t thread)
{
   unsigned i = host_thread_hash(thread);
   unsigned hole;
   int slot;
   for (;;)
   {
      if (!t->index_thread[i])
         return -1;
      if (t->index_thread[i] == thread)
         break;
      i = (i + 1) & (HOST_THREAD_INDEX - 1);
   }
   slot = t->index_slot[i];
   /* backward shift: an entry after the hole that would not be found past
    * it moves into it */
   hole = i;
   for (;;)
   {
      unsigned home;
      i = (i + 1) & (HOST_THREAD_INDEX - 1);
      if (!t->index_thread[i])
         break;
      home = host_thread_hash(t->index_thread[i]);
      if (hole <= i ? (home > hole && home <= i) : (home > hole || home <= i))
         continue;
      t->index_thread[hole] = t->index_thread[i];
      t->index_slot[hole]   = t->index_slot[i];
      hole = i;
   }
   t->index_thread[hole] = 0;
   t->index_slot[hole]   = 0;
   t->free_slots[t->free_count++] = (uint16_t)slot;
   return slot;
}
