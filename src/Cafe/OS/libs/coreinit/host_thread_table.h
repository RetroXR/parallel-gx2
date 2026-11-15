#ifndef COREINIT_HOST_THREAD_TABLE_H
#define COREINIT_HOST_THREAD_TABLE_H

/* Which host thread (a fiber and its interpreter, in a pool kept for the
 * title's life) runs which guest thread, with nothing allocated as threads
 * come and go. Slots are handed out once each as the pool grows, then
 * again from the ones given back, newest first - but never the one the
 * caller names (the fiber that is running: an ended thread's, still on its
 * way to the scheduler). A guest thread's slot is found by its guest
 * address through an open addressed index. With the scheduler lock. C89. */

#include <stdint.h>

#include <retro_common_api.h>

RETRO_BEGIN_DECLS

#define HOST_THREADS 256
#define HOST_THREAD_INDEX (HOST_THREADS * 4)

typedef struct host_thread_table
{
   uint32_t index_thread[HOST_THREAD_INDEX]; /* the guest thread, 0 empty */
   uint16_t index_slot[HOST_THREAD_INDEX];
   uint16_t free_slots[HOST_THREADS];
   unsigned free_count;
   unsigned slot_count;                     /* handed out so far */
} host_thread_table_t;

void host_thread_table_init(host_thread_table_t *t);
/* thread's slot, or -1 */
int host_thread_table_find(const host_thread_table_t *t, uint32_t thread);
/* A slot for thread (not in the table): one given back, newest first, but
 * not avoid; else the next one never handed out (*fresh set); -1 when all
 * HOST_THREADS are taken */
int host_thread_table_take(host_thread_table_t *t, uint32_t thread, int avoid, int *fresh);
/* thread's slot given back; -1 when it had none */
int host_thread_table_give(host_thread_table_t *t, uint32_t thread);

RETRO_END_DECLS

#endif
