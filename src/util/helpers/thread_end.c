#include <pthread.h>
#include <stdlib.h>

#include <retro_atomic.h>

#include "thread_end.h"

typedef struct thread_end_entry
{
   void (*fn)(void *arg);
   void *arg;
   void **slot;
} thread_end_entry_t;

typedef struct thread_end_list
{
   unsigned count;
   thread_end_entry_t entries[THREAD_END_MAX];
} thread_end_list_t;

static pthread_key_t s_key;
static pthread_once_t s_once = PTHREAD_ONCE_INIT;
/* read by any thread registering; set once the key is made, cleared at the
 * core's end, when s_gone is set until the core starts again */
static retro_atomic_int_t s_key_made;
static retro_atomic_int_t s_gone;

/* the key's destructor: the list, the thread's storage maybe gone */
static void thread_end_list_run(void *p)
{
   thread_end_list_t *list = (thread_end_list_t*)p;
   while (list->count)
   {
      thread_end_entry_t *e = &list->entries[--list->count];
      e->fn(e->arg);
   }
   free(list);
}

static void thread_end_make_key(void)
{
   if (pthread_key_create(&s_key, thread_end_list_run) == 0)
      retro_atomic_store_release_int(&s_key_made, 1);
}

/* the key, made at the first registration if the core's start has not */
static bool thread_end_have_key(void)
{
   if (retro_atomic_load_acquire_int(&s_key_made))
      return true;
   if (retro_atomic_load_acquire_int(&s_gone))
      return false;
   pthread_once(&s_once, thread_end_make_key);
   return retro_atomic_load_acquire_int(&s_key_made) != 0;
}

bool thread_end_init(void)
{
   if (retro_atomic_load_acquire_int(&s_key_made))
      return true;
   if (retro_atomic_load_acquire_int(&s_gone))
   {
      /* started again after its end: no thread of the core runs yet */
      if (pthread_key_create(&s_key, thread_end_list_run) != 0)
         return false;
      retro_atomic_store_release_int(&s_gone, 0);
      retro_atomic_store_release_int(&s_key_made, 1);
      return true;
   }
   return thread_end_have_key();
}

void thread_end_deinit(void)
{
   if (!retro_atomic_load_acquire_int(&s_key_made))
      return;
   thread_end_run();
   retro_atomic_store_release_int(&s_gone, 1);
   retro_atomic_store_release_int(&s_key_made, 0);
   pthread_key_delete(s_key);
}

bool thread_end_register(void (*fn)(void *arg), void *arg, void **slot)
{
   thread_end_list_t *list;
   thread_end_entry_t *e;
   if (!thread_end_have_key())
      return false;
   list = (thread_end_list_t*)pthread_getspecific(s_key);
   if (!list)
   {
      list = (thread_end_list_t*)calloc(1, sizeof(*list));
      if (!list)
         return false;
      if (pthread_setspecific(s_key, list) != 0)
      {
         free(list);
         return false;
      }
   }
   if (list->count == THREAD_END_MAX)
      return false;
   e = &list->entries[list->count++];
   e->fn = fn;
   e->arg = arg;
   e->slot = slot;
   return true;
}

void thread_end_run(void)
{
   thread_end_list_t *list;
   if (!retro_atomic_load_acquire_int(&s_key_made))
      return;
   list = (thread_end_list_t*)pthread_getspecific(s_key);
   if (!list)
      return;
   pthread_setspecific(s_key, NULL);
   while (list->count)
   {
      thread_end_entry_t *e = &list->entries[--list->count];
      if (e->slot)
         *e->slot = NULL;
      e->fn(e->arg);
   }
   free(list);
}

unsigned thread_end_count(void)
{
   thread_end_list_t *list;
   if (!retro_atomic_load_acquire_int(&s_key_made))
      return 0;
   list = (thread_end_list_t*)pthread_getspecific(s_key);
   return list ? list->count : 0;
}
