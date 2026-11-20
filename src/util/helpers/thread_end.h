#ifndef UTIL_HELPERS_THREAD_END_H
#define UTIL_HELPERS_THREAD_END_H

/* What a thread keeps for itself - a scratch buffer, a decoder, a hazard
 * slot - freed as the thread ends without a thread_local destructor: MinGW
 * (emutls) frees a thread_local's storage before it runs its destructor, so
 * one that holds anything reads freed memory. The thread reaches its object
 * through a thread_local pointer, trivially destroyed, and registers it on
 * its list, which a pthread key's destructor is handed itself as the thread
 * ends. Any thread, the frontend's included. C89. */

#include <boolean.h>
#include <retro_common_api.h>

RETRO_BEGIN_DECLS

#define THREAD_END_MAX 64

/* the key, at the core's start (made at the first registration otherwise,
 * once); false, and nothing kept for any thread, without it */
bool thread_end_init(void);
/* the calling thread's list run, then the key gone until thread_end_init:
 * at the core's end, so no thread's end runs into the module once it is
 * unloaded (a thread still holding a list then keeps it) */
void thread_end_deinit(void);

/* fn(arg) run as the calling thread ends, or by thread_end_run; slot is the
 * thread's own pointer to it, set to NULL by thread_end_run (never as the
 * thread ends, when the thread_local may be gone) - NULL for none. False,
 * and nothing run, with no key or THREAD_END_MAX registered already. */
bool thread_end_register(void (*fn)(void *arg), void *arg, void **slot);

/* the calling thread's list run now, newest first, its slots set to NULL:
 * for a thread that ends without returning (_endthreadex) or that goes on
 * after the core's end */
void thread_end_run(void);

/* registered on the calling thread, for the tests */
unsigned thread_end_count(void);

RETRO_END_DECLS

#endif
