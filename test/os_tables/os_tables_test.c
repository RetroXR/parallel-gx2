/* coreinit's tables without locks: the virtual address range a title
 * allocates from (src/Cafe/OS/libs/coreinit/virt_range.c) - the lowest
 * free pages that fit, freed once and only where an allocation begins, a
 * range across words given back whole when another takes part of it, and
 * in 32 byte pages as nn_fp's IPC buffers take it - and the FSA clients
 * (fsa_clients.c) - found by handle, a closed one not mistaken for the
 * next in its slot, taken out once - and the alarm tables (alarm_table.c):
 * host alarms fired in time order, periodic ones again a period on, a
 * stopped or restarted one where it should be; the title's alarms found by
 * address however their hashes collide, a removed one gone and the rest
 * still found, a full table refusing. Under TSan and ASan,
 * threads allocate and free at once and none is handed pages another
 * holds. C89, as the C89 lane builds it. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <retro_atomic.h>
#include <rthreads/rthreads.h>

#include "Cafe/OS/libs/coreinit/virt_range.h"
#include "Cafe/OS/libs/coreinit/fsa_clients.h"
#include "Cafe/OS/libs/coreinit/alarm_table.h"

extern void (*virt_range_test_mid_alloc)(void);

static int failures = 0;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #x); failures++; } } while (0)

#define BASE 0xA0000000u
#define SIZE 0x40000000u
#define PAGE 0x20000u
#define AT(p) (BASE + (uint32_t)(p) * PAGE)

static virt_range_t s_r;

static void test_basic(void)
{
   virt_range_init(&s_r, BASE, SIZE, PAGE);
   CHECK(virt_range_alloc(&s_r, 1, 1) == AT(0));
   CHECK(virt_range_alloc(&s_r, PAGE + 1, 1) == AT(1)); /* two pages */
   CHECK(virt_range_alloc(&s_r, 0, 1) == AT(3));        /* still a page */
   CHECK(!virt_range_free(&s_r, AT(2)));                /* inside, not a start */
   CHECK(!virt_range_free(&s_r, AT(1) + 4));
   CHECK(!virt_range_free(&s_r, BASE - PAGE));
   CHECK(!virt_range_free(&s_r, AT(8192)));
   CHECK(!virt_range_free(&s_r, AT(9)));                /* nothing there */
   CHECK(virt_range_free(&s_r, AT(1)));
   CHECK(!virt_range_free(&s_r, AT(1)));                /* once */
   /* the lowest hole that fits */
   CHECK(virt_range_alloc(&s_r, 3 * PAGE, 1) == AT(4));
   CHECK(virt_range_alloc(&s_r, 2 * PAGE, 1) == AT(1));
   CHECK(virt_range_alloc(&s_r, PAGE, 1) == AT(7));
   /* aligned past a page */
   CHECK(virt_range_alloc(&s_r, PAGE, 16 * PAGE) == AT(16));
   CHECK(virt_range_alloc(&s_r, PAGE, 1) == AT(8));
}

static void test_words(void)
{
   uint32_t a;
   virt_range_init(&s_r, BASE, SIZE, PAGE);
   CHECK(virt_range_alloc(&s_r, 30 * PAGE, 1) == AT(0));
   a = virt_range_alloc(&s_r, 5 * PAGE, 1);             /* across the first two words */
   CHECK(a == AT(30));
   CHECK(virt_range_alloc(&s_r, 31 * PAGE, 1) == AT(35));
   CHECK(virt_range_free(&s_r, a));
   CHECK(virt_range_alloc(&s_r, 5 * PAGE, 1) == AT(30));
   /* full words passed over whole, and nothing past them missed */
   virt_range_init(&s_r, BASE, SIZE, PAGE);
   CHECK(virt_range_alloc(&s_r, 64 * PAGE, 1) == AT(0));
   CHECK(virt_range_alloc(&s_r, PAGE, 1) == AT(64));
   CHECK(virt_range_alloc(&s_r, 31 * PAGE, 1) == AT(65));
   CHECK(virt_range_alloc(&s_r, PAGE, 1) == AT(96));
   /* all of it, then nothing */
   virt_range_init(&s_r, BASE, SIZE, PAGE);
   CHECK(virt_range_alloc(&s_r, SIZE + 1, 1) == 0);
   CHECK(virt_range_alloc(&s_r, SIZE, 1) == BASE);
   CHECK(virt_range_alloc(&s_r, 1, 1) == 0);
   CHECK(virt_range_free(&s_r, BASE));
   CHECK(virt_range_alloc(&s_r, 0xFFFFFFFFu, 1) == 0);
   CHECK(virt_range_alloc(&s_r, SIZE - PAGE, 1) == BASE);
   CHECK(virt_range_alloc(&s_r, PAGE, 1) == AT(8191));
   CHECK(virt_range_alloc(&s_r, PAGE, 1) == 0);
}

/* another allocation takes part of the pages found, between finding and
 * taking them */
static uint32_t s_inner;
static uint32_t s_inner_size, s_inner_align;

static void alloc_inside(void)
{
   virt_range_test_mid_alloc = NULL;
   s_inner = virt_range_alloc(&s_r, s_inner_size, s_inner_align);
}

static void test_race(void)
{
   uint32_t a;
   /* in the first word: the outer one goes on past it */
   virt_range_init(&s_r, BASE, SIZE, PAGE);
   s_inner_size = PAGE;
   s_inner_align = 1;
   virt_range_test_mid_alloc = alloc_inside;
   a = virt_range_alloc(&s_r, 3 * PAGE, 1);
   virt_range_test_mid_alloc = NULL;
   CHECK(s_inner == AT(0));
   CHECK(a == AT(1));
   CHECK(virt_range_free(&s_r, a));
   CHECK(virt_range_alloc(&s_r, 3 * PAGE, 1) == AT(1));
   /* in the second word: the pages taken in the first are given back */
   virt_range_init(&s_r, BASE, SIZE, PAGE);
   CHECK(virt_range_alloc(&s_r, 30 * PAGE, 1) == AT(0));
   s_inner_size = PAGE;
   s_inner_align = 32 * PAGE;
   virt_range_test_mid_alloc = alloc_inside;
   a = virt_range_alloc(&s_r, 5 * PAGE, 1);
   virt_range_test_mid_alloc = NULL;
   CHECK(s_inner == AT(32));
   CHECK(a == AT(33));
   CHECK(virt_range_alloc(&s_r, 2 * PAGE, 1) == AT(30));
}

/* threads allocating and freeing at once: no page held twice */
#define THREADS 4
#define ROUNDS 20000
static retro_atomic_int_t s_owner[VIRT_RANGE_MAX_PAGES];
static retro_atomic_int_t s_overlaps;
static retro_atomic_int_t s_go;

static void worker(void *ud)
{
   int id = (int)(size_t)ud;
   unsigned seed = 1234u + (unsigned)id * 7919u;
   uint32_t held[8];
   uint32_t held_n[8];
   int n = 0, i;
   while (!retro_atomic_load_acquire_int(&s_go))
      sthread_yield();
   for (i = 0; i < ROUNDS; i++)
   {
      seed = seed * 1103515245u + 12345u;
      if (n < 8 && (n == 0 || (seed >> 16) & 1))
      {
         uint32_t pages = 1 + ((seed >> 20) % 8), a, p;
         a = virt_range_alloc(&s_r, pages * PAGE, 1);
         if (!a)
            continue;
         for (p = (a - BASE) / PAGE; p < (a - BASE) / PAGE + pages; p++)
            if (!retro_atomic_cas_int(&s_owner[p], 0, id))
               retro_atomic_fetch_add_int(&s_overlaps, 1);
         held[n] = a;
         held_n[n++] = pages;
      }
      else
      {
         uint32_t p, a = held[--n];
         for (p = (a - BASE) / PAGE; p < (a - BASE) / PAGE + held_n[n]; p++)
            if (!retro_atomic_cas_int(&s_owner[p], id, 0))
               retro_atomic_fetch_add_int(&s_overlaps, 1);
         if (!virt_range_free(&s_r, a))
            retro_atomic_fetch_add_int(&s_overlaps, 1);
      }
   }
   while (n > 0)
   {
      uint32_t p, a = held[--n];
      for (p = (a - BASE) / PAGE; p < (a - BASE) / PAGE + held_n[n]; p++)
         retro_atomic_cas_int(&s_owner[p], id, 0);
      virt_range_free(&s_r, a);
   }
}

static void test_threads(void)
{
   sthread_t *t[THREADS];
   int i;
   /* 64 pages: the threads keep meeting */
   virt_range_init(&s_r, BASE, 64 * PAGE, PAGE);
   for (i = 0; i < VIRT_RANGE_MAX_PAGES; i++)
      retro_atomic_int_init(&s_owner[i], 0);
   retro_atomic_int_init(&s_overlaps, 0);
   retro_atomic_int_init(&s_go, 0);
   for (i = 0; i < THREADS; i++)
      t[i] = sthread_create(worker, (void*)(size_t)(i + 1));
   retro_atomic_store_release_int(&s_go, 1);
   for (i = 0; i < THREADS; i++)
      sthread_join(t[i]);
   CHECK(retro_atomic_load_acquire_int(&s_overlaps) == 0);
   /* all given back */
   CHECK(virt_range_alloc(&s_r, 64 * PAGE, 1) == BASE);
}

static void test_fsa_clients(void)
{
   fsa_clients_reset();
   CHECK(!fsa_clients_add(0));
   CHECK(!fsa_clients_add(0x1080));                     /* index 128 */
   CHECK(fsa_clients_add(0x1005));
   CHECK(fsa_clients_has(0x1005));
   CHECK(!fsa_clients_has(0x2005));                     /* same index, not this one */
   CHECK(!fsa_clients_has(0));
   CHECK(!fsa_clients_remove(0x2005));
   CHECK(fsa_clients_has(0x1005));
   CHECK(fsa_clients_remove(0x1005));
   CHECK(!fsa_clients_remove(0x1005));                  /* once */
   CHECK(!fsa_clients_has(0x1005));
   /* a closed handle's index given out again: only the new one is a client */
   CHECK(fsa_clients_add(0x1007));
   CHECK(fsa_clients_add(0x3007));
   CHECK(!fsa_clients_has(0x1007));
   CHECK(fsa_clients_has(0x3007));
   CHECK(!fsa_clients_remove(0x1007));
   CHECK(fsa_clients_has(0x3007));
   CHECK(fsa_clients_add(0x105F));
   fsa_clients_reset();
   CHECK(!fsa_clients_has(0x3007) && !fsa_clients_has(0x105F));
}

/* two threads deleting the same client: one takes it out */
static retro_atomic_int_t s_removed;
static uint32_t s_handle;

static void remover(void *ud)
{
   (void)ud;
   while (!retro_atomic_load_acquire_int(&s_go))
      sthread_yield();
   if (fsa_clients_remove(s_handle))
      retro_atomic_fetch_add_int(&s_removed, 1);
}

static void test_fsa_threads(void)
{
   int round;
   retro_atomic_int_init(&s_removed, 0);
   fsa_clients_reset();
   for (round = 0; round < 300; round++)
   {
      sthread_t *a, *b;
      s_handle = ((uint32_t)(round + 1) << 12) | (uint32_t)(round % 96);
      CHECK(fsa_clients_add(s_handle));
      retro_atomic_store_release_int(&s_go, 0);
      a = sthread_create(remover, NULL);
      b = sthread_create(remover, NULL);
      retro_atomic_store_release_int(&s_go, 1);
      sthread_join(a);
      sthread_join(b);
      CHECK(!fsa_clients_has(s_handle));
   }
   CHECK(retro_atomic_load_acquire_int(&s_removed) == 300);
}

/* nn_fp's IPC buffers: 0x12000 bytes of the console's memory in 32 byte
 * pages, sizes of a few bytes to a few KB, each aligned to 32 and inside,
 * none overlapping, and all of it there again once given back */
#define FP_BASE 0x10001000u
#define FP_SIZE 0x12000u
static void test_fp_buffers(void)
{
   static uint32_t got[FP_SIZE / 32];
   static unsigned char owner[FP_SIZE / 32];
   int n = 0, i;
   uint32_t size = 1;
   virt_range_init(&s_r, FP_BASE, FP_SIZE, 32);
   memset(owner, 0, sizeof(owner));
   for (;;)
   {
      uint32_t a = virt_range_alloc(&s_r, size, 32), p;
      if (!a)
         break;
      CHECK(a >= FP_BASE && a + size <= FP_BASE + FP_SIZE && !(a & 31));
      for (p = (a - FP_BASE) / 32; p < (a - FP_BASE + size + 31) / 32; p++)
      {
         CHECK(!owner[p]);
         owner[p] = 1;
      }
      got[n++] = a;
      size = size * 7 % 3001 + 1;
   }
   CHECK(n > 10);
   for (i = 0; i < n; i++)
      CHECK(virt_range_free(&s_r, got[i]));
   CHECK(virt_range_alloc(&s_r, FP_SIZE, 32) == FP_BASE);
}

/* --- alarm tables ------------------------------------------------------ */

static int s_fired[64];
static int s_fired_count;

static void record_fire(uint64_t now, void *ctx)
{
   (void)now;
   if (s_fired_count < 64)
      s_fired[s_fired_count++] = (int)(intptr_t)ctx;
}

static os_host_alarm_t s_host[8];
static os_alarm_list_t s_list;

static void stop_self(uint64_t now, void *ctx)
{
   record_fire(now, ctx);
   os_alarm_list_stop(&s_list, &s_host[(int)(intptr_t)ctx]);
}

static void test_alarm_list(void)
{
   int i;
   memset(s_host, 0, sizeof(s_host));
   os_alarm_list_init(&s_list);
   CHECK(os_alarm_list_soonest(&s_list) == UINT64_MAX);
   /* out of order, two due at once (lower address first) */
   os_alarm_list_start(&s_list, &s_host[3], 300, 0, record_fire, (void *)3);
   os_alarm_list_start(&s_list, &s_host[1], 100, 0, record_fire, (void *)1);
   os_alarm_list_start(&s_list, &s_host[2], 200, 0, record_fire, (void *)2);
   os_alarm_list_start(&s_list, &s_host[0], 200, 0, record_fire, (void *)0);
   CHECK(os_alarm_list_soonest(&s_list) == 100);
   s_fired_count = 0;
   os_alarm_list_fire(&s_list, 99);
   CHECK(s_fired_count == 0);
   os_alarm_list_fire(&s_list, 200);
   CHECK(s_fired_count == 3);
   CHECK(s_fired[0] == 1 && s_fired[1] == 0 && s_fired[2] == 2);
   CHECK(!s_host[1].active && s_host[3].active);
   CHECK(os_alarm_list_soonest(&s_list) == 300);
   /* restarted: moved, not on the list twice */
   os_alarm_list_start(&s_list, &s_host[3], 50, 0, record_fire, (void *)3);
   os_alarm_list_start(&s_list, &s_host[3], 400, 0, record_fire, (void *)3);
   CHECK(os_alarm_list_soonest(&s_list) == 400);
   os_alarm_list_stop(&s_list, &s_host[3]);
   os_alarm_list_stop(&s_list, &s_host[3]);
   CHECK(os_alarm_list_soonest(&s_list) == UINT64_MAX);
   /* periodic: every period, until stopped; and one stopping itself */
   os_alarm_list_start(&s_list, &s_host[4], 1000, 100, record_fire, (void *)4);
   os_alarm_list_start(&s_list, &s_host[5], 1050, 100, stop_self, (void *)5);
   s_fired_count = 0;
   os_alarm_list_fire(&s_list, 1250);
   /* 4 at 1000, 1100, 1200; 5 once at 1050 */
   CHECK(s_fired_count == 4);
   CHECK(s_fired[0] == 4 && s_fired[1] == 5 && s_fired[2] == 4 && s_fired[3] == 4);
   CHECK(os_alarm_list_soonest(&s_list) == 1300);
   CHECK(!s_host[5].active);
   os_alarm_list_stop(&s_list, &s_host[4]);
   for (i = 0; i < 8; i++)
      CHECK(!s_host[i].active);
   CHECK(s_list.head == NULL);
}

static os_guest_alarms_t s_guest;
static uint32_t s_ref_addr[OS_GUEST_ALARMS + 1];
static int s_ref_slot[OS_GUEST_ALARMS + 1];

static void test_guest_alarms(void)
{
   unsigned n = 0;
   unsigned step;
   unsigned i;
   uint32_t seed = 12345;
   os_guest_alarms_init(&s_guest);
   CHECK(os_guest_alarms_find(&s_guest, 0x10000000u) == -1);
   for (step = 0; step < 200000; step++)
   {
      unsigned op;
      seed = seed * 1103515245u + 12345u;
      op = (seed >> 16) % 3;
      if (op < 2 && n < OS_GUEST_ALARMS)
      {
         /* addresses from a narrow range, 0x58 apart as an array of alarms
          * is, so plenty collide */
         uint32_t addr;
         int found = 0;
         int slot;
         seed = seed * 1103515245u + 12345u;
         addr = 0x10000000u + ((seed >> 8) % 4096u) * 0x58u;
         for (i = 0; i < n; i++)
            if (s_ref_addr[i] == addr)
               found = 1;
         if (found)
            continue;
         slot = os_guest_alarms_add(&s_guest, addr);
         CHECK(slot >= 0 && slot < OS_GUEST_ALARMS);
         if (slot < 0)
            break;
         s_ref_addr[n] = addr;
         s_ref_slot[n] = slot;
         n++;
      }
      else if (n)
      {
         unsigned k;
         seed = seed * 1103515245u + 12345u;
         k = (seed >> 8) % n;
         os_guest_alarms_remove(&s_guest, s_ref_slot[k]);
         CHECK(os_guest_alarms_find(&s_guest, s_ref_addr[k]) == -1);
         s_ref_addr[k] = s_ref_addr[n - 1];
         s_ref_slot[k] = s_ref_slot[n - 1];
         n--;
      }
      if ((step & 255) == 0)
      {
         unsigned seen = 0;
         for (i = 0; i < n; i++)
            CHECK(os_guest_alarms_find(&s_guest, s_ref_addr[i]) == s_ref_slot[i]);
         CHECK(os_guest_alarms_count(&s_guest) == n);
         for (i = 0; i < os_guest_alarms_count(&s_guest); i++)
         {
            int slot = os_guest_alarms_at(&s_guest, i);
            unsigned j;
            for (j = 0; j < n; j++)
               if (s_ref_slot[j] == slot)
                  seen++;
         }
         CHECK(seen == n);
      }
   }
   /* full: every slot taken, the next refused, all still found */
   os_guest_alarms_init(&s_guest);
   for (i = 0; i < OS_GUEST_ALARMS; i++)
      CHECK(os_guest_alarms_add(&s_guest, 0x20000000u + i * 0x58u) >= 0);
   CHECK(os_guest_alarms_add(&s_guest, 0x30000000u) == -1);
   for (i = 0; i < OS_GUEST_ALARMS; i++)
      CHECK(os_guest_alarms_find(&s_guest, 0x20000000u + i * 0x58u) >= 0);
   CHECK(os_guest_alarms_find(&s_guest, 0x30000000u) == -1);
}

int main(void)
{
   test_basic();
   test_words();
   test_race();
   test_threads();
   test_fp_buffers();
   test_fsa_clients();
   test_fsa_threads();
   test_alarm_list();
   test_guest_alarms();
   if (failures)
   {
      fprintf(stderr, "os_tables_test: %d failed\n", failures);
      return 1;
   }
   printf("os_tables_test: all passed\n");
   return 0;
}
