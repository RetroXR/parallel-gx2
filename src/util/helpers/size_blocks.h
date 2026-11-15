#ifndef UTIL_HELPERS_SIZE_BLOCKS_H
#define UTIL_HELPERS_SIZE_BLOCKS_H

/* Blocks of a power of two of units, kept for reuse once given back: a
 * free list for each power of two, up to keep_max blocks in each (more are
 * freed). For storage whose length varies with what it holds - a GPU buffer
 * range's pages - made and let go while a game plays: once blocks as long
 * have been made, nothing more is allocated. One thread's (no locks). C89. */

#include <stddef.h>

#include <retro_common_api.h>

RETRO_BEGIN_DECLS

#define SIZE_BLOCKS_CLASSES 32

typedef struct size_blocks
{
   void *free_list[SIZE_BLOCKS_CLASSES];
   unsigned kept[SIZE_BLOCKS_CLASSES];
   unsigned keep_max;
   size_t unit;
} size_blocks_t;

/* unit: the octets of one unit (at least a pointer's) */
void size_blocks_init(size_blocks_t *b, size_t unit, unsigned keep_max);
/* A zeroed block of at least count units (count rounded up to a power of
 * two, which *capacity gets if not NULL); NULL when out of memory */
void *size_blocks_take(size_blocks_t *b, size_t count, size_t *capacity);
/* A block taken for count units back */
void size_blocks_give(size_blocks_t *b, void *block, size_t count);
/* How many blocks of count units are kept */
unsigned size_blocks_kept(const size_blocks_t *b, size_t count);
/* Every block kept freed */
void size_blocks_free_all(size_blocks_t *b);

RETRO_END_DECLS

#endif
