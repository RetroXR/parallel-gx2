#include "size_blocks.h"

#include <stdlib.h>
#include <string.h>

static unsigned size_blocks_class(size_t count)
{
   unsigned c = 0;
   while (c + 1 < SIZE_BLOCKS_CLASSES && ((size_t)1 << c) < count)
      c++;
   return c;
}

void size_blocks_init(size_blocks_t *b, size_t unit, unsigned keep_max)
{
   memset(b, 0, sizeof(*b));
   b->unit = unit < sizeof(void*) ? sizeof(void*) : unit;
   b->keep_max = keep_max;
}

void *size_blocks_take(size_blocks_t *b, size_t count, size_t *capacity)
{
   unsigned c = size_blocks_class(count ? count : 1);
   size_t units = (size_t)1 << c;
   void *block = b->free_list[c];
   if (block)
   {
      b->free_list[c] = *(void**)block;
      b->kept[c]--;
   }
   else
   {
      block = malloc(units * b->unit);
      if (!block)
         return NULL;
   }
   memset(block, 0, units * b->unit);
   if (capacity)
      *capacity = units;
   return block;
}

void size_blocks_give(size_blocks_t *b, void *block, size_t count)
{
   unsigned c;
   if (!block)
      return;
   c = size_blocks_class(count ? count : 1);
   if (b->kept[c] >= b->keep_max)
   {
      free(block);
      return;
   }
   *(void**)block = b->free_list[c];
   b->free_list[c] = block;
   b->kept[c]++;
}

unsigned size_blocks_kept(const size_blocks_t *b, size_t count)
{
   return b->kept[size_blocks_class(count ? count : 1)];
}

void size_blocks_free_all(size_blocks_t *b)
{
   unsigned c;
   for (c = 0; c < SIZE_BLOCKS_CLASSES; c++)
   {
      while (b->free_list[c])
      {
         void *next = *(void**)b->free_list[c];
         free(b->free_list[c]);
         b->free_list[c] = next;
      }
      b->kept[c] = 0;
   }
}
