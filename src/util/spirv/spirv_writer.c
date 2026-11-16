#include <stdlib.h>
#include <string.h>

#include "spirv_writer.h"
#include "spirv_ops.h"

static int words_reserve(spv_writer_t *w, spv_words_t *s, size_t more)
{
   size_t cap;
   uint32_t *p;
   if (w->failed)
      return 0;
   if (s->n + more <= s->cap)
      return 1;
   cap = s->cap ? s->cap : 256;
   while (cap < s->n + more)
      cap *= 2;
   p = (uint32_t*)realloc(s->w, cap * sizeof(uint32_t));
   if (!p)
   {
      w->failed = 1;
      return 0;
   }
   s->w = p;
   s->cap = cap;
   return 1;
}

void spv_writer_init(spv_writer_t *w)
{
   memset(w, 0, sizeof(*w));
   w->bound = 1;
}

void spv_writer_reset(spv_writer_t *w)
{
   unsigned i;
   for (i = 0; i < SPV_SEC_COUNT; i++)
      w->sec[i].n = 0;
   w->out.n = 0;
   if (w->dedup && w->dedup_used)
      memset(w->dedup, 0, w->dedup_cap * sizeof(uint32_t));
   w->dedup_used = 0;
   w->bound = 1;
   w->failed = 0;
}

void spv_writer_free(spv_writer_t *w)
{
   unsigned i;
   for (i = 0; i < SPV_SEC_COUNT; i++)
      free(w->sec[i].w);
   free(w->out.w);
   free(w->dedup);
   memset(w, 0, sizeof(*w));
}

uint32_t spv_id(spv_writer_t *w)
{
   return w->bound++;
}

void spv_inst(spv_writer_t *w, unsigned sec, uint32_t opcode, const uint32_t *operands, size_t n)
{
   spv_words_t *s = &w->sec[sec];
   if (!words_reserve(w, s, n + 1))
      return;
   s->w[s->n++] = ((uint32_t)(n + 1) << 16) | opcode;
   if (n)
      memcpy(s->w + s->n, operands, n * sizeof(uint32_t));
   s->n += n;
}

void spv_inst_str(spv_writer_t *w, unsigned sec, uint32_t opcode, const uint32_t *operands, size_t n,
      const char *str, const uint32_t *after, size_t n_after)
{
   spv_words_t *s = &w->sec[sec];
   size_t len = strlen(str);
   size_t str_words = len / 4 + 1; /* the nul and the padding */
   size_t total = 1 + n + str_words + n_after;
   if (!words_reserve(w, s, total))
      return;
   s->w[s->n++] = ((uint32_t)total << 16) | opcode;
   if (n)
      memcpy(s->w + s->n, operands, n * sizeof(uint32_t));
   s->n += n;
   memset(s->w + s->n, 0, str_words * sizeof(uint32_t));
   memcpy(s->w + s->n, str, len);
   s->n += str_words;
   if (n_after)
      memcpy(s->w + s->n, after, n_after * sizeof(uint32_t));
   s->n += n_after;
}

/* the hash of an instruction's words with its result id left out */
static uint32_t inst_hash(const uint32_t *inst, size_t count, size_t result_at)
{
   uint32_t h = 2166136261u;
   size_t i;
   for (i = 0; i < count; i++)
   {
      uint32_t v = i == result_at + 1 ? 0 : inst[i];
      h = (h ^ v) * 16777619u;
      h ^= h >> 15;
   }
   return h ? h : 1;
}

static int inst_equal(const uint32_t *a, const uint32_t *b, size_t count, size_t result_at)
{
   size_t i;
   for (i = 0; i < count; i++)
      if (i != result_at + 1 && a[i] != b[i])
         return 0;
   return 1;
}

/* the table's slots: the offset of an instruction in the global section,
 * plus one (0 is empty); its result id is read back from there */
static int dedup_grow(spv_writer_t *w);

static size_t dedup_find(spv_writer_t *w, const uint32_t *inst, size_t count, size_t result_at, uint32_t h, int *found)
{
   size_t mask = w->dedup_cap - 1;
   size_t i = h & mask;
   const spv_words_t *g = &w->sec[SPV_SEC_GLOBAL];
   for (;;)
   {
      uint32_t slot = w->dedup[i];
      if (!slot)
      {
         *found = 0;
         return i;
      }
      {
         const uint32_t *other = g->w + (slot - 1);
         if ((other[0] >> 16) == count && inst_equal(other, inst, count, result_at))
         {
            *found = 1;
            return i;
         }
      }
      i = (i + 1) & mask;
   }
}

static int dedup_grow(spv_writer_t *w)
{
   size_t cap = w->dedup_cap ? w->dedup_cap * 2 : 256;
   uint32_t *table = (uint32_t*)calloc(cap, sizeof(uint32_t));
   uint32_t *old = w->dedup;
   size_t old_cap = w->dedup_cap;
   size_t i;
   if (!table)
   {
      w->failed = 1;
      return 0;
   }
   w->dedup = table;
   w->dedup_cap = cap;
   /* the instructions kept: each found again by its words */
   for (i = 0; i < old_cap; i++)
   {
      uint32_t slot = old[i];
      const uint32_t *inst;
      size_t count, at, result_at;
      uint32_t op;
      int found;
      if (!slot)
         continue;
      inst = w->sec[SPV_SEC_GLOBAL].w + (slot - 1);
      count = inst[0] >> 16;
      op = inst[0] & 0xFFFF;
      /* a constant's result id is after its type, a type's first */
      result_at = (op >= SpvOpConstantTrue && op <= SpvOpConstantNull) ? 1 : 0;
      at = dedup_find(w, inst, count, result_at, inst_hash(inst, count, result_at), &found);
      w->dedup[at] = slot;
   }
   free(old);
   return 1;
}

uint32_t spv_global(spv_writer_t *w, uint32_t opcode, const uint32_t *operands, size_t n, size_t result_at)
{
   uint32_t inst[16];
   size_t count = n + 1;
   uint32_t h, id;
   size_t at;
   int found;
   spv_words_t *g = &w->sec[SPV_SEC_GLOBAL];
   if (w->failed || count > 16)
   {
      w->failed = 1;
      return 0;
   }
   inst[0] = ((uint32_t)count << 16) | opcode;
   memcpy(inst + 1, operands, n * sizeof(uint32_t));
   inst[result_at + 1] = 0;
   if ((w->dedup_used + 1) * 2 > w->dedup_cap && !dedup_grow(w))
      return 0;
   h = inst_hash(inst, count, result_at);
   at = dedup_find(w, inst, count, result_at, h, &found);
   if (found)
      return g->w[w->dedup[at] - 1 + result_at + 1];
   if (!words_reserve(w, g, count))
      return 0;
   id = spv_id(w);
   inst[result_at + 1] = id;
   w->dedup[at] = (uint32_t)(g->n + 1);
   w->dedup_used++;
   memcpy(g->w + g->n, inst, count * sizeof(uint32_t));
   g->n += count;
   return id;
}

uint32_t spv_type_void(spv_writer_t *w)
{
   uint32_t o[1] = { 0 };
   return spv_global(w, SpvOpTypeVoid, o, 1, 0);
}

uint32_t spv_type_bool(spv_writer_t *w)
{
   uint32_t o[1] = { 0 };
   return spv_global(w, SpvOpTypeBool, o, 1, 0);
}

uint32_t spv_type_int(spv_writer_t *w, int is_signed)
{
   uint32_t o[3];
   o[0] = 0;
   o[1] = 32;
   o[2] = is_signed ? 1 : 0;
   return spv_global(w, SpvOpTypeInt, o, 3, 0);
}

uint32_t spv_type_float(spv_writer_t *w)
{
   uint32_t o[2];
   o[0] = 0;
   o[1] = 32;
   return spv_global(w, SpvOpTypeFloat, o, 2, 0);
}

uint32_t spv_type_vector(spv_writer_t *w, uint32_t component, uint32_t count)
{
   uint32_t o[3];
   o[0] = 0;
   o[1] = component;
   o[2] = count;
   return spv_global(w, SpvOpTypeVector, o, 3, 0);
}

uint32_t spv_type_image(spv_writer_t *w, uint32_t sampled_type, uint32_t dim, int arrayed)
{
   uint32_t o[8];
   o[0] = 0;
   o[1] = sampled_type;
   o[2] = dim;
   o[3] = 0;                 /* no depth */
   o[4] = arrayed ? 1u : 0u;
   o[5] = 0;                 /* single sampled */
   o[6] = 1;                 /* sampled */
   o[7] = 0;                 /* Unknown format */
   return spv_global(w, SpvOpTypeImage, o, 8, 0);
}

uint32_t spv_type_sampled_image(spv_writer_t *w, uint32_t image)
{
   uint32_t o[2];
   o[0] = 0;
   o[1] = image;
   return spv_global(w, SpvOpTypeSampledImage, o, 2, 0);
}

uint32_t spv_type_array(spv_writer_t *w, uint32_t element, uint32_t length_const)
{
   uint32_t o[3];
   o[0] = 0;
   o[1] = element;
   o[2] = length_const;
   return spv_global(w, SpvOpTypeArray, o, 3, 0);
}

uint32_t spv_type_pointer(spv_writer_t *w, uint32_t storage_class, uint32_t type)
{
   uint32_t o[3];
   o[0] = 0;
   o[1] = storage_class;
   o[2] = type;
   return spv_global(w, SpvOpTypePointer, o, 3, 0);
}

uint32_t spv_type_function(spv_writer_t *w, uint32_t ret, const uint32_t *params, size_t n)
{
   uint32_t o[16];
   if (n > 14)
   {
      w->failed = 1;
      return 0;
   }
   o[0] = 0;
   o[1] = ret;
   if (n)
      memcpy(o + 2, params, n * sizeof(uint32_t));
   return spv_global(w, SpvOpTypeFunction, o, n + 2, 0);
}

uint32_t spv_type_struct(spv_writer_t *w, const uint32_t *members, size_t n)
{
   spv_words_t *g = &w->sec[SPV_SEC_GLOBAL];
   uint32_t id;
   if (!words_reserve(w, g, n + 2))
      return 0;
   id = spv_id(w);
   g->w[g->n++] = ((uint32_t)(n + 2) << 16) | SpvOpTypeStruct;
   g->w[g->n++] = id;
   if (n)
      memcpy(g->w + g->n, members, n * sizeof(uint32_t));
   g->n += n;
   return id;
}

uint32_t spv_const_bits(spv_writer_t *w, uint32_t type, uint32_t bits)
{
   uint32_t o[3];
   o[0] = type;
   o[1] = 0;
   o[2] = bits;
   return spv_global(w, SpvOpConstant, o, 3, 1);
}

uint32_t spv_const_bool(spv_writer_t *w, int value)
{
   uint32_t o[2];
   o[0] = spv_type_bool(w);
   o[1] = 0;
   return spv_global(w, value ? SpvOpConstantTrue : SpvOpConstantFalse, o, 2, 1);
}

uint32_t spv_const_composite(spv_writer_t *w, uint32_t type, const uint32_t *parts, size_t n)
{
   uint32_t o[16];
   if (n > 14)
   {
      w->failed = 1;
      return 0;
   }
   o[0] = type;
   o[1] = 0;
   memcpy(o + 2, parts, n * sizeof(uint32_t));
   return spv_global(w, SpvOpConstantComposite, o, n + 2, 1);
}

uint32_t spv_const_null(spv_writer_t *w, uint32_t type)
{
   uint32_t o[2];
   o[0] = type;
   o[1] = 0;
   return spv_global(w, SpvOpConstantNull, o, 2, 1);
}

uint32_t spv_global_variable(spv_writer_t *w, uint32_t pointer_type, uint32_t storage_class)
{
   uint32_t o[3];
   o[0] = pointer_type;
   o[1] = spv_id(w);
   o[2] = storage_class;
   spv_inst(w, SPV_SEC_GLOBAL, SpvOpVariable, o, 3);
   return w->failed ? 0 : o[1];
}

const uint32_t *spv_writer_finish(spv_writer_t *w, size_t *count)
{
   size_t total = 5;
   unsigned i;
   *count = 0;
   for (i = 0; i < SPV_SEC_COUNT; i++)
      total += w->sec[i].n;
   w->out.n = 0;
   if (!words_reserve(w, &w->out, total))
      return NULL;
   w->out.w[0] = 0x07230203u;   /* magic */
   w->out.w[1] = 0x00010000u;   /* 1.0 */
   w->out.w[2] = 0;             /* generator */
   w->out.w[3] = w->bound;
   w->out.w[4] = 0;
   w->out.n = 5;
   for (i = 0; i < SPV_SEC_COUNT; i++)
   {
      if (w->sec[i].n)
         memcpy(w->out.w + w->out.n, w->sec[i].w, w->sec[i].n * sizeof(uint32_t));
      w->out.n += w->sec[i].n;
   }
   *count = w->out.n;
   return w->out.w;
}
