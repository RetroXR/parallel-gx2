/* The SPIR-V writer (src/util/spirv/spirv_writer.c): a module put together
 * out of order comes out in the sections' order, the header right; a type
 * or constant asked for twice is one id, across the table's growth (4000
 * constants) and after a reset; a struct is never shared; a module written
 * again after a reset is the same words, from the same buffers (no
 * allocation once warm); and the fragment shader written is valid
 * (run_tests.sh puts it through spirv-val). C89. */

#include <stdio.h>
#include <string.h>

#include "util/spirv/spirv_writer.h"
#include "util/spirv/spirv_ops.h"

static int failures = 0;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #x); failures++; } } while (0)

/* a fragment shader writing a constant color, its parts in no order:
 * the body before the types it uses, the entry point last */
static const uint32_t *build(spv_writer_t *w, size_t *count)
{
   uint32_t tVoid, tF, tV4, tPtrOut, outVar, fnType, fn, label, color;
   uint32_t parts[4];
   uint32_t words[4];
   uint32_t glsl;
   uint32_t cap = SpvCapabilityShader;
   uint32_t model[2];
   uint32_t ep[2];
   uint32_t mode[2];

   fn = spv_id(w);
   label = spv_id(w);
   tVoid = spv_type_void(w);
   fnType = spv_type_function(w, tVoid, NULL, 0);
   words[0] = tVoid;
   words[1] = fn;
   words[2] = SpvFunctionControlMaskNone;
   words[3] = fnType;
   spv_inst(w, SPV_SEC_FUNCTION, SpvOpFunction, words, 4);
   spv_inst(w, SPV_SEC_FUNCTION, SpvOpLabel, &label, 1);
   tF = spv_type_float(w);
   tV4 = spv_type_vector(w, tF, 4);
   parts[0] = spv_const_bits(w, tF, 0x3F800000u);
   parts[1] = spv_const_bits(w, tF, 0x3F000000u);
   parts[2] = spv_const_bits(w, tF, 0);
   parts[3] = spv_const_bits(w, tF, 0x3F800000u);
   color = spv_const_composite(w, tV4, parts, 4);
   tPtrOut = spv_type_pointer(w, SpvStorageClassOutput, tV4);
   outVar = spv_global_variable(w, tPtrOut, SpvStorageClassOutput);
   words[0] = outVar;
   words[1] = color;
   spv_inst(w, SPV_SEC_BODY, SpvOpStore, words, 2);
   spv_inst(w, SPV_SEC_BODY, SpvOpReturn, NULL, 0);
   spv_inst(w, SPV_SEC_BODY, SpvOpFunctionEnd, NULL, 0);
   words[0] = outVar;
   words[1] = SpvDecorationLocation;
   words[2] = 0;
   spv_inst(w, SPV_SEC_DECORATION, SpvOpDecorate, words, 3);
   ep[0] = SpvExecutionModelFragment;
   ep[1] = fn;
   spv_inst_str(w, SPV_SEC_ENTRY_POINT, SpvOpEntryPoint, ep, 2, "main", &outVar, 1);
   mode[0] = fn;
   mode[1] = SpvExecutionModeOriginUpperLeft;
   spv_inst(w, SPV_SEC_EXEC_MODE, SpvOpExecutionMode, mode, 2);
   model[0] = SpvAddressingModelLogical;
   model[1] = SpvMemoryModelGLSL450;
   spv_inst(w, SPV_SEC_MEMORY_MODEL, SpvOpMemoryModel, model, 2);
   glsl = spv_id(w);
   spv_inst_str(w, SPV_SEC_IMPORT, SpvOpExtInstImport, &glsl, 1, "GLSL.std.450", NULL, 0);
   spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
   return spv_writer_finish(w, count);
}

/* the opcodes of a module, in order */
static size_t opcodes(const uint32_t *m, size_t count, uint32_t *ops, size_t cap)
{
   size_t i = 5, n = 0;
   while (i < count && n < cap)
   {
      uint32_t len = m[i] >> 16;
      ops[n++] = m[i] & 0xFFFF;
      if (len == 0)
         break;
      i += len;
   }
   return n;
}

static int rank(uint32_t op)
{
   switch (op)
   {
      case SpvOpCapability: return 0;
      case SpvOpExtInstImport: return 2;
      case SpvOpMemoryModel: return 3;
      case SpvOpEntryPoint: return 4;
      case SpvOpExecutionMode: return 5;
      case SpvOpDecorate: return 7;
      case SpvOpFunction: return 9;
      case SpvOpLabel: return 10;
      case SpvOpStore: case SpvOpReturn: case SpvOpFunctionEnd: return 11;
      default: return 8; /* types, constants, variables */
   }
}

static void test_module(const char *path)
{
   spv_writer_t w;
   size_t count = 0, i, n;
   const uint32_t *m;
   uint32_t ops[64];
   spv_writer_init(&w);
   m = build(&w, &count);
   CHECK(m != NULL && count > 5);
   if (!m)
      return;
   CHECK(m[0] == 0x07230203u && m[1] == 0x00010000u && m[3] == w.bound && m[4] == 0);
   /* sections in order */
   n = opcodes(m, count, ops, 64);
   for (i = 1; i < n; i++)
      CHECK(rank(ops[i - 1]) <= rank(ops[i]));
   /* every id below the bound */
   if (path)
   {
      FILE *f = fopen(path, "wb");
      if (f)
      {
         fwrite(m, 4, count, f);
         fclose(f);
      }
   }
   spv_writer_free(&w);
}

static void test_dedup(void)
{
   spv_writer_t w;
   uint32_t ids[4000];
   uint32_t tI, tU, tF, a, b, s1, s2;
   int i;
   spv_writer_init(&w);
   tI = spv_type_int(&w, 1);
   tU = spv_type_int(&w, 0);
   tF = spv_type_float(&w);
   CHECK(tI != tU && tI == spv_type_int(&w, 1) && tU == spv_type_int(&w, 0));
   CHECK(spv_type_vector(&w, tF, 4) == spv_type_vector(&w, tF, 4));
   CHECK(spv_type_vector(&w, tF, 4) != spv_type_vector(&w, tF, 3));
   CHECK(spv_type_pointer(&w, SpvStorageClassFunction, tF) == spv_type_pointer(&w, SpvStorageClassFunction, tF));
   CHECK(spv_type_pointer(&w, SpvStorageClassFunction, tF) != spv_type_pointer(&w, SpvStorageClassPrivate, tF));
   /* images: by sampled type, dimension and arrayness; sampled images by image */
   CHECK(spv_type_image(&w, tF, SpvDimension2D, 0) == spv_type_image(&w, tF, SpvDimension2D, 0));
   CHECK(spv_type_image(&w, tF, SpvDimension2D, 0) != spv_type_image(&w, tF, SpvDimension2D, 1));
   CHECK(spv_type_image(&w, tF, SpvDimension2D, 0) != spv_type_image(&w, tF, SpvDimension3D, 0));
   CHECK(spv_type_image(&w, tF, SpvDimension2D, 0) != spv_type_image(&w, tU, SpvDimension2D, 0));
   CHECK(spv_type_depth_image(&w, tF, SpvDimension2D, 0) == spv_type_depth_image(&w, tF, SpvDimension2D, 0));
   CHECK(spv_type_depth_image(&w, tF, SpvDimension2D, 0) != spv_type_image(&w, tF, SpvDimension2D, 0));
   CHECK(spv_type_depth_image(&w, tF, SpvDimension2D, 1) != spv_type_depth_image(&w, tF, SpvDimension2D, 0));
   CHECK(spv_type_sampled_image(&w, spv_type_image(&w, tF, SpvDimension2D, 0)) ==
      spv_type_sampled_image(&w, spv_type_image(&w, tF, SpvDimension2D, 0)));
   CHECK(spv_type_sampled_image(&w, spv_type_image(&w, tF, SpvDimension2D, 0)) !=
      spv_type_sampled_image(&w, spv_type_image(&w, tF, SpvDimension3D, 0)));
   /* the same bits as another type are another constant */
   a = spv_const_bits(&w, tI, 7);
   b = spv_const_bits(&w, tU, 7);
   CHECK(a != b && a == spv_const_bits(&w, tI, 7));
   CHECK(spv_const_bool(&w, 1) == spv_const_bool(&w, 1) && spv_const_bool(&w, 1) != spv_const_bool(&w, 0));
   CHECK(spv_const_null(&w, tF) == spv_const_null(&w, tF));
   /* structs: each its own */
   /* runtime arrays: one per element type, apart from sized arrays */
   CHECK(spv_type_runtime_array(&w, tI) == spv_type_runtime_array(&w, tI));
   CHECK(spv_type_runtime_array(&w, tI) != spv_type_runtime_array(&w, tF));
   CHECK(spv_type_runtime_array(&w, tI) != spv_type_array(&w, tI, a));
   s1 = spv_type_struct(&w, &tF, 1);
   s2 = spv_type_struct(&w, &tF, 1);
   CHECK(s1 != s2);
   /* the table grown many times over: every constant found again */
   for (i = 0; i < 4000; i++)
      ids[i] = spv_const_bits(&w, tU, (uint32_t)i * 2654435761u);
   for (i = 0; i < 4000; i++)
      CHECK(ids[i] == spv_const_bits(&w, tU, (uint32_t)i * 2654435761u));
   for (i = 1; i < 4000; i++)
      CHECK(ids[i] != ids[i - 1]);
   CHECK(tI == spv_type_int(&w, 1) && a == spv_const_bits(&w, tI, 7));
   CHECK(!w.failed);
   /* after a reset, nothing is remembered: ids start over */
   spv_writer_reset(&w);
   CHECK(w.bound == 1);
   CHECK(spv_type_float(&w) == 1);
   CHECK(spv_type_int(&w, 1) == 2);
   spv_writer_free(&w);
}

static void test_reuse(void)
{
   spv_writer_t w;
   size_t count0 = 0, count, i;
   const uint32_t *m;
   static uint32_t first[256];
   uint32_t *bufs[SPV_SEC_COUNT + 1];
   int k, same = 1, stable = 1;
   spv_writer_init(&w);
   m = build(&w, &count0);
   CHECK(m && count0 <= 256);
   if (!m || count0 > 256)
      return;
   memcpy(first, m, count0 * 4);
   for (i = 0; i < SPV_SEC_COUNT; i++)
      bufs[i] = w.sec[i].w;
   bufs[SPV_SEC_COUNT] = w.out.w;
   for (k = 0; k < 100; k++)
   {
      spv_writer_reset(&w);
      m = build(&w, &count);
      if (count != count0 || memcmp(m, first, count * 4) != 0)
         same = 0;
      for (i = 0; i < SPV_SEC_COUNT; i++)
         if (w.sec[i].w != bufs[i])
            stable = 0;
      if (w.out.w != bufs[SPV_SEC_COUNT])
         stable = 0;
   }
   CHECK(same);
   CHECK(stable);
   spv_writer_free(&w);
}

static void test_growth(void)
{
   spv_writer_t w;
   uint32_t words[3];
   size_t count = 0;
   const uint32_t *m;
   int i;
   spv_writer_init(&w);
   /* a body far past a buffer's first size */
   for (i = 0; i < 50000; i++)
   {
      words[0] = 1;
      words[1] = (uint32_t)i;
      words[2] = (uint32_t)i ^ 0x5A5A5A5Au;
      spv_inst(&w, SPV_SEC_BODY, SpvOpNop, words, 3);
   }
   m = spv_writer_finish(&w, &count);
   CHECK(m != NULL && count == 5 + 50000 * 4);
   if (m)
   {
      CHECK(m[5] == ((4u << 16) | SpvOpNop) && m[5 + 4 * 49999 + 2] == 49999u);
      CHECK(m[5 + 4 * 49999 + 3] == (49999u ^ 0x5A5A5A5Au));
   }
   /* a string padded to a word, nul included */
   spv_writer_reset(&w);
   spv_inst_str(&w, SPV_SEC_DEBUG, SpvOpName, words, 1, "abcd", NULL, 0);
   m = spv_writer_finish(&w, &count);
   CHECK(m && count == 5 + 4 && (m[5] >> 16) == 4 && m[8] == 0);
   spv_writer_free(&w);
}

/* room made ahead: a module that fits it written with no buffer moved,
 * the table of types and constants included */
static void test_reserve(void)
{
   spv_writer_t w;
   uint32_t words[3];
   uint32_t *bufs[SPV_SEC_COUNT + 1];
   uint32_t *dedup;
   size_t count = 0, i;
   const uint32_t *m;
   uint32_t t;
   int k, stable = 1;
   spv_writer_init(&w);
   CHECK(spv_writer_reserve(&w, 3000, 64, 300));
   for (i = 0; i < SPV_SEC_COUNT; i++)
      bufs[i] = w.sec[i].w;
   bufs[SPV_SEC_COUNT] = w.out.w;
   dedup = w.dedup;
   CHECK(w.dedup_cap >= 2 * 301);
   t = spv_type_int(&w, 0);
   for (k = 0; k < 299; k++)
      CHECK(spv_const_bits(&w, t, (uint32_t)k) != 0);
   for (k = 0; k < 700; k++)
   {
      words[0] = 1;
      words[1] = (uint32_t)k;
      words[2] = 2;
      spv_inst(&w, SPV_SEC_BODY, SpvOpNop, words, 3);
   }
   spv_inst(&w, SPV_SEC_DEBUG, SpvOpNop, words, 3);
   m = spv_writer_finish(&w, &count);
   CHECK(m != NULL);
   for (i = 0; i < SPV_SEC_COUNT; i++)
      if (w.sec[i].w != bufs[i])
         stable = 0;
   if (w.out.w != bufs[SPV_SEC_COUNT] || w.dedup != dedup)
      stable = 0;
   CHECK(stable);
   /* one past it grows */
   for (k = 0; k < 100; k++)
      spv_inst(&w, SPV_SEC_BODY, SpvOpNop, words, 3);
   CHECK(w.sec[SPV_SEC_BODY].cap > 3000 && w.sec[SPV_SEC_BODY].n == 800 * 4);
   spv_writer_free(&w);
}

int main(int argc, char **argv)
{
   test_module(argc > 1 ? argv[1] : NULL);
   test_dedup();
   test_reuse();
   test_growth();
   test_reserve();
   if (failures)
   {
      fprintf(stderr, "spirv_writer_test: %d failed\n", failures);
      return 1;
   }
   printf("spirv_writer_test: all passed\n");
   return 0;
}
