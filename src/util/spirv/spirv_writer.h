#ifndef SPIRV_WRITER_H
#define SPIRV_WRITER_H

/* A SPIR-V module written word by word, in its sections' order whatever
 * order it is built in: capabilities, extensions, imports, the memory
 * model, entry points, execution modes, names, decorations, then types,
 * constants and global variables, functions written whole, then a
 * function written in parts: its start, its local variables and its body,
 * each as it comes. A type or constant asked
 * for twice is the same id (the module may declare one only once). The
 * words live in buffers the writer keeps: reset for the next module, they
 * are reused, so a thread writing one shader after another allocates only
 * while a module is bigger than any before it. C89. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* the sections, in the order the module has them */
enum
{
   SPV_SEC_CAPABILITY = 0,
   SPV_SEC_EXTENSION,
   SPV_SEC_IMPORT,
   SPV_SEC_MEMORY_MODEL,
   SPV_SEC_ENTRY_POINT,
   SPV_SEC_EXEC_MODE,
   SPV_SEC_DEBUG,
   SPV_SEC_DECORATION,
   SPV_SEC_GLOBAL,   /* types, constants, global variables */
   SPV_SEC_HELPERS,  /* functions written whole, before the entry point */
   SPV_SEC_FUNCTION, /* OpFunction and its first OpLabel */
   SPV_SEC_LOCALS,   /* its variables, which open its first block */
   SPV_SEC_BODY,     /* the rest, to OpFunctionEnd */
   SPV_SEC_COUNT
};

typedef struct spv_words
{
   uint32_t *w;
   size_t n;
   size_t cap;
} spv_words_t;

typedef struct spv_writer
{
   spv_words_t sec[SPV_SEC_COUNT];
   spv_words_t out;
   /* the global section's instructions by their words (result id aside):
    * an open-addressed table of word offsets, 0 empty */
   uint32_t *dedup;
   size_t dedup_cap;
   size_t dedup_used;
   uint32_t bound;  /* the next id */
   int failed;      /* out of memory: every later call does nothing */
} spv_writer_t;

void spv_writer_init(spv_writer_t *w);
/* empty for the next module; the buffers stay */
void spv_writer_reset(spv_writer_t *w);
void spv_writer_free(spv_writer_t *w);

uint32_t spv_id(spv_writer_t *w);

/* An instruction appended to a section: opcode and the words after it,
 * its word count put in */
void spv_inst(spv_writer_t *w, unsigned sec, uint32_t opcode, const uint32_t *operands, size_t n);
/* as spv_inst, then a literal string (nul terminated, padded to a word) */
void spv_inst_str(spv_writer_t *w, unsigned sec, uint32_t opcode, const uint32_t *operands, size_t n,
      const char *str, const uint32_t *after, size_t n_after);

/* A type or constant in the global section: operands[0] is where its
 * result id goes (the type, for a constant, before it). The id of the
 * one already declared with the same words, or a new one. */
uint32_t spv_global(spv_writer_t *w, uint32_t opcode, const uint32_t *operands, size_t n, size_t result_at);

/* the common ones */
uint32_t spv_type_void(spv_writer_t *w);
uint32_t spv_type_bool(spv_writer_t *w);
uint32_t spv_type_int(spv_writer_t *w, int is_signed);
uint32_t spv_type_float(spv_writer_t *w);
uint32_t spv_type_vector(spv_writer_t *w, uint32_t component, uint32_t count);
uint32_t spv_type_array(spv_writer_t *w, uint32_t element, uint32_t length_const);
uint32_t spv_type_pointer(spv_writer_t *w, uint32_t storage_class, uint32_t type);
uint32_t spv_type_function(spv_writer_t *w, uint32_t ret, const uint32_t *params, size_t n);
/* not deduplicated: a struct is its own type, decorated as it is */
uint32_t spv_type_struct(spv_writer_t *w, const uint32_t *members, size_t n);

uint32_t spv_const_bits(spv_writer_t *w, uint32_t type, uint32_t bits);
uint32_t spv_const_bool(spv_writer_t *w, int value);
uint32_t spv_const_composite(spv_writer_t *w, uint32_t type, const uint32_t *parts, size_t n);
uint32_t spv_const_null(spv_writer_t *w, uint32_t type);

/* a global variable (OpVariable in the global section) */
uint32_t spv_global_variable(spv_writer_t *w, uint32_t pointer_type, uint32_t storage_class);

/* the module: header and sections, valid until the next call on w; NULL
 * (and *count 0) when the writer ran out of memory */
const uint32_t *spv_writer_finish(spv_writer_t *w, size_t *count);

#ifdef __cplusplus
}
#endif

#endif
