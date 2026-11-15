#ifndef IOSU_FSA_PATH_H
#define IOSU_FSA_PATH_H

/* A path a title hands FSA, made absolute against the client's working
 * directory and tidied - repeated slashes, "." and ".." - into a buffer the
 * caller owns, so that opening a file allocates nothing for its path. C89.
 *
 * As FSA always has: an empty path is the working directory; one starting
 * with '/' is absolute, with '~' is absolute too; anything else is under
 * the working directory. ".." goes up a node (not past the root), "./" is
 * nothing. end_with_slash leaves a '/' at the end (a new working
 * directory). */

#include <stddef.h>

#include <retro_common_api.h>

RETRO_BEGIN_DECLS

/* The longest a translated path can be: a working directory and a path a
 * title sent, each as long as FSA takes (FSA_PATH_SIZE_MAX, 0x280), with a
 * slash between and the terminator */
#define FSA_TRANSLATED_PATH_MAX 1300

/* Returns the length written, without the terminator out always gets, or
 * -1 when it does not fit in cap (out is then empty) */
int fsa_path_translate(const char *cwd, const char *input, size_t input_len,
      int end_with_slash, char *out, size_t cap);

RETRO_END_DECLS

#endif
