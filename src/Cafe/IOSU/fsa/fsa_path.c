#include "fsa_path.h"

#include <string.h>

/* A string being built in out, which fails for good once it would not fit */
typedef struct
{
   char *s;
   size_t len;
   size_t cap;
   int overflow;
} fsa_path_buf_t;

static void fsa_path_push(fsa_path_buf_t *b, char c)
{
   if (b->len + 1 >= b->cap)
   {
      b->overflow = 1;
      return;
   }
   b->s[b->len++] = c;
}

static void fsa_path_assign(fsa_path_buf_t *b, const char *s)
{
   b->len = 0;
   while (*s)
      fsa_path_push(b, *s++);
}

/* removes the last node and the slash before it */
static void fsa_path_pop_node(fsa_path_buf_t *b)
{
   while (b->len)
   {
      if (b->s[--b->len] == '/')
         break;
   }
}

int fsa_path_translate(const char *cwd, const char *input, size_t input_len,
      int end_with_slash, char *out, size_t cap)
{
   fsa_path_buf_t b;
   size_t idx = 0;
   char c;

   b.s = out;
   b.len = 0;
   b.cap = cap;
   b.overflow = 0;
   if (!cap)
      return -1;

   if (input_len == 0)
   {
      fsa_path_assign(&b, cwd);
      if (end_with_slash)
         fsa_path_push(&b, '/');
   }
   else
   {
      c = input[0];
      if (c == '/')
         fsa_path_assign(&b, "/");
      else if (c == '~')
      {
         input++;
         input_len--;
         fsa_path_assign(&b, "/");
      }
      else
         fsa_path_assign(&b, cwd);

      while (idx < input_len)
      {
         c = input[idx];
         if (c == '/')
         {
            /* leading and repeated slashes */
            idx++;
            continue;
         }
         if (c == '.')
         {
            if (input_len - idx >= 3 && input[idx + 1] == '.' && input[idx + 2] == '/')
            {
               fsa_path_pop_node(&b);
               idx += 3;
               continue;
            }
            if (input_len - idx == 2 && input[idx + 1] == '.')
            {
               fsa_path_pop_node(&b);
               idx += 2;
               continue;
            }
            if (input_len - idx >= 2 && input[idx + 1] == '/')
            {
               idx += 2;
               continue;
            }
         }
         if (b.len && b.s[b.len - 1] != '/')
            fsa_path_push(&b, '/');
         while (idx < input_len)
         {
            c = input[idx];
            if (c == '/')
               break;
            fsa_path_push(&b, c);
            idx++;
         }
      }
      if (end_with_slash && (!b.len || b.s[b.len - 1] != '/'))
         fsa_path_push(&b, '/');
   }

   if (b.overflow)
   {
      out[0] = '\0';
      return -1;
   }
   out[b.len] = '\0';
   return (int)b.len;
}
