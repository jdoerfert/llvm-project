//===------- LibC.cpp - Simple implementation of libc functions --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibC.h"
#include "Debug.h"
#include "Memory.h"
#include "Synchronization.h"
#include "Utils.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t);
}

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
extern "C" int32_t vprintf(const char *, void *);
namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return vprintf(Format, Arguments);
}
} // namespace impl
#pragma omp end declare variant

// We do not have a vprintf implementation for AMD GPU yet so we use a stub.
#pragma omp begin declare variant match(device = {arch(amdgcn)})
namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return -1;
}
} // namespace impl

#pragma omp end declare variant

// Dummy variable for stdio.
int StdInDummyVar;
int StdOutDummyVar;
int StdErrDummyVar;

struct FILE;
__attribute__((used, retain, weak, visibility("protected"))) FILE *stdin =
    (FILE *)&StdInDummyVar;
__attribute__((used, retain, weak, visibility("protected"))) FILE *stdout =
    (FILE *)&StdOutDummyVar;
__attribute__((used, retain, weak, visibility("protected"))) FILE *stderr =
    (FILE *)&StdErrDummyVar;

typedef int (*__compar_fn_t)(const void *, const void *);
typedef int (*__compar_d_fn_t)(const void *, const void *, void *);

namespace {
const int32_t ToLowerMapTable[] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,   3,   4,   5,   6,
    7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
    37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
    52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  'a', 'b',
    'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 91,  92,  93,  94,  95,  96,
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 123, 124, 125, 126,
    127, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,
};
const int32_t *ToLowerPTable = ToLowerMapTable + 128;

#define X(x) (((x) / 256 | (x)*256) % 65536)

static const unsigned short BLocTable[] = {
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        X(0x200), X(0x200), X(0x200), X(0x200), X(0x200),
    X(0x200), X(0x200), X(0x200), X(0x200), X(0x320), X(0x220), X(0x220),
    X(0x220), X(0x220), X(0x200), X(0x200), X(0x200), X(0x200), X(0x200),
    X(0x200), X(0x200), X(0x200), X(0x200), X(0x200), X(0x200), X(0x200),
    X(0x200), X(0x200), X(0x200), X(0x200), X(0x200), X(0x200), X(0x160),
    X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0),
    X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0),
    X(0x4c0), X(0x8d8), X(0x8d8), X(0x8d8), X(0x8d8), X(0x8d8), X(0x8d8),
    X(0x8d8), X(0x8d8), X(0x8d8), X(0x8d8), X(0x4c0), X(0x4c0), X(0x4c0),
    X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x8d5), X(0x8d5), X(0x8d5),
    X(0x8d5), X(0x8d5), X(0x8d5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5),
    X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5),
    X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5), X(0x8c5),
    X(0x8c5), X(0x8c5), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0), X(0x4c0),
    X(0x4c0), X(0x8d6), X(0x8d6), X(0x8d6), X(0x8d6), X(0x8d6), X(0x8d6),
    X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6),
    X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6),
    X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x8c6), X(0x4c0),
    X(0x4c0), X(0x4c0), X(0x4c0), X(0x200), 0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,        0,
    0,        0,        0,        0,        0,        0,
};

#undef X

static const unsigned short *BLocTablePTable = BLocTable + 128;

int ErrNo __attribute__((used, retain));

static unsigned long int RandomNext = 1;

#define SWAP(a, b, size)                                                       \
  do {                                                                         \
    size_t __size = (size);                                                    \
    char *__a = (a), *__b = (b);                                               \
    do {                                                                       \
      char __tmp = *__a;                                                       \
      *__a++ = *__b;                                                           \
      *__b++ = __tmp;                                                          \
    } while (--__size > 0);                                                    \
  } while (0)

/* Discontinue quicksort algorithm when partition gets below this size.
   This particular magic number was chosen to work best on a Sun 4/260. */
#define MAX_THRESH 4

/* Stack node declarations used to store unfulfilled partition obligations. */
typedef struct {
  char *lo;
  char *hi;
} stack_node;

/* The next 4 #defines implement a very fast in-line stack abstraction. */
/* The stack needs log (total_elements) entries (we could even subtract
   log(MAX_THRESH)).  Since total_elements has type size_t, we get as
   upper bound for log (total_elements):
   bits per byte (CHAR_BIT) * sizeof(size_t).  */
#define CHAR_BIT 8
#define STACK_SIZE (CHAR_BIT * sizeof(size_t))
#define PUSH(low, high) ((void)((top->lo = (low)), (top->hi = (high)), ++top))
#define POP(low, high) ((void)(--top, (low = top->lo), (high = top->hi)))
#define STACK_NOT_EMPTY (stack < top)

/* Order size using quicksort.  This implementation incorporates
   four optimizations discussed in Sedgewick:
   1. Non-recursive, using an explicit stack of pointer that store the
      next array partition to sort.  To save time, this maximum amount
      of space required to store an array of SIZE_MAX is allocated on the
      stack.  Assuming a 32-bit (64 bit) integer for size_t, this needs
      only 32 * sizeof(stack_node) == 256 bytes (for 64 bit: 1024 bytes).
      Pretty cheap, actually.
   2. Chose the pivot element using a median-of-three decision tree.
      This reduces the probability of selecting a bad pivot value and
      eliminates certain extraneous comparisons.
   3. Only quicksorts TOTAL_ELEMS / MAX_THRESH partitions, leaving
      insertion sort to order the MAX_THRESH items within each partition.
      This is a big win, since insertion sort is faster for small, mostly
      sorted array segments.
   4. The larger of the two sub-partitions is always pushed onto the
      stack first, with the algorithm then concentrating on the
      smaller partition.  This *guarantees* no more than log (total_elems)
      stack size is needed (actually O(1) in this case)!  */

void _quicksort(void *const pbase, size_t total_elems, size_t size,
                __compar_d_fn_t cmp, void *arg) {
  char *base_ptr = (char *)pbase;

  const size_t max_thresh = MAX_THRESH * size;

  if (total_elems == 0)
    /* Avoid lossage with unsigned arithmetic below.  */
    return;

  if (total_elems > MAX_THRESH) {
    char *lo = base_ptr;
    char *hi = &lo[size * (total_elems - 1)];
    stack_node stack[STACK_SIZE];
    stack_node *top = stack;

    PUSH(nullptr, nullptr);

    while (STACK_NOT_EMPTY) {
      char *left_ptr;
      char *right_ptr;

      /* Select median value from among LO, MID, and HI. Rearrange
         LO and HI so the three values are sorted. This lowers the
         probability of picking a pathological pivot value and
         skips a comparison for both the LEFT_PTR and RIGHT_PTR in
         the while loops. */

      char *mid = lo + size * ((hi - lo) / size >> 1);

      if ((*cmp)((void *)mid, (void *)lo, arg) < 0)
        SWAP(mid, lo, size);
      if ((*cmp)((void *)hi, (void *)mid, arg) < 0)
        SWAP(mid, hi, size);
      else
        goto jump_over;
      if ((*cmp)((void *)mid, (void *)lo, arg) < 0)
        SWAP(mid, lo, size);
    jump_over:;

      left_ptr = lo + size;
      right_ptr = hi - size;

      /* Here's the famous ``collapse the walls'' section of quicksort.
         Gotta like those tight inner loops!  They are the main reason
         that this algorithm runs much faster than others. */
      do {
        while ((*cmp)((void *)left_ptr, (void *)mid, arg) < 0)
          left_ptr += size;

        while ((*cmp)((void *)mid, (void *)right_ptr, arg) < 0)
          right_ptr -= size;

        if (left_ptr < right_ptr) {
          SWAP(left_ptr, right_ptr, size);
          if (mid == left_ptr)
            mid = right_ptr;
          else if (mid == right_ptr)
            mid = left_ptr;
          left_ptr += size;
          right_ptr -= size;
        } else if (left_ptr == right_ptr) {
          left_ptr += size;
          right_ptr -= size;
          break;
        }
      } while (left_ptr <= right_ptr);

      /* Set up pointers for next iteration.  First determine whether
         left and right partitions are below the threshold size.  If so,
         ignore one or both.  Otherwise, push the larger partition's
         bounds on the stack and continue sorting the smaller one. */

      if ((size_t)(right_ptr - lo) <= max_thresh) {
        if ((size_t)(hi - left_ptr) <= max_thresh)
          /* Ignore both small partitions. */
          POP(lo, hi);
        else
          /* Ignore small left partition. */
          lo = left_ptr;
      } else if ((size_t)(hi - left_ptr) <= max_thresh)
        /* Ignore small right partition. */
        hi = right_ptr;
      else if ((right_ptr - lo) > (hi - left_ptr)) {
        /* Push larger left partition indices. */
        PUSH(lo, right_ptr);
        lo = left_ptr;
      } else {
        /* Push larger right partition indices. */
        PUSH(left_ptr, hi);
        hi = right_ptr;
      }
    }
  }

  /* Once the BASE_PTR array is partially sorted by quicksort the rest
     is completely sorted using insertion sort, since this is efficient
     for partitions below MAX_THRESH size. BASE_PTR points to the beginning
     of the array to sort, and END_PTR points at the very last element in
     the array (*not* one beyond it!). */

#define min(x, y) ((x) < (y) ? (x) : (y))

  {
    char *const end_ptr = &base_ptr[size * (total_elems - 1)];
    char *tmp_ptr = base_ptr;
    char *thresh = min(end_ptr, base_ptr + max_thresh);
    char *run_ptr;

    /* Find smallest element in first threshold and place it at the
       array's beginning.  This is the smallest array element,
       and the operation speeds up insertion sort's inner loop. */

    for (run_ptr = tmp_ptr + size; run_ptr <= thresh; run_ptr += size)
      if ((*cmp)((void *)run_ptr, (void *)tmp_ptr, arg) < 0)
        tmp_ptr = run_ptr;

    if (tmp_ptr != base_ptr)
      SWAP(tmp_ptr, base_ptr, size);

    /* Insertion sort, running from left-hand-side up to right-hand-side.  */

    run_ptr = base_ptr + size;
    while ((run_ptr += size) <= end_ptr) {
      tmp_ptr = run_ptr - size;
      while ((*cmp)((void *)run_ptr, (void *)tmp_ptr, arg) < 0)
        tmp_ptr -= size;

      tmp_ptr += size;
      if (tmp_ptr != run_ptr) {
        char *trav;

        trav = run_ptr + size;
        while (--trav >= run_ptr) {
          char c = *trav;
          char *hi, *lo;

          for (hi = lo = trav; (lo -= size) >= tmp_ptr; hi = lo)
            *hi = *lo;
          *hi = c;
        }
      }
    }
  }
}

} // namespace

extern "C" {

int memcmp(const void *lhs, const void *rhs, size_t count) {
  auto *L = reinterpret_cast<const unsigned char *>(lhs);
  auto *R = reinterpret_cast<const unsigned char *>(rhs);

  for (size_t I = 0; I < count; ++I)
    if (L[I] != R[I])
      return (int)L[I] - (int)R[I];

  return 0;
}

/// printf() calls are rewritten by CGGPUBuiltin to __llvm_omp_vprintf
int32_t __llvm_omp_vprintf(const char *Format, void *Arguments, uint32_t Size) {
  return impl::omp_vprintf(Format, Arguments, Size);
}

// -----------------------------------------------------------------------------

#ifndef ULONG_MAX
#define ULONG_MAX ((unsigned long)(~0L)) /* 0xFFFFFFFF */
#endif

#ifndef LONG_MAX
#define LONG_MAX ((long)(ULONG_MAX >> 1)) /* 0x7FFFFFFF */
#endif

#ifndef LONG_MIN
#define LONG_MIN ((long)(~LONG_MAX)) /* 0x80000000 */
#endif

#ifndef ISDIGIT
#define ISDIGIT(C) ((C) >= '0' && (C) <= '9')
#endif

#ifndef ISSPACE
#define ISSPACE(C) ((C) == ' ')
#endif

#ifndef ISALPHA
#define ISALPHA(C) (((C) >= 'a' && (C) <= 'z') || ((C) >= 'A' && (C) <= 'Z'))
#endif

#ifndef ISUPPER
#define ISUPPER(C) ((C) >= 'A' && (C) <= 'Z')
#endif

long strtol(const char *nptr, char **endptr, int base) {
  const char *s = nptr;
  unsigned long acc;
  int c;
  unsigned long cutoff;
  int neg = 0, any, cutlim;

  /*
   * Skip white space and pick up leading +/- sign if any.
   * If base is 0, allow 0x for hex and 0 for octal, else
   * assume decimal; if base is already 16, allow 0x.
   */
  do {
    c = *s++;
  } while (ISSPACE(c));
  if (c == '-') {
    neg = 1;
    c = *s++;
  } else if (c == '+')
    c = *s++;
  if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
    c = s[1];
    s += 2;
    base = 16;
  }
  if (base == 0)
    base = c == '0' ? 8 : 10;

  /*
   * Compute the cutoff value between legal numbers and illegal
   * numbers.  That is the largest legal value, divided by the
   * base.  An input number that is greater than this value, if
   * followed by a legal input character, is too big.  One that
   * is equal to this value may be valid or not; the limit
   * between valid and invalid numbers is then based on the last
   * digit.  For instance, if the range for longs is
   * [-2147483648..2147483647] and the input base is 10,
   * cutoff will be set to 214748364 and cutlim to either
   * 7 (neg==0) or 8 (neg==1), meaning that if we have accumulated
   * a value > 214748364, or equal but the next digit is > 7 (or 8),
   * the number is too big, and we will return a range error.
   *
   * Set any if any `digits' consumed; make it negative to indicate
   * overflow.
   */
  cutoff = neg ? -(unsigned long)LONG_MIN : LONG_MAX;
  cutlim = cutoff % (unsigned long)base;
  cutoff /= (unsigned long)base;
  for (acc = 0, any = 0;; c = *s++) {
    if (ISDIGIT(c))
      c -= '0';
    else if (ISALPHA(c))
      c -= ISUPPER(c) ? 'A' - 10 : 'a' - 10;
    else
      break;
    if (c >= base)
      break;
    if (any < 0 || acc > cutoff || (acc == cutoff && c > cutlim))
      any = -1;
    else {
      any = 1;
      acc *= base;
      acc += c;
    }
  }
  if (any < 0) {
    acc = neg ? LONG_MIN : LONG_MAX;
    errno = ERANGE;
  } else if (neg)
    acc = -acc;
  if (endptr != 0)
    *endptr = const_cast<char *>(any ? s - 1 : nptr);
  return (acc);
}

int strcmp(const char *lhs, const char *rhs) {
  while (*lhs != '\0' && *rhs != '\0') {
    if (*lhs == *rhs) {
      ++lhs;
      ++rhs;
    }
    return *lhs - *rhs;
  }
  if (*lhs != '\0')
    return 1;

  return -1;
}

void *calloc(size_t num, size_t size) {
  size_t bits = num * size;
  char *p = (char *)malloc(bits);
  if (!p)
    return p;
  char *q = (char *)p;
  while (q - p < bits) {
    *(int *)q = 0;
    q += sizeof(int);
  }
  while (q - p < bits) {
    *q = 0;
    q++;
  }
  return p;
}

void exit(int exit_code) { asm volatile("exit;"); }

size_t strlen(const char *str) {
  size_t r = 0;
  while (*str == ' ')
    ++str;

  while (*str != '\0') {
    ++r;
    ++str;
  }

  return r;
}

char *strcpy(char *dest, const char *src) {
  char *pd = dest;
  const char *ps = src;

  while (*ps != '\0')
    *(pd++) = *(ps++);

  *pd = '\0';

  return dest;
}

int *__errno_location() { return &ErrNo; }

char *strcat(char *dest, const char *src) {
  char *pd = dest;
  const char *ps = src;

  while (*pd != '\0')
    ++pd;

  while (*ps != '\0')
    *(pd++) = *(ps++);

  *pd = '\0';

  return dest;
}

void perror(const char *s) { printf("%s", s); }

int strncmp(const char *lhs, const char *rhs, size_t count) {
  size_t c = 0;
  while (*lhs != '\0' && *rhs != '\0' && c < count) {
    if (*lhs == *rhs) {
      ++lhs;
      ++rhs;
      ++c;
    } else
      return *lhs - *rhs;
  }

  return 0;
}

char *strncpy(char *dest, const char *src, size_t count) {
  char *pd = dest;
  const char *ps = src;
  size_t c = 0;

  while (*ps != '\0' && c < count) {
    *(pd++) = *(ps++);
    ++c;
  }

  if (c < count)
    *pd = '\0';

  return dest;
}

char *strchr(const char *s, int c) {
  do {
    if (*s == c)
      return const_cast<char *>(s);
  } while (*s++);
  return nullptr;
}

char *strtok(char *str, const char *delim) {
  static char *s = nullptr;
  char *tok;
  if (str == nullptr) {
    if (s == nullptr)
      return nullptr;
  } else
    s = str;
  for (size_t i; (*s != '\0'); s++) {
    for (i = 0; (delim[i] != '\0') && (*s != delim[i]); i++)
      ;
    if (delim[i] == '\0')
      break;
  }
  if (*s == '\0')
    return s = nullptr;
  tok = s++;
  for (size_t i; (*s != '\0'); s++) {
    for (i = 0; (delim[i] != '\0') && (*s != delim[i]); i++)
      ;
    if (delim[i] != '\0')
      break;
  }
  if (*s != '\0') {
    *s = '\0';
    s++;
  }
  return tok;
}

void srand(unsigned seed) { RandomNext = seed; }

int rand() {
  RandomNext = RandomNext * 1103515245 + 12345;
  return static_cast<unsigned int>((RandomNext / 65536)) % 32768;
}

int abs(int n) { return n > 0 ? n : -n; }

void *memcpy(void *dest, const void *src, size_t count) {
  __builtin_memcpy(dest, src, count);
  return dest;
}

unsigned long strtoul(const char *str, char **str_end, int base) {
  unsigned long res = 0;
  while (*str != '\0') {
    if (*str == ' ') {
      ++str;
      continue;
    }
    if (*str >= '0' && *str <= '9') {
      res = res * 10 + *str - '0';
      ++str;
      continue;
    }
    break;
  }

  if (*str_end)
    *str_end = const_cast<char *>(str);
  return res;
}

double atof(const char *s) { return strtod(s, nullptr); }

int atoi(const char *str) { return (int)strtol(str, nullptr, 10); }

double strtod(const char *str, char **ptr) {
  char *p;

  if (ptr == (char **)0)
    return atof(str);

  p = const_cast<char *>(str);

  while (ISSPACE(*p))
    ++p;

  if (*p == '+' || *p == '-')
    ++p;

  /* INF or INFINITY.  */
  if ((p[0] == 'i' || p[0] == 'I') && (p[1] == 'n' || p[1] == 'N') &&
      (p[2] == 'f' || p[2] == 'F')) {
    if ((p[3] == 'i' || p[3] == 'I') && (p[4] == 'n' || p[4] == 'N') &&
        (p[5] == 'i' || p[5] == 'I') && (p[6] == 't' || p[6] == 'T') &&
        (p[7] == 'y' || p[7] == 'Y')) {
      *ptr = p + 8;
      return atof(str);
    } else {
      *ptr = p + 3;
      return atof(str);
    }
  }

  /* NAN or NAN(foo).  */
  if ((p[0] == 'n' || p[0] == 'N') && (p[1] == 'a' || p[1] == 'A') &&
      (p[2] == 'n' || p[2] == 'N')) {
    p += 3;
    if (*p == '(') {
      ++p;
      while (*p != '\0' && *p != ')')
        ++p;
      if (*p == ')')
        ++p;
    }
    *ptr = p;
    return atof(str);
  }

  /* digits, with 0 or 1 periods in it.  */
  if (ISDIGIT(*p) || *p == '.') {
    int got_dot = 0;
    while (ISDIGIT(*p) || (!got_dot && *p == '.')) {
      if (*p == '.')
        got_dot = 1;
      ++p;
    }

    /* Exponent.  */
    if (*p == 'e' || *p == 'E') {
      int i;
      i = 1;
      if (p[i] == '+' || p[i] == '-')
        ++i;
      if (ISDIGIT(p[i])) {
        while (ISDIGIT(p[i]))
          ++i;
        *ptr = p + i;
        return atof(str);
      }
    }
    *ptr = p;
    return atof(str);
  }
  /* Didn't find any digits.  Doesn't look like a number.  */
  *ptr = const_cast<char *>(str);
  return 0.0;
}

void *memset(void *dest, int ch, size_t count) {
  auto *P = reinterpret_cast<unsigned char *>(dest);
  for (size_t I = 0; I < count; ++I, ++P)
    *P = (unsigned char)ch;
  return dest;
}

int tolower(int ch) {
  if (ch >= 'A' && ch <= 'Z')
    return ch + 32;
  return ch;
}

char *strstr(const char *s1, const char *s2) {
  const size_t len = strlen(s2);
  while (*s1) {
    if (!memcmp(s1, s2, len))
      return const_cast<char *>(s1);
    ++s1;
  }
  return (0);
}

char *__xpg_basename(const char *path) { return const_cast<char *>(path); }

const unsigned short **__ctype_b_loc(void) { return &BLocTablePTable; }

const int32_t **__ctype_tolower_loc(void) { return &ToLowerPTable; }

void qsort(void *b, size_t n, size_t s, __compar_fn_t cmp) {
  return _quicksort(b, n, s, (__compar_d_fn_t)cmp, nullptr);
}

int strcasecmp(const char *s1, const char *s2) {
  const unsigned char *p1 = (const unsigned char *)s1;
  const unsigned char *p2 = (const unsigned char *)s2;
  int result;

  if (p1 == p2)
    return 0;

  while ((result = tolower(*p1) - tolower(*p2++)) == 0)
    if (*p1++ == '\0')
      break;

  return result;
}
}

#pragma omp end declare target
