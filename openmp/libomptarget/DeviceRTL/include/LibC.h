//===--------- LibC.h - Simple implementation of libc functions --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_LIBC_H
#define OMPTARGET_LIBC_H

#include "Types.h"

struct FILE;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

#ifndef _ASM_GENERIC_ERRNO_BASE_H
#define _ASM_GENERIC_ERRNO_BASE_H

#define EPERM 1    /* Operation not permitted */
#define ENOENT 2   /* No such file or directory */
#define ESRCH 3    /* No such process */
#define EINTR 4    /* Interrupted system call */
#define EIO 5      /* I/O error */
#define ENXIO 6    /* No such device or address */
#define E2BIG 7    /* Argument list too long */
#define ENOEXEC 8  /* Exec format error */
#define EBADF 9    /* Bad file number */
#define ECHILD 10  /* No child processes */
#define EAGAIN 11  /* Try again */
#define ENOMEM 12  /* Out of memory */
#define EACCES 13  /* Permission denied */
#define EFAULT 14  /* Bad address */
#define ENOTBLK 15 /* Block device required */
#define EBUSY 16   /* Device or resource busy */
#define EEXIST 17  /* File exists */
#define EXDEV 18   /* Cross-device link */
#define ENODEV 19  /* No such device */
#define ENOTDIR 20 /* Not a directory */
#define EISDIR 21  /* Is a directory */
#define EINVAL 22  /* Invalid argument */
#define ENFILE 23  /* File table overflow */
#define EMFILE 24  /* Too many open files */
#define ENOTTY 25  /* Not a typewriter */
#define ETXTBSY 26 /* Text file busy */
#define EFBIG 27   /* File too large */
#define ENOSPC 28  /* No space left on device */
#define ESPIPE 29  /* Illegal seek */
#define EROFS 30   /* Read-only file system */
#define EMLINK 31  /* Too many links */
#define EPIPE 32   /* Broken pipe */
#define EDOM 33    /* Math argument out of domain of func */
#define ERANGE 34  /* Math result not representable */

#endif

#define errno (*__errno_location())

extern "C" {

int memcmp(const void *lhs, const void *rhs, size_t count);

int printf(const char *format, ...);

long strtol(const char *str, char **str_end, int base);

int strcmp(const char *lhs, const char *rhs);

void *calloc(size_t num, size_t size);

int strcasecmp(const char *string1, const char *string2);

void exit(int exit_code);

size_t strlen(const char *str);

int atoi(const char *str);

char *strcpy(char *dest, const char *src);

int stat(const char *path, struct stat *buf);

int *__errno_location();

char *strcat(char *dest, const char *src);

void perror(const char *s);

int strncmp(const char *lhs, const char *rhs, size_t count);

char *strncpy(char *dest, const char *src, size_t count);

char *strchr(const char *str, int ch);

char *strtok(char *str, const char *delim);

const unsigned short **__ctype_b_loc(void);

void *realloc(void *ptr, size_t new_size);

void qsort(void *const pbase, size_t total_elems, size_t size,
           int (*comp)(const void *, const void *));

int gettimeofday(struct timeval *tv, struct timezone *tz);

char *__xpg_basename(const char *path);

void srand(unsigned seed);

int rand();

int abs(int n);

void *memcpy(void *dest, const void *src, size_t count);

double atof(const char *str);

double strtod(const char *str, char **ptr);

long strtol(const char *nptr, char **endptr, int base);
}

#endif
