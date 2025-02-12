// LLVM Instrumentor stub runtime

#include <stdint.h>
#include <stdio.h>

void __instrumentor_pre_function(void *address, char *name) {
  printf("function pre -- address: %p, name: %s\n", address, name);
}

void __instrumentor_pre_call(void *callee, char *callee_name) {
  printf("call pre -- callee: %p, callee_name: %s\n", callee, callee_name);
}

void __instrumentor_post_call(void *callee, char *callee_name) {
  printf("call post -- callee: %p, callee_name: %s\n", callee, callee_name);
}

