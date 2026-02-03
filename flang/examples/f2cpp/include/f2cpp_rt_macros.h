//===- f2cpp_rt_macros.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
//===----------------------------------------------------------------------===//

#ifndef F2CPP_RT_MACROS_H
#define F2CPP_RT_MACROS_H

#include "f2cpp_rt_types.h"

#define INTEGER_PARAMETER const INTEGER
#define AUTO_PARAMETER const auto

#define INTEGER_DIMENSION(X) StaticArray<INTEGER, X>
#define REAL_DIMENSION(X) StaticArray<REAL, X>

#define PRINT(X, ...) flc::print(#X, __FILE__, __LINE__, __VA_ARGS__)
#define READ(X, ARG) \
  flc::read(#X, __FILE__, __LINE__, (char *)ARG, flc::arraySize(ARG))

#define DO(VAR, INIT, COND, STEP) for (VAR = INIT; VAR <= COND; VAR += STEP) {
#define DO_WHILE(COND) while (COND) {
#define END_DO }

#define IF(COND) if (COND)
#define THEN {
#define ELSE \
  } \
  else {
#define ELSE_IF(COND) \
  } \
  else if (COND)
#define END_IF }

#define CALL

#define SELECT_CASE(X) switch (X) {
#define CASE(COND) case (COND):
#define CASE_DEFAULT default:
#define END_SELECT }

#define MODULE(NAME) namespace NAME {
#define END_MODULE }
#define USE using namespace

#define WHILE(COND) while (COND) {
#define END_WHILE }

#define CYCLE continue;
#define EXIT break;

#define TRUE true
#define FALSE false
#define AND &&
#define OR ||
#define NOT !
#define EQV ==
#define NEQV !=

#define PROGRAM extern "C" void _QQmain() {
#define END_PROGRAM }

#define USE using namespace

#define TYPE(X) struct X

#endif