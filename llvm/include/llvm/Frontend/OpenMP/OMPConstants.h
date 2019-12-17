//===- OMPConstants.h - OpenMP related constants and helpers ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines constans and helpers used when dealing with OpenMP.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_CONSTANTS_H
#define LLVM_OPENMP_CONSTANTS_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Type;
class Module;
class StructType;
class PointerType;
class FunctionType;

namespace omp {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Encoding for a known binary value.
enum class BinraryChoice {
  OMP_FALSE,
  OMP_TRUE,
};

/// Encoding for a potentially unknown binary value. Used in runtime calls.
enum class TernaryChoice {
  OMP_UNKNOWN = -1,
  OMP_FALSE,
  OMP_TRUE,
};

/// IDs for all OpenMP directives.
enum class Directive {
#define OMP_DIRECTIVE(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

/// Make the enum values available in the llvm::omp namespace. This allows us to
/// write something like OMPD_parallel if we have a `using namespace omp`. At
/// the same time we do not loose the strong type guarantees of the enum class,
/// that is we cannot pass an unsigned as Directive without an explicit cast.
#define OMP_DIRECTIVE(Enum, ...) constexpr auto Enum = omp::Directive::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// IDs for all omp runtime library (RTL) functions.
enum class RuntimeFunction {
#define OMP_RTL(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define OMP_RTL(Enum, ...) constexpr auto Enum = omp::RuntimeFunction::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// IDs for all omp runtime library ident_t flag encodings (see
/// their defintion in openmp/runtime/src/kmp.h).
enum class IdentFlag {
#define OMP_IDENT_FLAG(Enum, Str, Value) Enum = Value,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  LLVM_MARK_AS_BITMASK_ENUM(0x7FFFFFFF)
};

#define OMP_IDENT_FLAG(Enum, ...) constexpr auto Enum = omp::IdentFlag::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// Parse \p Str and return the directive it matches or OMPD_unknown if none.
Directive getOpenMPDirectiveKind(StringRef Str);

/// Return a textual representation of the directive \p D.
StringRef getOpenMPDirectiveName(Directive D);

/// Forward declarations for LLVM-IR types (simple, function and structure) are
/// generated below. Their names are defined and used in OpenMPKinds.def. Here
/// we provide the forward declarations, the initializeTypes function will
/// provide the values.
///
///{
namespace types {

/// Make the binary/ternary choice values available as common integers and
/// ensure their values are always in-sync.
///
///{
static constexpr int OMP_UNKNOWN = int(TernaryChoice::OMP_UNKNOWN);
static constexpr int OMP_FALSE = int(TernaryChoice::OMP_FALSE);
static constexpr int OMP_TRUE = int(TernaryChoice::OMP_TRUE);

static_assert(OMP_UNKNOWN == int(TernaryChoice::OMP_UNKNOWN),
              "OMP_UNKNOWN initialization mismatch!");
static_assert(OMP_FALSE == int(BinraryChoice::OMP_FALSE) &&
                  OMP_FALSE == int(TernaryChoice::OMP_FALSE),
              "OMP_FALSE initialization mismatch!");
static_assert(OMP_TRUE == int(BinraryChoice::OMP_TRUE) &&
                  OMP_TRUE == int(TernaryChoice::OMP_TRUE),
              "OMP_TRUE initialization mismatch!");
///}

#define OMP_TYPE(VarName, InitValue) extern Type *VarName;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  extern FunctionType *VarName;                                                \
  extern PointerType *VarName##Ptr;
#define OMP_STRUCT_TYPE(VarName, StrName, ...)                                 \
  extern StructType *VarName;                                                  \
  extern PointerType *VarName##Ptr;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// Helper to initialize all types defined in OpenMPKinds.def.
void initializeTypes(Module &M);

/// Helper to uninitialize all types defined in OpenMPKinds.def.
void uninitializeTypes();

} // namespace types
///}

} // end namespace omp

} // end namespace llvm

#endif // LLVM_OPENMP_CONSTANTS_H
