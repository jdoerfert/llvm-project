//===- UnrollModelFeatureMaps.h - common model runner defs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_UNROLLMODELFEATUREMAPS_H
#define LLVM_ANALYSIS_UNROLLMODELFEATUREMAPS_H

#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/TensorSpec.h"

#include <array>
#include <cstddef>
#include <vector>

namespace llvm {
namespace mlgo {

template <typename T, size_t N>
constexpr std::array<T, N> makeArray(const T (&A)[N]) {
  std::array<T, N> Arr = {};
  for (size_t I = 0; I < N; ++I)
    Arr[I] = A[I];
  return Arr;
}

// ---------------------- Exact Binning Tensors ---------------------- //

/// The tensor holds the following constant sizes, (sizes other less than 64),
/// (sizes other greater than 64).
/// Mainly guided by x86_64 needs as AVX-512 allows up to 64 byte accesses.
/// Multiples of 8 are also included? Not sure if we need them.
constexpr auto AccessSizesBins = makeArray({
    1,
    2,
    4,
    8,
    16,
    32,
    64,
});
constexpr unsigned AccessSizesBinsNum = AccessSizesBins.size() + 2;

constexpr auto AccessAlignmentsBins = makeArray({
    1,
    2,
    4,
    8,
    16,
});
constexpr unsigned AccessAlignmentsBinsNum = AccessAlignmentsBins.size() + 2;

// TODO
constexpr auto SpacialReuseDistanceBins = makeArray({
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    16,
    20,
    24,
    32,
    40,
    48,
    56,
    64,
});
constexpr unsigned SpacialReuseDistanceBinsNum =
    SpacialReuseDistanceBins.size() + 2;

// TODO
constexpr auto PtrStridesBins = makeArray({
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    16,
    20,
    24,
    32,
    40,
    48,
    56,
    64,
//     1,
//     2,
//     4,
//     8,
//     16,
//     24,
//     32,
//     40,
//     48,
//     56,
//     64,
});
constexpr unsigned PtrStridesBinsNum = PtrStridesBins.size() + 2;

// ---------------------- Interval Binning Tensors ---------------------- //

constexpr auto LoopBlocksizesIntervals = makeArray({
    2,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    128,
    256,
});
constexpr unsigned LoopBlocksizesBinsNum = LoopBlocksizesIntervals.size() + 1;

constexpr auto InstructionCostsRecipThroughputIntervals = makeArray({
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    32,
});
constexpr unsigned InstructionCostsRecipThroughputBinsNum =
    InstructionCostsRecipThroughputIntervals.size() + 1;

constexpr auto InstructionCostsLatencyIntervals = makeArray({
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    24,
    32,
});
constexpr unsigned InstructionCostsLatencyBinsNum =
    InstructionCostsLatencyIntervals.size() + 1;

constexpr auto InstructionCostsCodeSizeIntervals = makeArray({
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
});
constexpr unsigned InstructionCostsCodeSizeBinsNum =
    InstructionCostsCodeSizeIntervals.size() + 1;

// ---------------------- Enum Binning Tensors ---------------------- //

constexpr unsigned RecurranceInfosBinsNum =
    static_cast<unsigned>(llvm::RecurKind::NumRecurKinds);
constexpr unsigned DependenceInfosBinsNum =
    static_cast<unsigned>(llvm::MemoryDepChecker::Dependence::NumDepTypes);

#define LOOP_UNROLL_FEATURE_ITERATOR(M)                                        \
  M(int64_t, {1}, loop_size, "size of loop")                                   \
  M(int64_t, {1}, trip_count, "static trip count of loop")

// clang-format off
enum class UnrollFeatureIndex : size_t {

#define POPULATE_INDICES(DTYPE, SHAPE, NAME, DOC) NAME,
  LOOP_UNROLL_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES

#define MAP_UINT_UINT_PROPERTY(NAME, DEFAULT) NAME,
#define MAP_UINT64_UINT64_PROPERTY(NAME, DEFAULT) NAME,
#define BOOL_PROPERTY(NAME, DEFAULT) NAME,
#define UINT64_PROPERTY(NAME, DEFAULT) NAME,
#define STRING_PROPERTY(NAME, DEFAULT)
#define APINT_PROPERTY(NAME, DEFAULT) NAME,
#define INSTCOST_PROPERTY(NAME, DEFAULT) NAME,
#include "llvm/Analysis/LoopProperties.def"
#undef INSTCOST_PROPERTY
#undef BOOL_PROPERTY
#undef UINT64_PROPERTY
#undef STRING_PROPERTY
#undef APINT_PROPERTY
#undef MAP_UINT_UINT_PROPERTY
#undef MAP_UINT64_UINT64_PROPERTY

  NumberOfFeatures
};
// clang-format on

// These need to be kept in sync with the ones in unrolling_runner.py
static constexpr unsigned MaxUnrollFactor = 32;
static constexpr unsigned UnrollFactorOffset = 2;
// + 1 because inclusive
static constexpr unsigned UnrollModelOutputLength =
    1 + MaxUnrollFactor - UnrollFactorOffset;

struct UnrollDecisionTy {
  float Out[UnrollModelOutputLength];
};
static_assert(offsetof(UnrollDecisionTy, Out) == 0);

extern const std::vector<TensorSpec> UnrollFeatureMap;

extern const char *const UnrollDecisionName;
extern const TensorSpec UnrollDecisionSpec;

} // namespace mlgo
} // namespace llvm

#endif //
