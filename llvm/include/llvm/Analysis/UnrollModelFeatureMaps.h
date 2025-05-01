//===- UnrollModelFeatureMaps.h - common model runner defs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_UNROLLMODELFEATUREMAPS_H
#define LLVM_ANALYSIS_UNROLLMODELFEATUREMAPS_H

#include "llvm/Analysis/TensorSpec.h"

#include <cstddef>
#include <vector>

namespace llvm {
namespace mlgo {

#define LOOP_UNROLL_FEATURE_ITERATOR(M)                                        \
  M(int64_t, {1}, loop_size, "size of loop")                                   \
  M(int64_t, {1}, trip_count, "static trip count of loop")                     \
  M(int64_t, {1}, LoopBackEdgeCount, "")                                       \
  M(int64_t, {1}, HasLoopPreheader, "")                                        \
  M(int64_t, {1}, IsCountableLoop, "")                                         \
  M(int64_t, {1}, IsLoopBackEdgeCountConstant, "")                                  \
  M(int64_t, {1}, PreheaderBlocksize, "")                                      \
  M(int64_t, {1}, BasicBlockAllCount, "")                                      \
  M(int64_t, {1}, BasicBlockCount, "")                                         \
  M(int64_t, {1}, LoopDepth, "")                                               \
  M(int64_t, {1}, NumInnerLoops, "")                                           \
  M(int64_t, {1}, LoopLatchCount, "")                                          \
  M(int64_t, {1}, LoadInstCount, "")                                           \
  M(int64_t, {1}, LoadedBytes, "")                                             \
  M(int64_t, {1}, StoreInstCount, "")                                          \
  M(int64_t, {1}, StoredBytes, "")                                             \
  M(int64_t, {1}, AtomicCount, "")                                             \
  M(int64_t, {1}, FloatArithCount, "")                                         \
  M(int64_t, {1}, IntArithCount, "")                                           \
  M(int64_t, {1}, FloatDivRemCount, "")                                        \
  M(int64_t, {1}, IntDivRemCount, "")                                          \
  M(int64_t, {1}, LogicalInstCount, "")                                        \
  M(int64_t, {1}, ExpensiveCastInstCount, "")                                  \
  M(int64_t, {1}, FreeCastInstCount, "")                                       \
  M(int64_t, {1}, AlmostFreeCastInstCount, "")                                 \
  M(int64_t, {1}, FloatCmpCount, "")                                           \
  M(int64_t, {1}, IntCmpCount, "")                                             \
  M(int64_t, {1}, CondBrCount, "")                                             \
  M(int64_t, {1}, InstCount, "")                                               \
  M(int64_t, {1}, VectorInstCount, "")                                         \
  M(int64_t, {1}, DirectCallDefCount, "")                                      \
  M(int64_t, {1}, DirectCallDeclCount, "")                                     \
  M(int64_t, {1}, IndirectCall, "")                                            \
  M(int64_t, {1}, IntrinsicCount, "")

// clang-format off
enum class UnrollFeatureIndex : size_t {
#define POPULATE_INDICES(DTYPE, SHAPE, NAME, DOC) NAME,
  LOOP_UNROLL_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES

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
