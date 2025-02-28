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
  M(int64_t, {1}, is_innermost_loop, "whether the loop is the innermost loop") \
  M(int64_t, {1}, preheader_blocksize, "preheader blocksize (by instruction)") \
  M(int64_t, {1}, bb_count, "number of basic blocks (ignoring subloops)")      \
  M(int64_t, {1}, num_of_loop_latch, "number of loop latches")                 \
  M(int64_t, {1}, load_inst_count, "load instruction count")                   \
  M(int64_t, {1}, store_inst_count, "store instruction count")                 \
  M(int64_t, {1}, logical_inst_count, "logical instruction count")             \
  M(int64_t, {1}, cast_inst_count, "cast instruction count")

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
