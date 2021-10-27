//===--- Profile.h - OpenMP device profile interface -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_PROFILE_H
#define OMPTARGET_PROFILE_H

#include "Debug.h"
#include "Types.h"
#include "KernelEnvironment.h"

namespace _OMP {
namespace profile {

enum EventKind : uint32_t {
  KernelInit,
  ParallelRegion,
  SharedStackUsage,
  ThreadStateUsage,
  PrintCall,
  AssertionCall,
  UserICVUpdate,
  SequentializedParallel,
  NumThreadsClause,
  NonDefaultNumThreads,
  IdleThreads,
};

enum EventLocation : uint8_t {
  EL_ENTER,
  EL_EXIT,
  EL_SINGLETON,
};

void init(bool IsSPMD, kernel::KernelEnvironmentTy &KernelEnv);

bool isInProfileMode();
bool isInAdvisorMode();
bool isInProfileOrAdvisorMode();

/// Default implementation as a catch all which prints a warning in
/// debug mode.
template <profile::EventKind Kind> struct EventHandlerBase {
  template <typename... ArgsTy> static void enter(ArgsTy... Args) {
    WARN("Enter event for kind %u not handled (got %i arguments)\n",
         uint32_t(Kind), int32_t(sizeof...(Args)));
  }
  static void exit() { WARN("Exit event for kind %u not handled\n", Kind); }
  template <typename... ArgsTy> static void singleton(ArgsTy... Args) {
    WARN("Singleton event for kind %u not handled (got %i arguments)\n",
         uint32_t(Kind), int32_t(sizeof...(Args)));
  }
};

template <profile::EventKind Kind>
struct EventHandler : public EventHandlerBase<Kind> {};

#define EntryExitEvent(Kind, ...)                                              \
  template <> struct EventHandler<Kind> : public EventHandlerBase<Kind> {      \
    static void enter(__VA_ARGS__);                                            \
    static void exit();                                                        \
  }
#define SingletonEvent(Kind, ...)                                              \
  template <> struct EventHandler<Kind> : public EventHandlerBase<Kind> {      \
    static void singleton(__VA_ARGS__);                                        \
  }

EntryExitEvent(profile::KernelInit);
EntryExitEvent(profile::ParallelRegion, IdentTy *);
SingletonEvent(profile::SharedStackUsage, IdentTy *);
SingletonEvent(profile::ThreadStateUsage, IdentTy *);
SingletonEvent(profile::PrintCall);
SingletonEvent(profile::AssertionCall, const char*, unsigned, const char*);
SingletonEvent(profile::UserICVUpdate, IdentTy *);
SingletonEvent(profile::SequentializedParallel, IdentTy *);
SingletonEvent(profile::IdleThreads, IdentTy *);

#undef EntryExitEvent
#undef SingletonEvent

template <EventKind Kind, EventLocation Location, typename... ArgsTy>
void registerEvent(ArgsTy... Args) {
  if (!isInProfileOrAdvisorMode())
    return;
  switch (Location) {
  case EL_ENTER:
    return EventHandler<Kind>::enter(Args...);
  case EL_EXIT:
    return EventHandler<Kind>::exit();
  case EL_SINGLETON:
    return EventHandler<Kind>::singleton(Args...);
  }
}

template <EventKind Kind, typename... ArgsTy>
void singletonEvent(ArgsTy... Args) {
  return registerEvent<Kind, EL_SINGLETON, ArgsTy...>(Args...);
}

template <EventKind Kind> struct RAII {
  template <typename... ArgsTy> RAII(ArgsTy... Args) {
    if (isInProfileOrAdvisorMode())
      EventHandler<Kind>::enter(Args...);
  }
  ~RAII() {
    if (isInProfileOrAdvisorMode())
      EventHandler<Kind>::exit();
  }
};

} // namespace profile
} // namespace _OMP

#endif
