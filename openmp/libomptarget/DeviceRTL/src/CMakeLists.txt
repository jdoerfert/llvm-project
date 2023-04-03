target_sources(omptarget.devicertl PRIVATE
  Configuration.cpp
  Debug.cpp
  HostRPC.cpp
  GenericAllocator.cpp
  Kernel.cpp
  LibC.cpp
  Mapping.cpp
  Misc.cpp
  Parallelism.cpp
  Reduction.cpp
  State.cpp
  Synchronization.cpp
  Tasking.cpp
  Utils.cpp
  WarpAllocator.cpp
  Workshare.cpp)
