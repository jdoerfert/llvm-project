CXX=clang++
CXXFLAGS=-std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
CPPFLAGS=-I./include -D__HIP_PLATFORM_AMD__

GPUARCH?=gfx942

OBJSAN_IR_CXXFLAGS=

RT_OBJS=objsan_ir_rt.o objsan_rt.o
PL_OBJS=objsan_preload.o objsan_preload_impl_hip.o objsan_preload_hip.o
OBJS=$(RT_OBJS) $(PL_OBJS) example.hip.o

PL_LIB=libobjsan_preload_hip.a

all: $(PL_LIB) example.hip

objsan_preload.o: preload/objsan_preload.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $< -c -o $@

objsan_preload_impl_hip.o: preload/objsan_preload_impl_hip.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x hip --offload-arch=$(GPUARCH) -fgpu-rdc -foffload-lto --offload-new-driver $< -c -o $@

objsan_preload_hip.o: preload/objsan_preload_hip.cpp
	$(CXX) $(CPPFLAGS) -I$(ROCM_PATH)/include $(CXXFLAGS) -fPIC $< -c -o $@

objsan_ir_rt.o: objsan_ir_rt.cpp
	$(CXX) $(CPPFLAGS) -D__OBJSAN_DEVICE__ $(CXXFLAGS) --target=amdgcn-amd-amdhsa -march=$(GPUARCH) -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -Wno-unknown-cuda-version -flto $< -c -o $@

objsan_rt.o: objsan_rt.cpp
	$(CXX) $(CPPFLAGS) -D__OBJSAN_DEVICE__ $(CXXFLAGS) --target=amdgcn-amd-amdhsa -march=$(GPUARCH) -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -Wno-unknown-cuda-version -flto $< -c -o $@

libobjsan_preload_hip.a: $(PL_OBJS) $(RT_OBJS)
	$(CXX) -I$(ROCM_PATH)/include --emit-static-lib -foffload-lto --offload-new-driver --offload-arch=$(GPUARCH) $(PL_OBJS) -o $@

example.hip.o: examples/example.hip.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fsanitize=object -mllvm -objsan-gpu-only=1 -mllvm -objsan-runtime-bitcode=objsan_ir_rt.bc -x hip --offload-arch=$(GPUARCH) -fgpu-rdc -foffload-lto --offload-new-driver $< -c -o $@

example.hip: example.hip.o objsan_ir_rt.bc objsan_rt.o $(PL_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fsanitize=object --offload-arch=$(GPUARCH) -fgpu-rdc -foffload-lto --offload-new-driver --offload-link -Xoffload-linker objsan_rt.o $< libobjsan_preload_hip.a -o $@ -lamdhip64 -L$(ROCM_PATH)/lib

objsan_ir_rt.bc: objsan_ir_rt.cpp
	$(CXX) -D__OBJSAN_DEVICE__ $(CXXFLAGS) $(OBJSAN_IR_CXXFLAGS) --target=amdgcn-amd-amdhsa -march=$(GPUARCH) -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -Wno-unknown-cuda-version -c -emit-llvm $< -o $@

clean:
	rm -f $(OBJS) $(PL_LIB) example.hip objsan_ir_rt.bc
