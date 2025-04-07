CXX=clang++
CXXFLAGS=-std=c++20 -O3 -Wall -Wextra
CPPFLAGS=-I./include -D__HIP_PLATFORM_AMD__

GPUARCH?=gfx942

RT_OBJS=objsan_ir_rt.o objsan_rt.o
PL_OBJS=objsan_preload.o objsan_preload_impl_hip.o objsan_preload_hip.o
OBJS=$(RT_OBJS) $(PL_OBJS)

all: libobjsan_preload_hip.so

objsan_preload.o: preload/objsan_preload.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $< -c -o $@

objsan_preload_impl_hip.o: preload/objsan_preload_impl_hip.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x hip --offload-arch=$(GPUARCH) -fgpu-rdc -foffload-lto --offload-new-driver $< -c -o $@

objsan_preload_hip.o: preload/objsan_preload_hip.cpp
	$(CXX) $(CPPFLAGS) -I$(ROCM_PATH)/include $(CXXFLAGS) -fPIC $< -c -o $@

objsan_ir_rt.o: objsan_ir_rt.cpp
	$(CXX) $(CPPFLAGS) -D__OBJSAN_DEVICE__ $(CXXFLAGS) --target=amdgcn-amd-amdhsa -march=$(GPUARCH) -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -flto $< -c -o $@

objsan_rt.o: objsan_rt.cpp
	$(CXX) $(CPPFLAGS) -D__OBJSAN_DEVICE__ $(CXXFLAGS) --target=amdgcn-amd-amdhsa -march=$(GPUARCH) -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -flto $< -c -o $@

libobjsan_preload_hip.so: $(PL_OBJS) $(RT_OBJS)
	$(CXX) -shared -foffload-lto --offload-new-driver --offload-link --offload-arch=$(GPUARCH) -Xoffload-linker objsan_rt.o -Xoffload-linker objsan_ir_rt.o $(PL_OBJS) -o $@ -ldl -lamdhip64 -L$(ROCM_PATH)/lib

clean:
	rm -f $(OBJS)
