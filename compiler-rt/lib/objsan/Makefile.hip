CXX=clang++
CXXFLAGS=-std=c++20 -O3 -Wall -Wextra
CPPFLAGS=-I./include

RT_OBJS=objsan_ir_rt.o objsan_rt.o
PL_OBJS=objsan_preload.o objsan_preload_impl_hip.o objsan_preload_hip.o
OBJS=$(RT_OBJS) $(PL_OBJS)

all: libobjsan_preload_hip.so

objsan_preload.o: preload/objsan_preload.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $< -c -o $@

objsan_preload_impl_hip.o: preload/objsan_preload_impl_hip.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x hip --offload-arch=gfx942 -fgpu-rdc -foffload-lto $< -c -o $@

objsan_preload_hip.o: preload/objsan_preload_hip.cpp
	$(CXX) $(CPPFLAGS) -I$(ROCM_PATH)/include $(CXXFLAGS) -fPIC $< -c -o $@

objsan_ir_rt.o: objsan_ir_rt.cpp
	$(CXX) $(CPPFLAGS) -D__OBJSAN_DEVICE__ $(CXXFLAGS) --target=amdgcn-amd-amdhsa -march=gfx942 -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -flto $< -c -o $@

objsan_rt.o: objsan_rt.cpp
	$(CXX) $(CPPFLAGS) -D__OBJSAN_DEVICE__ $(CXXFLAGS) --target=amdgcn-amd-amdhsa -march=gfx942 -nogpulib -nostdlibinc -fno-rtti -fno-exceptions -fconvergent-functions -flto $< -c -o $@

libobjsan_preload_hip.so: $(PL_OBJS) $(RT_OBJS)
	$(CXX) -I$(ROCM_PATH)/include -shared -foffload-lto --offload-link --offload-arch=sm_70 -Xoffload-linker objsan_rt.o -Xoffload-linker objsan_ir_rt.o $(PL_OBJS) -o $@ -ldl -lamdhip64 -L$(ROCM_PATH)/lib64

clean:
	rm -f $(OBJS)
