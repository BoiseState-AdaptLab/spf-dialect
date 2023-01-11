add_library(CPURuntime
  SHARED
	CPURuntime.cpp
)

add_library(GPURuntime
  SHARED
	GPURuntime.cu
	CPURuntime
)