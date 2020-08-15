# build C version of nbody simulator
GPUCC = nvcc
CPUCC = nvcc
NVCCFLAGS = -g -dc
CFLAGS = -g
HEADERDIRS = src

default: all

all: cu_nbmath.o cu_cphys.o cu_utils.o
	$(CPUCC) $(CFLAGS) -o CUDAIntegrate -L /usr/local/cuda/lib64 -lcuda -lcudart -I $(HEADERDIRS) $(HEADERDIRS)/Nbody.cu cu_nbmath.o cu_cphys.o cu_utils.o
	@mkdir -p bin
	mv *.o bin/
	mv CUDAIntegrate bin/

cu_nbmath.o: $(HEADERDIRS)/cu_nbmath.cu $(HEADERDIRS)/cu_nbmath.cuh
	$(GPUCC) $(NVCCFLAGS) -I $(HEADERDIRS) --device-c $(HEADERDIRS)/cu_nbmath.cu

cu_cphys.o: $(HEADERDIRS)/cu_cphys.cu $(HEADERDIRS)/cu_cphys.cuh $(HEADERDIRS)/cu_nbmath.cuh
	$(GPUCC) $(NVCCFLAGS) -I $(HEADERDIRS) --device-c $(HEADERDIRS)/cu_cphys.cu

cu_utils.o: $(HEADERDIRS)/cu_utils.cu $(HEADERDIRS)/cu_utils.cuh
	$(GPUCC) $(NVCCFLAGS) -I $(HEADERDIRS) -c $(HEADERDIRS)/cu_utils.cu 	

clean:
	rm -r bin/
