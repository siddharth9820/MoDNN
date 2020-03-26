home = /content/src
all: main.o network.o layers.o kernels.o
	nvcc -arch=sm_35 -std=c++11 -o test main.o network.o layers.o kernels.o -lcudnn -lcublas -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

main.o: $(home)/main.cu $(home)/layers.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/main.cu


network.o: $(home)/network.cu $(home)/layers.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/network.cu


layers.o: $(home)/layers.cu $(home)/layers.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/layers.cu

kernels.o: $(home)/kernels.cu
	 nvcc -c -arch=sm_35 -std=c++11 $(home)/kernels.cu

clean:
	rm main.o network.o layers.o kernels.o test
