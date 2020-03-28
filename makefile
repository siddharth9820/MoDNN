# For windows && opencv 4.2
# nvcc -I ..\opencv\build\include -L ..\opencv\build\x64\vc15\lib main.cu network.cu layers.cu pooling_layer.cu input_layer.cu kernels.cu  -lcudnn -lcublas -o test -lopencv_world420

home = /content/src
all: main.o network.o layers.o kernels.o
	nvcc -arch=sm_35 -std=c++11 -o test main.o network.o layers.o kernels.o -lcudnn -lcublas -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

main.o: $(home)/main.cu $(home)/layers.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/main.cu


network.o: $(home)/network.cu $(home)/layers.h $(home)/input_layer.h $(home)/pooling_layer.h $(home)/conv_layer.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/network.cu


layers.o: $(home)/layers.cu $(home)/layers.h $(home)/input_layer.h $(home)/pooling_layer.h $(home)/conv_layer.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/layers.cu $(home)/input_layer.cu $(home)/pooling_layer.cu $(home)/conv_layer.cu

kernels.o: $(home)/kernels.cu
	 nvcc -c -arch=sm_35 -std=c++11 $(home)/kernels.cu

clean:
	rm main.o network.o layers.o kernels.o test
