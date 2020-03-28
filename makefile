# For windows && opencv 4.2
# nvcc -I ..\opencv\build\include -L ..\opencv\build\x64\vc15\lib main.cu network.cu layers/layers.cu layers/pooling_layer.cu layers/input_layer.cu layers/conv_layer.cu layers/fc_layer.cu layers/softmax_layer.cu layers/flatten_layer.cu kernels/softmax_kernel.cu  -lcudnn -lcublas -o test -lopencv_world420

home = /content/src
all: main.o network.o layers.o kernels.o
	nvcc -arch=sm_35 -std=c++11 -o test main.o network.o layers.o kernels.o -lcudnn -lcublas -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

main.o: $(home)/main.cu $(home)/layers/layers.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/main.cu


network.o: $(home)/network.cu $(home)/layers/layers.h $(home)/layers/input_layer.h $(home)/layers/pooling_layer.h $(home)/layers/conv_layer.h $(home)/layers/fc_layer.h $(home)/layers/flatten_layer.h $(home)/layers/softmax_layer.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/network.cu


layers.o: $(home)/layers/layers.cu $(home)/layers/layers.h $(home)/layers/input_layer.h $(home)/layers/pooling_layer.h $(home)/layers/conv_layer.h $(home)/layers/fc_layer.h $(home)/layers/flatten_layer.h $(home)/layers/softmax_layer.h
	nvcc -c -arch=sm_35 -std=c++11 $(home)/layers/layers.cu $(home)/layers/input_layer.cu $(home)/layers/pooling_layer.cu $(home)/layers/conv_layer.cu $(home)/layers/fc_layer.cu $(home)/layers/flatten_layer.cu $(home)/layers/softmax_layer.cu

kernels.o: $(home)/kernel/softmax_kernel.cu
	 nvcc -c -arch=sm_35 -std=c++11 $(home)/kernel/softmax_kernel.cu

clean:
	rm main.o network.o layers.o kernels.o test
