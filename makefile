# For windows && opencv 4.2
# nvcc -I ..\opencv\build\include -L ..\opencv\build\x64\vc15\lib main.cu network.cu layers/layers.cu layers/pooling_layer.cu layers/input_layer.cu layers/conv_layer.cu layers/fc_layer.cu layers/softmax_layer.cu layers/flatten_layer.cu kernels/softmax_kernel.cu  -lcudnn -lcublas -o test -lopencv_world420
layers = conv_layer.o fc_layer.o flatten_layer.o input_layer.o layers.o pooling_layer.o softmax_layer.o
kernels = fc_kernel.o transpose_kernel.o softmax_kernel.o update_kernel.o
home = /content/src
cc = nvcc
flags = -arch=sm_35 -std=c++11
nvidia_flags = -lcudnn -lcublas
opencv_flags = -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

all: main.o network.o $(layers) $(kernels)
	$(cc) $(flags) -o test main.o network.o $(layers) $(kernels) $(nvidia_flags) $(opencv_flags)

main.o: $(home)/main.cu $(home)/layers/layers.h
	$(cc) -c $(flags) $(home)/main.cu

network.o:  $(home)/network.cu $(home)/layers/layers.h
	$(cc) -c $(flags) $(home)/network.cu

$(layers): %.o: $(home)/layers/%.cu $(home)/layers/layers.h
	$(cc) -c $(flags) $< -o $@

$(kernels): %.o: $(home)/kernels/%.cu $(home)/layers/layers.h
	$(cc) -c $(flags) $< -o $@


clean:
	rm $(layers) $(kernels) test
