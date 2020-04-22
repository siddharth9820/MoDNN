# For windows && opencv 4.2
# nvcc -w -I ..\opencv\build\include -L ..\opencv\build\x64\vc15\lib main.cu network.cu layers/layers.cu layers/pooling_layer.cu layers/relu_layer.cu layers/input_layer.cu layers/conv_layer.cu layers/fc_layer.cu layers/softmax_layer.cu layers/flatten_layer.cu kernels/softmax_kernel.cu kernels/update_kernel.cu kernels/fc_kernel.cu kernels/transpose_kernel.cu mnist_dataset/mnist.cpp data_core/data_loader.cu vmm/vmm.cpp trainer/utils.cu trainer/full_mem_trainer.cu trainer/min_mem_trainer.cu trainer/prefetching_heuristic_fetch_next.cu trainer/prefetching_heuristic_half_window.cu -lcudnn -lcublas -o test -lopencv_world420
home = /content/src
layers = conv_layer.o fc_layer.o flatten_layer.o input_layer.o layers.o pooling_layer.o softmax_layer.o relu_layer.o
layers_headers = $(home)/layers/conv_layer.h $(home)/layers/fc_layer.h $(home)/layers/flatten_layer.h $(home)/layers/input_layer.h $(home)/layers/layers.h $(home)/layers/pooling_layer.h $(home)/layers/softmax_layer.h $(home)/layers/relu_layer.h
kernels = fc_kernel.o transpose_kernel.o softmax_kernel.o update_kernel.o
trainers = full_mem_trainer.o min_mem_trainer.o prefetching_heuristic_half_window.o prefetching_heuristic_fetch_next.o

cc = nvcc
flags = -arch=sm_35 -std=c++11
nvidia_flags = -lcudnn -lcublas
opencv_flags = -lopencv_imgcodecs -lopencv_imgproc -lopencv_core




all: main.o network.o $(layers) $(kernels) mnist.o data_loader.o vmm.o $(trainers) utils.o
	$(cc) $(flags) -o test main.o network.o $(layers) $(kernels) $(trainers) mnist.o data_loader.o vmm.o utils.o $(nvidia_flags) $(opencv_flags)

main.o: $(home)/main.cu $(home)/layers/layers.h $(home)/mnist_dataset/mnist.h $(home)/data_core/data_loader.h $(home)/trainer/trainer.h
	$(cc) -c $(flags) $(home)/main.cu

network.o:  $(home)/network.cu $(layer_headers) $(home)/vmm/vmm.h
	$(cc) -c $(flags) $(home)/network.cu

$(layers): %.o: $(home)/layers/%.cu $(home)/layers/%.h
	$(cc) -c $(flags) $< -o $@

$(kernels): %.o: $(home)/kernels/%.cu $(home)/layers/layers.h
	$(cc) -c $(flags) $< -o $@

$(trainers): %.o: $(home)/trainer/%.cu $(home)/trainer/trainer.h
	$(cc) -c $(flags) $< -o $@

mnist.o : $(home)/mnist_dataset/mnist.cpp $(home)/mnist_dataset/mnist.h $(home)/data_core/dataset.h
	$(cc) -c $(flags) $(home)/mnist_dataset/mnist.cpp

data_loader.o : $(home)/data_core/data_loader.cu $(home)/data_core/data_loader.h
	$(cc) -c $(flags) $(home)/data_core/data_loader.cu

vmm.o : $(home)/vmm/vmm.cpp $(home)/vmm/vmm.h
	$(cc) -c $(flags) $(home)/vmm/vmm.cpp


utils.o : $(home)/trainer/utils.cu $(home)/trainer/trainer.h
	$(cc) -c $(flags) $(home)/trainer/utils.cu

clean:
	rm $(layers) $(kernels) test mnist.o data_loader.o vmm.o
