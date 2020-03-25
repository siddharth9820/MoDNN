all:
	nvcc -arch=sm_35 -std=c++11 -O2 /content/src/main.cu /content/src/network.cu /content/src/layers.cu  -lcudnn -lcublas -o test -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
