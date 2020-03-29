#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_  

#include "layers.h"

namespace layers{

class PoolingLayer : public layers::Layer {
	private:
		cudnnHandle_t* handle_;
		cudnnTensorDescriptor_t input_descriptor;
		cudnnTensorDescriptor_t output_descriptor;
		cudnnPoolingDescriptor_t pooling_descriptor;
		size_t forward_workspace_bytes, backward_workspace_bytes;
	public:
		PoolingLayer(cudnnHandle_t* handle, 
					int window_height, 
					int window_width, 
					int vertical_stride,
					int horizontal_stride,
					int batch_size,
                    int input_height,
                    int input_width,
                    int input_channels,
					padding_type pad, 
					cudnnPoolingMode_t mode
				);
		void forward(float alpha, float beta, float* d_input, float* d_output);
		void backward(float alpha, float beta, float* d_y, float* d_dy, float* d_x, float* d_dx);
		int get_output_shape_and_bytes(int shape[]);
		~PoolingLayer();
	};
}

#endif