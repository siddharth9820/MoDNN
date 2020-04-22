#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_

#include "layers.h"

namespace layers{

/*! \class Pooling
    \brief Pooling Layer

	This class is used to create a Pooling layer.
*/ 
class PoolingLayer : public layers::Layer {
	private:

		cudnnHandle_t* handle_; /*!< CUDNN Handle. */
		cudnnTensorDescriptor_t input_descriptor; /*!< cudnnTensor descriptor for input data to this layer. */
		cudnnTensorDescriptor_t output_descriptor; /*!< cudnnTensor descriptor for output data from this layer. */
		cudnnPoolingDescriptor_t pooling_descriptor; /*!< cudnn descriptor for pooling operation. */
		size_t forward_workspace_bytes; /*!< Bytes occupied by workspace of forward pass. */
		size_t backward_workspace_bytes; /*!< Bytes occupied by workspace of backward pass. */
	public:

		/**
		 * Intializes Pooling Layer Class.
		 * @param cudnn CUDNN handle.
		 * @param window_height Pooling window height.
		 * @param window_width Pooling window width.
		 * @param vertical_stride Pooling window stride in vertical direction.
		 * @param horizontal_stride Pooling window stride in horizontal direction.
		 * @param batch_size Input batch size.
		 * @param input_height Input data height. 
		 * @param input_width Input data width.
		 * @param input_channels Input data channels.
		 * @param pad Type of padding.
		 * @param mode Mode of pooling (average, max).
		 */
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

		/**
		 * Forward pass 
		 * @param alpha alpha is multiplied to the output before writing it in the d_output (Set to 1 usually).
		 * @param beta beta is multiplied to the d_output buffer before the output of the pooling is added to the d_output buffer. (Usually set to 0).
		 * @param d_input Input buffer pointer.
		 * @param d_output output buffer pointer.
		 */
		void forward(float alpha, float beta, float* d_input, float* d_output);

		/**
		 * Backward pass 
		 * @param alpha alpha is multiplied to the computed values before writing it in the buffers (Set to 1 usually).
		 * @param beta beta is multiplied to the d_dx buffer before the input gradient of the pooling is added to the d_dx buffer. (Usually set to 0, set to 1 to accumulate gradients).
		 * @param d_y Output buffer pointer.
		 * @param d_dy Output gradients buffer pointer.
		 * @param d_x Input buffer pointer.
		 * @param d_dx Input gradients buffer pointer.
		 */
		void backward(float alpha, float beta, float* d_y, float* d_dy, float* d_x, float* d_dx);

		/**
		 * Return the no of bytes occupied by the output and set the shape of the output in the passed array.
		 * @param shape Shape of the output is set in this array.
		 */
		int get_output_shape_and_bytes(int shape[]);

		/**
		 * Return the no of bytes occupied by the intput and set the shape of the input in the passed array.
		 * @param shape Shape of the input is set in this array.
		 */
		int get_input_shape_and_bytes(int shape[]);

		/**
		 * Return the total memory occupied by Pooling Layer.
		 */
		int get_total_memory();

		/**
		 * Destructor
		 */
		~PoolingLayer();
	};
}

#endif
