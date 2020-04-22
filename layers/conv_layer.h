#ifndef CONV_LAYER_H_
#define CONV_LAYER_H_

#include "layers.h"

namespace layers {

  /*! \class ConvLayer
   \brief Convolution Layer

   This class is used to create a convolution layer.
  */
  class ConvLayer : public layers::Layer
  {
    public:
      cudnnHandle_t handle; /*!< CUDNN Handle. */
      cudnnTensorDescriptor_t input_descriptor; /*!< cudnnTensor descriptor for input data to this layer. */
      cudnnFilterDescriptor_t kernel_descriptor; /*!< cudnnTensor descriptor for kernel of this layer. */
      cudnnConvolutionDescriptor_t convolution_descriptor; /*!< cudnn convolution operation descriptor. */
      cudnnTensorDescriptor_t output_descriptor; /*!< cudnnTensor descriptor for output data from this layer. */
      cudnnTensorDescriptor_t filter_derivative_descriptor; /*!< cudnnTensor descriptor for filter gradients of this layer. */
      cudnnConvolutionFwdAlgo_t convolution_algorithm; /*!< cudnn descriptor for convolution forward pass algorithm. */
      cudnnConvolutionBwdFilterAlgo_t filter_algo; /*!< cudnn descriptor for backward pass filter algorithm. */
      cudnnConvolutionBwdDataAlgo_t data_algo; /*!< cudnn descriptor for backward pass data algorithm. */
      size_t forward_workspace_bytes; /*!< Workspace bytes for forward pass of convolution layer. */
      size_t backward_workspace_bytes; /*!< Workspace bytes for backward pass of convolution layer. */
      int ikernel_width; /*!< Kernel width */
      int ikernel_height; /*!< Kernel height */

    /**
     * Intializes Convolution Layer Class.
     * @param cudnn CUDNN handle
     * @param batch_size Input batch size.
     * @param input_height Input data height. 
     * @param input_width Input data width.
     * @param input_channels Input data channels.
     * @param kernel_height Kernel height.
     * @param kernel_width Kernel width.
     * @param output_channels No of channels in the output.
     * @param pad Type of padding.
     */
    ConvLayer(cudnnHandle_t cudnn,
                    int batch_size,
                    int input_height,
                    int input_width,
                    int input_channels,
                    int kernel_height,
                    int kernel_width,
                    int output_channels,
                    padding_type pad
             );

    /**
     * Return the no of bytes occupied by the input and set the shape of the input.
     * @param shape Shape of the input is set in this array.
     */
    int get_input_shape_and_bytes(int shape[]);

    /**
     * Return the no of bytes occupied by the parameters and set the shape of the kernel.
     * @param shape Shape of the kernel is set in this array.
     */
    int get_params_shape_and_bytes(int shape[]);

    /**
     * Return the workspace size for forward pass.
     */
    size_t get_forward_workspace_bytes();
    
    /**
     * Return the workspace size for backward pass.
     */
    size_t get_backward_workspace_bytes();

    /**
     * Return the total workspace size.
     */
    size_t get_total_workspace_size();

    /**
     * Forward pass 
     * @param alpha alpha is multiplied to the output before writing it in the d_output (Set to 1 usually).
     * @param beta beta is multiplied to the d_output buffer before the output of the convolution is added to the d_output buffer. (Usually set to 0).
     * @param d_input Input buffer pointer.
     * @param d_kernel Kernel buffer pointer.
     * @param d_workspace workspace pointer.
     * @param d_output output buffer pointer.
     */
    void forward(float alpha,
      float beta,
      float* d_input,
      float* d_kernel,
      void* d_workspace,
      float * d_output
    );

    /**
     * Backward pass 
     * @param alpha alpha is multiplied to the computed values before writing it in the buffers (Set to 1 usually).
     * @param beta_filter beta_filter is multiplied to the params gradient before the gradient is added to the d_dkernel buffer. (Usually set to 0, set to 1 to accumulate gradients).
     * @param beta_data beta_data is multiplied to the input gradient before the gradient is added to the d_dx buffer. (Usually set to 0, set to 1 to accumulate gradients).
     * @param d_dy Output gradients buffer pointer.
     * @param d_workspace workspace pointer.
     * @param d_kernel Kernel buffer poitner.
     * @param d_x Input buffer pointer.
     * @param d_dx Input gradients buffer pointer.
     * @param d_dkernel Kernel gradient buffer pointer.
     * @param lr Learning rate.
     */
    void backward(float alpha,
      float beta_filter,
      float beta_data,
      float* d_dy,
      void* d_workspace,
      float* d_kernel,
      float* d_x,
      float* d_dx,
      float* d_dkernel,
      float lr
    );

    /**
     * Randomly generate and populate filter params.
     * @param d_kernel kernel buffer pointer.
     */
    void populate_filter_params(float *d_kernel);

    /**
     * Update params by applying gradients.
     * @param d_kernel Pointer to kernel buffer.
     * @param d_diffkernel Pointer to kernel gradients buffer.
     * @param lr Learning rate.
     * @param compute_stream Cuda stream for computations.
     */
    void update_weights(float* d_kernel, float* d_diffkernel, float lr,cudaStream_t compute_stream);

    /**
     * Return the no of bytes occupied by the output and set the shape of the output in the passed array pointer.
     * @param shape Shape of the output is set in this array.
     */
    int get_output_shape_and_bytes(int shape[]);

    /**
     * Returns the total memory occupied by Convolution Layer.
     */
    int get_total_memory();

    /**
     * Destructor
     */
    ~ConvLayer();

  };
}
#endif
