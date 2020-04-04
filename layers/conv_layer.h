#ifndef CONV_LAYER_H_
#define CONV_LAYER_H_

#include "layers.h"

namespace layers {

  class ConvLayer : public layers::Layer
  {
    private:
      void reset_gradients(float* d_dkernel);
    public:
      cudnnHandle_t handle;
      cudnnTensorDescriptor_t input_descriptor;
      cudnnFilterDescriptor_t kernel_descriptor;
      cudnnConvolutionDescriptor_t convolution_descriptor;
      cudnnTensorDescriptor_t output_descriptor;
      cudnnTensorDescriptor_t filter_derivative_descriptor;
      cudnnConvolutionFwdAlgo_t convolution_algorithm;
      cudnnConvolutionBwdFilterAlgo_t filter_algo;
      cudnnConvolutionBwdDataAlgo_t data_algo;
      size_t forward_workspace_bytes, backward_workspace_bytes;
      int ikernel_width,ikernel_height;



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
    int get_input_shape_and_bytes(int shape[]);
    int get_params_shape_and_bytes(int shape[]);
    size_t get_forward_workspace_bytes();
    size_t get_backward_workspace_bytes();
    size_t get_total_workspace_size();
    void forward(float alpha,
      float beta,
      float* d_input,
      float* d_kernel,
      void* d_workspace,
      float * d_output
    );
    void backward(float alpha,
      float beta,
      float* d_y,
      float* d_dy,
      void* d_workspace,
      float* d_kernel,
      float* d_x,
      float* d_dx,
      float* d_dkernel,
      float lr
    );
    int allocate_internal_mem(float **d_kernel, void **d_workspace,float **d_diffkernel);
    void populate_filter_params(float *d_kernel);
    void update_weights(float* d_kernel, float* d_diffkernel, float lr);
    int get_output_shape_and_bytes(int shape[]);

    ~ConvLayer();

  };
}
#endif
