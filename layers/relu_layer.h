#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layers.h"

namespace layers {
  class relu : public Layer
  {

    public:
      cudnnHandle_t handle;
      cudnnTensorDescriptor_t input_descriptor;
      cudnnTensorDescriptor_t output_descriptor;
      cudnnActivationDescriptor_t activation_descriptor;

      relu(cudnnHandle_t cudnn,int batch_size,int input_channels,int input_height,int input_width);
      int get_output_shape_and_bytes(int shape[]);
      int get_input_shape_and_bytes(int shape[]);
      void forward(float* d_input, float * d_output);
      void backward(float * d_input, float *d_output, float *d_diffinput, float *d_diffoutput);
  };
}
#endif
