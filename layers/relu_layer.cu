#include "relu_layer.h"

using namespace layers;

relu::relu(cudnnHandle_t cudnn,int batch_size,int input_channels,int input_height,int input_width)
{
  handle = cudnn;
  ibatch_size = obatch_size = batch_size;
  ichannels = ochannels = input_channels;
  iheight = oheight = input_height;
  iwidth = owidth = input_width;

  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnCreateActivationDescriptor(&activation_descriptor);
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,ibatch_size,ichannels,iheight,iwidth));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,obatch_size,ochannels,oheight,owidth));
  checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0));
}

int relu::get_output_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = ochannels;
  shape[2] = oheight;
  shape[3] = owidth;

  return obatch_size*ochannels*owidth*oheight*sizeof(float);
}

int relu::get_input_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = ochannels;
  shape[2] = oheight;
  shape[3] = owidth;

  return obatch_size*ochannels*owidth*oheight*sizeof(float);
}

void relu::forward(float* d_input, float * d_output)
{
  float alpha = 1.0;
  float beta = 0.0;
  checkCUDNN(cudnnActivationForward(handle,
			 activation_descriptor,
			 &alpha,
			 input_descriptor,
			 d_input,
			 &beta,
			 output_descriptor,
			 d_output));
}

void relu::backward(float * d_input, float *d_output, float *d_diffinput, float *d_diffoutput)
{
    float alpha = 1.0;
    float beta = 0.0;
    checkCUDNN(cudnnActivationBackward(handle,
      activation_descriptor,
      &alpha,
      output_descriptor,
      d_output,
      output_descriptor,
      d_diffoutput,
      input_descriptor,
      d_input,
      &beta,
      input_descriptor,
      d_diffinput
    ));
}
int relu::get_total_memory()
{
  int shape[4];
  return get_output_shape_and_bytes(shape);
}
