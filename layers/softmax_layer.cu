#include "softmax_layer.h"

using namespace layers;

Softmax::Softmax(cudnnHandle_t cudnn,int batch_size,int input_height)
{
  handle = cudnn;
  ibatch_size = obatch_size = batch_size;
  iheight = oheight = input_height;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&diff_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptorEx(input_descriptor,CUDNN_DATA_FLOAT,obatch_size,oheight,1,1,oheight,1,1,1));
  checkCUDNN(cudnnSetTensor4dDescriptorEx(output_descriptor,CUDNN_DATA_FLOAT,obatch_size,oheight,1,1,oheight,1,1,1));
  checkCUDNN(cudnnSetTensor4dDescriptorEx(diff_descriptor,CUDNN_DATA_FLOAT,obatch_size,oheight,1,1,oheight,1,1,1));

}

int Softmax::get_output_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

int Softmax::get_input_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

void Softmax::forward(float* d_input, float * d_output)
{
  float alpha = 1.0;
  float beta = 0.0;
  checkCUDNN(cudnnSoftmaxForward(handle,
                      CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_CHANNEL,
                      &alpha,
                      input_descriptor,
                      d_input,
                      &beta,
                      output_descriptor,
                      d_output));

}


void Softmax::backward(const int *label, float *diff, float * output)
{
  cudaMemcpy(diff,output,obatch_size*oheight*sizeof(float),cudaMemcpyDeviceToDevice);
  SoftmaxLossBackprop<<<(obatch_size+255)/256, 256>>>(label, oheight, obatch_size, diff);
}
