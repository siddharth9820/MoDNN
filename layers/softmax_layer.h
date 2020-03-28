#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

#include "layers.h"

namespace layers {
 class Softmax : public layers::Layer
 {

  public:
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnTensorDescriptor_t diff_descriptor;


    Softmax(cudnnHandle_t cudnn,int batch_size,int input_height);
    int get_output_shape_and_bytes(int shape[]);
    int get_input_shape_and_bytes(int shape[]);
    void forward(float* d_input, float * d_output);
    void backward(const int *label, float *diff, float * output);

 };
}
#endif