#ifndef LAYER_H_
#define LAYER_H_

#include <cudnn.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <cublas_v2.h>
#include <cassert>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <random>
#include <cuda.h>
#include <fstream>

enum padding_type{
  SAME,
  VALID
};

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

static const char *cublasGetErrorString(cublasStatus_t error);

#define checkCUBLAS(expression)                              \
  {                                                          \
    cublasStatus_t status = (expression);                     \
    std::cerr << "Doing a CUBLAS OP" <<" ";                   \
    std::cerr << status <<" "<<CUBLAS_STATUS_SUCCESS<< std::endl;\
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cublasGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


#define MU 0
#define SIGMA 0.1


std::map<std::string,float*> init_buffer_map();
__global__ void SoftmaxLossBackprop(const int *label, int num_labels, int batch_size, float *diff);
int calc_bytes_from_shape(int shape[]);

namespace layers
{

  class Layer
  {
    public:
      int obatch_size,ochannels,oheight,owidth;
      int ibatch_size,ichannels,iheight,iwidth;
      void forward();

  };

 class FCLayer : public Layer
 {
  public:
    cublasHandle_t handle;
    FCLayer(cublasHandle_t cublas,int batch_size,int input_height,int output_height);
    int get_output_shape_and_bytes(int shape[]);
    void forward(float* d_input, float * d_kernel, float * d_output); //checked with numpy and working correctly
    void backward(float *d_input, float* d_kernel,float *d_diffkernel,float *d_diffinput, float *d_diffoutput); //checked with numpy for parameter gradient
    int allocate_internal_mem(float **d_kernel,float **d_diffkernel);
    void populate_filter_params(float *d_kernel);
    int get_input_shape_and_bytes(int shape[]);
    int get_params_shape_and_bytes(int shape[]);

 };
 class Flatten : public Layer
 {
  public:
   Flatten(int batch_size,int input_height,int input_width,int input_channels);
   int get_output_shape_and_bytes(int shape[]);
   int get_input_shape_and_bytes(int shape[]);
 };

 class Softmax : public Layer
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

namespace network
{
  class seqNetwork
  {
    public:
      int num_layers;
      std::vector<std::vector<std::string > > layer_info;
      std::vector<std::map<std::string,float*> > layer_buffers;
      std::vector<std::map<std::string,float*> > layer_offloaded_buffers;
      std::vector< layers::Layer *> layer_objects;


      cudnnHandle_t handle;
      cublasHandle_t blas_handle;


      seqNetwork(cudnnHandle_t cudnn,cublasHandle_t cublas,std::vector<std::string> &specs);
      void print_network_info();
      void allocate_memory();
      void get_output_shape(int shape[], int i);
      void randomise_batch(); //randomise input to the neural network
      void enqueue_batch(float * batch);
      void randomise_params();
      void forward();
      void backward();



      float* offload_buffer(int layer_number,std::string type,int shape[]); //type is one of "output","workspace","input"
      void prefetch_buffer(int layer_number,std::string type);
      ~seqNetwork();

  };
}

#endif